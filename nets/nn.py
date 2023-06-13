import math

import torch


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 conv.dilation,
                                 conv.groups, True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, (k - 1) // 2, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1801.04381.pdf]
    """

    def __init__(self, in_ch, out_ch, activation, r, k, s, fused=True):
        super().__init__()
        identity = torch.nn.Identity()
        self.add = s == 1 and in_ch == out_ch

        if fused:
            modules = [Conv(in_ch, r * in_ch, activation, k, s),
                       Conv(r * in_ch, out_ch, identity) if r != 1 else identity]
        else:
            modules = [Conv(in_ch, r * in_ch, activation) if r != 1 else identity,
                       Conv(r * in_ch, r * in_ch, activation, k, s, r * in_ch),
                       Conv(r * in_ch, out_ch, identity)]

        self.res = torch.nn.Sequential(*modules)
        self.quant = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.quant.add(x, self.res(x)) if self.add else self.res(x)


class PIPNet(torch.nn.Module):
    def __init__(self, args, params,
                 mean_indices, reverse_index1, reverse_index2, max_len):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []
        filters = [3, 16, 24, 40, 80, 160, 1280]

        # p1/2
        self.p1.append(Conv(filters[0], filters[1], torch.nn.ReLU(inplace=True), 3, s=2, g=1))
        self.p1.append(Residual(filters[1], filters[1], torch.nn.ReLU(inplace=True), 1, 3, 1))
        # p2/4
        self.p2.append(Residual(filters[1], filters[2], torch.nn.ReLU(inplace=True), 3, 3, 2))
        self.p2.append(Residual(filters[2], filters[2], torch.nn.ReLU(inplace=True), 3, 3, 1))
        # p3/8
        self.p3.append(Residual(filters[2], filters[3], torch.nn.ReLU(inplace=True), 3, 3, 2))
        self.p3.append(Residual(filters[3], filters[3], torch.nn.ReLU(inplace=True), 3, 3, 1))
        # p4/16
        self.p4.append(Residual(filters[3], filters[4], torch.nn.ReLU(inplace=True), 6, 3, 2, False))
        self.p4.append(Residual(filters[4], filters[4], torch.nn.ReLU(inplace=True), 3, 3, 1, False))
        self.p4.append(Residual(filters[4], filters[4], torch.nn.ReLU(inplace=True), 3, 3, 1, False))
        self.p4.append(Residual(filters[4], filters[4], torch.nn.ReLU(inplace=True), 3, 3, 1, False))
        self.p4.append(Residual(filters[4], filters[4], torch.nn.ReLU(inplace=True), 6, 3, 1, False))
        self.p4.append(Residual(filters[4], filters[4], torch.nn.ReLU(inplace=True), 6, 3, 1, False))
        # p5/32
        self.p5.append(Residual(filters[4], filters[5], torch.nn.ReLU(inplace=True), 6, 5, 2, False))
        self.p5.append(Residual(filters[5], filters[5], torch.nn.ReLU(inplace=True), 6, 5, 1, False))
        self.p5.append(Residual(filters[5], filters[5], torch.nn.ReLU(inplace=True), 6, 5, 1, False))
        self.p5.append(Residual(filters[5], filters[5], torch.nn.ReLU(inplace=True), 6, 5, 1, False))
        self.p5.append(Conv(filters[5], filters[6], torch.nn.ReLU(inplace=True), k=1, s=1, g=1))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

        self.args = args
        self.params = params
        self.max_len = max_len
        self.mean_indices = mean_indices
        self.reverse_index1 = reverse_index1
        self.reverse_index2 = reverse_index2

        self.score = torch.nn.Conv2d(filters[-1], params['num_lms'], 1)
        self.offset_x = torch.nn.Conv2d(filters[-1], params['num_lms'], 1)
        self.offset_y = torch.nn.Conv2d(filters[-1], params['num_lms'], 1)
        self.neighbor_x = torch.nn.Conv2d(filters[-1], params['num_nb'] * params['num_lms'], 1)
        self.neighbor_y = torch.nn.Conv2d(filters[-1], params['num_nb'] * params['num_lms'], 1)

        torch.nn.init.normal_(self.score.weight, std=0.001)
        if self.score.bias is not None:
            torch.nn.init.constant_(self.score.bias, 0)

        torch.nn.init.normal_(self.offset_x.weight, std=0.001)
        if self.offset_x.bias is not None:
            torch.nn.init.constant_(self.offset_x.bias, 0)

        torch.nn.init.normal_(self.offset_y.weight, std=0.001)
        if self.offset_y.bias is not None:
            torch.nn.init.constant_(self.offset_y.bias, 0)

        torch.nn.init.normal_(self.neighbor_x.weight, std=0.001)
        if self.neighbor_x.bias is not None:
            torch.nn.init.constant_(self.neighbor_x.bias, 0)

        torch.nn.init.normal_(self.neighbor_y.weight, std=0.001)
        if self.neighbor_y.bias is not None:
            torch.nn.init.constant_(self.neighbor_y.bias, 0)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)

        score = self.score(x)
        offset_x = self.offset_x(x)
        offset_y = self.offset_y(x)
        neighbor_x = self.neighbor_x(x)
        neighbor_y = self.neighbor_y(x)

        return [score, offset_x, offset_y, neighbor_x, neighbor_y]

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class QAT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.de_quant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)

        for i in range(len(x)):
            x[i] = self.de_quant(x[i])
        if self.training:
            return x

        score = x[0]
        offset_x = x[1]
        offset_y = x[2]
        neighbor_x = x[3]
        neighbor_y = x[4]

        b, c, h, w = score.size()
        assert b == 1

        score = score.view(b * c, -1)
        max_idx = torch.argmax(score, 1).view(-1, 1)
        max_idx_neighbor = max_idx.repeat(1, self.model.params['num_nb']).view(-1, 1)

        offset_x = offset_x.view(b * c, -1)
        offset_y = offset_y.view(b * c, -1)
        offset_x_select = torch.gather(offset_x, 1, max_idx).squeeze(1)
        offset_y_select = torch.gather(offset_y, 1, max_idx).squeeze(1)

        neighbor_x = neighbor_x.view(b * self.model.params['num_nb'] * c, -1)
        neighbor_y = neighbor_y.view(b * self.model.params['num_nb'] * c, -1)
        neighbor_x_select = torch.gather(neighbor_x, 1, max_idx_neighbor)
        neighbor_y_select = torch.gather(neighbor_y, 1, max_idx_neighbor)
        neighbor_x_select = neighbor_x_select.squeeze(1).view(-1, self.model.params['num_nb'])
        neighbor_y_select = neighbor_y_select.squeeze(1).view(-1, self.model.params['num_nb'])

        offset_x = (max_idx % w).view(-1, 1).float() + offset_x_select.view(-1, 1)
        offset_y = (max_idx // w).view(-1, 1).float() + offset_y_select.view(-1, 1)
        offset_x /= 1.0 * self.model.args.input_size / self.model.params['stride']
        offset_y /= 1.0 * self.model.args.input_size / self.model.params['stride']

        neighbor_x = (max_idx % w).view(-1, 1).float() + neighbor_x_select
        neighbor_y = (max_idx // w).view(-1, 1).float() + neighbor_y_select
        neighbor_x = neighbor_x.view(-1, self.model.params['num_nb'])
        neighbor_y = neighbor_y.view(-1, self.model.params['num_nb'])
        neighbor_x /= 1.0 * self.model.args.input_size / self.model.params['stride']
        neighbor_y /= 1.0 * self.model.args.input_size / self.model.params['stride']

        # merge neighbor predictions
        neighbor_x = neighbor_x[self.model.reverse_index1, self.model.reverse_index2]
        neighbor_y = neighbor_y[self.model.reverse_index1, self.model.reverse_index2]
        neighbor_x = neighbor_x.view(self.model.params['num_lms'], self.model.max_len)
        neighbor_y = neighbor_y.view(self.model.params['num_lms'], self.model.max_len)

        offset_x = torch.mean(torch.cat((offset_x, neighbor_x), dim=1), dim=1).view(-1, 1)
        offset_y = torch.mean(torch.cat((offset_y, neighbor_y), dim=1), dim=1).view(-1, 1)

        return torch.flatten(torch.cat((offset_x, offset_y), dim=1))


class CosineLR:
    def __init__(self, lr, args, optimizer):
        self.lr = lr
        self.min = 1E-5
        self.max = 1E-4
        self.args = args
        self.warmup_epochs = 5

        for param_group in optimizer.param_groups:
            param_group['lr'] = self.max

    def step(self, epoch, optimizer):
        epochs = self.args.epochs
        if epoch < self.warmup_epochs:
            lr = self.max + epoch * (self.lr - self.max) / self.warmup_epochs
        else:
            epoch = epoch - self.warmup_epochs
            if epoch < epochs:
                alpha = math.pi * (epoch - (epochs * (epoch // epochs))) / epochs
                lr = self.min + 0.5 * (self.lr - self.min) * (1 + math.cos(alpha))
            else:
                lr = self.min

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
