import argparse
import copy
import csv
import os
import warnings

import cv2
import numpy
import torch
import tqdm
import yaml
from timm import utils
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

data_dir = '../Dataset/LMK'
warnings.filterwarnings("ignore")


def train(args, params):
    util.setup_seed()
    util.setup_multi_processes()
    # Model
    model = nn.PIPNet(args, params, *util.compute_indices(f'{data_dir}/indices.txt', params))
    model = util.load_weight('./weights/best.pt', model)

    # Fuse bn into conv
    model.eval()
    for m in model.modules():
        if type(m) is nn.Conv:
            modules_to_fuse = ["conv", "norm"]
            if type(m.relu) is torch.nn.ReLU:
                modules_to_fuse.append("relu")
            torch.ao.quantization.fuse_modules_qat(m, modules_to_fuse, inplace=True)

    model.train()

    model = nn.QAT(model)
    quantization_config = torch.quantization.get_default_qconfig("qnnpack")
    model.qconfig = quantization_config
    torch.quantization.prepare_qat(model, inplace=True)

    model.cuda()

    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)

    # Optimizer
    lr = 0.0005
    optimizer = torch.optim.Adam(util.weight_decay(model), lr)

    # Scheduler
    scheduler = nn.CosineLR(lr, args, optimizer)

    # Dataset
    sampler = None
    dataset = Dataset(f'{data_dir}/train.txt', args, params)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, False, sampler,
                             num_workers=8, pin_memory=True, drop_last=True)

    if args.distributed:
        # DDP mode
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # Start training
    best = float('inf')
    num_batch = len(loader)
    criterion = util.ComputeLoss(params)
    with open('weights/step.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'NME'])
            writer.writeheader()
        for epoch in range(args.epochs):

            p_bar = enumerate(loader)
            m_loss = util.AverageMeter()

            if args.distributed:
                sampler.set_epoch(epoch)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                p_bar = tqdm.tqdm(iterable=p_bar, total=num_batch)

            model.train()
            optimizer.zero_grad()

            for i, (samples, targets) in p_bar:
                samples = samples.cuda()
                # Forward
                loss = criterion(model(samples), targets)

                # Backward
                loss.backward()

                # Optimize
                if (i + num_batch * epoch) % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # Log
                if args.distributed:
                    loss = utils.reduce_tensor(loss.data, args.world_size)
                m_loss.update(loss.item(), samples.size(0))
                if args.local_rank == 0:
                    s = ('%10s' + '%10.4g') % (f'{epoch + 1}/{args.epochs}', m_loss.avg)
                    p_bar.set_description(s)

            # Scheduler
            scheduler.step(epoch + 1, optimizer)

            if args.local_rank == 0:
                # NME
                save = copy.deepcopy(model.module if args.distributed else model)
                save.eval()
                save.to(torch.device(torch.device('cpu')))
                torch.ao.quantization.convert(save, inplace=True)
                last = test(args, params, save)

                writer.writerow({'NME': str(f'{last:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                f.flush()

                # Update best NME
                if best > last:
                    best = last

                # Save last, best
                util.export_onnx(save, args, './weights/last.onnx')
                if best == last:
                    util.export_onnx(save, args, './weights/best.onnx')

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model):
    index = [36, 45]
    loader = data.DataLoader(Dataset(f'{data_dir}/test.txt', args, params, False))

    model.eval()

    nme_merge = []
    for sample, target in tqdm.tqdm(loader, '%20s' % 'NME'):
        sample = sample.float()

        output = model(sample)
        target = target.view(-1)
        target = target.cpu().numpy()

        a = target.reshape(-1, 2)
        b = target.reshape(-1, 2)
        norm = numpy.linalg.norm(a[index[0]] - b[index[1]])
        nme_merge.append(util.compute_nme(output, target, norm))

    # Print results
    nme = numpy.mean(nme_merge) * 100
    print('%20.3g' % nme)

    # Return results
    model.float()  # for training
    return nme


def demo(args, params):
    std = numpy.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)
    mean = numpy.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)

    face = util.FaceDetector('./weights/detection.onnx')

    landmark = util.LandmarkDetector('./weights/last.onnx')

    scale = 1.2
    stream = cv2.VideoCapture('./weights/demo.avi')

    # Check if camera opened successfully
    if not stream.isOpened():
        print("Error opening video stream or file")

    w = 650
    h = 600

    writer = cv2.VideoWriter('demo.avi',
                             cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             int(stream.get(cv2.CAP_PROP_FPS)), (w, h))

    # Read until video is completed
    while stream.isOpened():
        # Capture frame-by-frame
        success, frame = stream.read()
        if success:
            frame = frame[900:1500, 1000:1650, ]
            boxes = face(frame, (640, 640))
            boxes = boxes.astype('int32')
            for box in boxes:
                x_min = box[0]
                y_min = box[1]
                x_max = box[2]
                y_max = box[3]
                box_w = x_max - x_min
                box_h = y_max - y_min

                # remove a part of top area for alignment, see paper for details
                x_min -= int(box_w * (scale - 1) / 2)
                y_min += int(box_h * (scale - 1) / 2)
                x_max += int(box_w * (scale - 1) / 2)
                y_max += int(box_h * (scale - 1) / 2)
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, w - 1)
                y_max = min(y_max, h - 1)
                box_w = x_max - x_min + 1
                box_h = y_max - y_min + 1
                image = frame[y_min:y_max, x_min:x_max, :]
                image = cv2.resize(image, (args.input_size, args.input_size))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = numpy.stack((image, image, image), -1)
                image = image.astype("float32")
                cv2.subtract(image, mean, image)  # inplace
                cv2.multiply(image, 1 / std, image)  # inplace
                image = image.transpose((2, 0, 1))
                image = numpy.ascontiguousarray(image)
                image = numpy.expand_dims(image, 0)

                output = landmark(image)[0]

                for i in range(params['num_lms']):
                    x = int(output[i * 2] * box_w)
                    y = int(output[i * 2 + 1] * box_h)
                    cv2.circle(frame, (x + x_min, y + y_min), 1, (0, 255, 0), 1)

            cv2.imshow('IMAGE', frame)

            writer.write(frame.astype('uint8'))

            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    stream.release()
    writer.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=256, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)
    if args.demo:
        demo(args, params)


if __name__ == "__main__":
    main()
