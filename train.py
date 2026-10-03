
import torch
import time
import os
import datetime
import random
import numpy as np
import argparse

from models import build_model_from_name
from models.loss import build_criterion
from data.fastmri import build_dataset
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from engine import train_one_epoch, evaluate, distributed_evaluate, do_vis
from util.misc import init_distributed_mode, get_rank, save_on_master
from config import build_config


def main(args, work):

    init_distributed_mode(args)

    # build criterion and model first
    model = build_model_from_name(args, work)
    criterion = build_criterion(args)

    start_epoch = 0

    seed = args.SEED + get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.SOLVER.DEVICE)

    model.to(device)
    criterion.to(device)

    #if args.distributed:
    #    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
    elif len(args.SOLVER.DEVICE_IDS) > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=args.SOLVER.DEVICE_IDS
        )


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: %.2f M' % (n_parameters / 1024 / 1024))

    # build optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.SOLVER.LR, momentum=0.9, weight_decay=args.SOLVER.WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.SOLVER.LR_DROP)

    # build dataset
    dataset_train = build_dataset(args, mode='train')
    dataset_val = build_dataset(args, mode='val')

    dataset_val_len = len(dataset_val)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.SOLVER.BATCH_SIZE, drop_last=True)

    dataloader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  num_workers=args.SOLVER.NUM_WORKERS, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.SOLVER.BATCH_SIZE,
                                sampler=sampler_val, num_workers=args.SOLVER.NUM_WORKERS,
                                pin_memory=True)

    best_status = {'NMSE': 10000000, 'PSNR': 0, 'SSIM': 0}
    best_epoch = -1

    if args.RESUME != '':
        checkpoint = torch.load(
            args.RESUME, map_location='cpu', weights_only=False
        )
        model_state = {
            key.replace("module.", ""): val
            for key, val in checkpoint['model'].items()
        }

        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(model_state, strict=False)

        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        start_epoch = checkpoint.get('epoch', -1) + 1
        best_status = checkpoint.get('best_status', best_status)
        best_epoch = checkpoint.get('best_epoch', checkpoint.get('epoch', -1))

        print('resume from %s' % args.RESUME)
        print('start epoch:', start_epoch)

    if args.OUTPUTDIR:
        Path(args.OUTPUTDIR).mkdir(parents=True, exist_ok=True)


    start_time = time.time()

    for epoch in range(start_epoch, args.TRAIN.EPOCHS):
        train_status = train_one_epoch(
            args, model, criterion, dataloader_train, optimizer,
            epoch, args.SOLVER.PRINT_FREQ, device
        )
        lr_scheduler.step()

        if args.distributed:
            eval_status = distributed_evaluate(
                args, model, criterion, dataloader_val,
                device, dataset_val_len
            )
        else:
            
            #eval_status = evaluate(
            #    args, model, criterion, dataloader_val, device
            #)
            eval_status = evaluate(
                args, model, criterion, dataloader_val,
                device, args.OUTPUTDIR
            )           
            
            
        is_best = eval_status['PSNR'] > best_status['PSNR']
        if is_best:
            best_status = eval_status
            best_epoch = epoch

        if args.OUTPUTDIR:
            Path(args.OUTPUTDIR).mkdir(parents=True, exist_ok=True)

            target_model = model.module if hasattr(model, "module") else model
            checkpoint = {
                'model': target_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_status': best_status,
                'args': args,
            }

            last_path = os.path.join(
                args.OUTPUTDIR, 'checkpoint_last.pth'
            )

            if args.distributed:
                save_on_master(checkpoint, last_path)
            else:
                torch.save(checkpoint, last_path)

            if is_best:
                best_path = os.path.join(args.OUTPUTDIR, 'best.pth')
                if args.distributed:
                    save_on_master(checkpoint, best_path)
                else:
                    torch.save(checkpoint, best_path)

    print('The best epoch is ', best_epoch)
    print("Results ----------")
    print("NMSE: {:.4}".format(best_status['NMSE']))
    print("PSNR: {:.4}".format(best_status['PSNR']))
    print("SSIM: {:.4}".format(best_status['SSIM']))
    print("------------------")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="a unit Cross Multi modity transformer")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument(
        "--experiment", default="sr_multi_early", help="choose a experiment to do")
    args = parser.parse_args()

    print('doing ', args.experiment)

    cfg = build_config(args.experiment)

    print(cfg)

    main(cfg, args.experiment)





