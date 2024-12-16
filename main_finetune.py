import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import Dataset

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from llama.llama_adapter import LLaMA_adapter

# from data.dataset import FinetuneDataset, transform_train
from data.dataset import AssisterDataset, CaptionDataset, transform_train
from data import (
    MultiStepNavData,
    MlmDataset, mlm_collate,
    SapDataset, sap_collate,
    SarDataset, sar_collate,
    SprelDataset, sprel_collate,
    MrcDataset, mrc_collate,
    ItmDataset, itm_collate,
    LmpDataset, lmp_collate,
    MetaLoader, PrefetchLoader,
    build_dataloader)

import argparse
import datetime
from easydict import EasyDict
import json
import numpy as np
import os
import time
from pathlib import Path

from engine_finetune import train_one_epoch, eval_one_epoch, train_one_epoch_img, eval_one_epoch_img


def create_dataloaders(
    data_cfg, nav_db, tok, is_train: bool, device: torch.device, opts
):
    dataloaders = {}
    for k, task_name in enumerate(data_cfg.tasks):
        if task_name == 'mlm':
            task_dataset = MlmDataset(nav_db, tok)
            task_collate_fn = mlm_collate
        elif task_name == 'sap':
            task_dataset = SapDataset(
                nav_db,
                tok,
                opts.ob_random_kill_v if is_train else 0,
                opts.ob_random_kill_a if is_train else 0
            )
            task_collate_fn = sap_collate
        elif task_name == 'sar':
            task_dataset = SarDataset(
                nav_db,
                tok,
                opts.ob_random_kill_v if is_train else 0,
                opts.ob_random_kill_a if is_train else 0
            )
            task_collate_fn = sar_collate
        elif task_name == 'sprel':
            task_dataset = SprelDataset(
                nav_db,
                tok,
                opts.ob_random_kill_v if is_train else 0,
                opts.ob_random_kill_a if is_train else 0
            )
            task_collate_fn = sprel_collate
        elif task_name == 'mrc':
            task_dataset = MrcDataset(nav_db, tok, opts.mrc_mask_prob)
            task_collate_fn = mrc_collate
        elif task_name == 'itm':
            task_dataset = ItmDataset(nav_db, tok)
            task_collate_fn = itm_collate
        elif task_name == 'lmp':
            task_dataset = LmpDataset(nav_db, tok)
            task_collate_fn = lmp_collate
        else:
            raise ValueError(f'Undefined task {task_name}')

        print(f"{task_name}: {len(task_dataset)} samples loaded")

        task_loader, pre_epoch = build_dataloader(
            task_name, task_dataset, task_collate_fn, is_train, opts
        )

        if is_train:
            ratio = data_cfg.mix_ratio[k]
            dataloaders[task_name] = (task_loader, ratio, pre_epoch)
        else:
            dataloaders[task_name] = PrefetchLoader(task_loader, device)
    return dataloaders


def get_args_parser():
    parser = argparse.ArgumentParser(
        'llama_adapterV2 pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='7B', type=str,
                        help='Type of LLaMA model')
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='path to checkpoint from pretrain stage')
    parser.add_argument('--max_words', default=384, type=int,
                        help='max number of input words')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_config', default='config/data/pretrain_r2r.json', type=str,
                        help='dataset config path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--ob_random_kill_v', default=0.3, type=float)
    parser.add_argument('--ob_random_kill_a', default=0.43, type=float)

    parser.add_argument('--assister_path', default='/data/user/kxh/instructllm/ASSISTER',
                        help='path to the ASSISTER dataset')
    parser.add_argument(
        "--caption_data_path",
        default=".",
        help="path to the captioned dataset",
    )

    return parser


def main(args):
    misc.init_distributed_mode(args)

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = os.environ['LOCAL_RANK']

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # define the model
    llama_type = args.llama_type
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    llama_tokenzier_path = os.path.join(args.llama_path, 'tokenizer.model')
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path, max_batch_size=args.batch_size, max_seq_len=args.max_words)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    print("Trainable Params:")
    print([(key, val.shape)
          for key, val in model.named_parameters() if val.requires_grad])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # training detail
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(model_without_ddp, args.pretrained_path)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # dataset_train = FinetuneDataset(args.data_config, transform=transform_train,
    #                                 max_words=args.max_words, tokenizer_path=llama_tokenzier_path)
    # print(dataset_train)
    
    # sampler_train = torch.utils.data.DistributedSampler(
    #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    # print("Sampler_train = %s" % str(sampler_train))

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    # dataset_train = AssisterDataset(args.assister_path, transform=transform_train,
    #                                 max_words=args.max_words, tokenizer_path=llama_tokenzier_path)
    # print(dataset_train)
    
    # sampler_train = torch.utils.data.DistributedSampler(
    #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    # print("Sampler_train = %s" % str(sampler_train))

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    # dataset_val = AssisterDataset(args.assister_path, transform=transform_train,
    #                               max_words=args.max_words, tokenizer_path=llama_tokenzier_path, training=False)
    # print(dataset_val)
    
    # sampler_val = torch.utils.data.DistributedSampler(
    #     dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    # print("Sampler_val = %s" % str(sampler_val))

    # data_loader_val = torch.utils.data.DataLoader(
    #     dataset_val, sampler=sampler_val,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=False,
    # )

    # dataset_type = 'r2r'

    # dataset_train = CaptionDataset(
    #     args.caption_data_path,
    #     dataset_name=dataset_type,
    #     max_words=args.max_words,
    #     tokenizer_path=llama_tokenzier_path
    # )
    # print(dataset_train)
    
    # sampler_train = torch.utils.data.DistributedSampler(
    #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    # print("Sampler_train = %s" % str(sampler_train))

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    # dataset_val = CaptionDataset(
    #     args.caption_data_path,
    #     dataset_name=dataset_type,
    #     max_words=args.max_words,
    #     tokenizer_path=llama_tokenzier_path,
    #     training=False,
    # )
    # print(dataset_val)
    
    # sampler_val = torch.utils.data.DistributedSampler(
    #     dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    # print("Sampler_val = %s" % str(sampler_val))

    # data_loader_val = torch.utils.data.DataLoader(
    #     dataset_val, sampler=sampler_val,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=False,
    # )

    dataset_cfg = json.load(open(args.data_config))
    r2r_cfg = EasyDict(dataset_cfg['train_datasets']['R2R'])
    train_nav_db = MultiStepNavData(
        r2r_cfg.train_traj_files, r2r_cfg.img_ft_file, 
        r2r_cfg.scanvp_cands_file, r2r_cfg.connectivity_dir, 
        image_prob_size=0,
        image_feat_size=768, 
        angle_feat_size=4,
        max_txt_len=args.max_words, max_act_len=100,
        hist_enc_pano=True, 
        ob_cand_pano_view=False,
        val_sample_num=None, in_memory=True,
        tokenizer_path=llama_tokenzier_path,
        bboxes_file=r2r_cfg.bboxes_file
    )
    val2_nav_db = MultiStepNavData(
        r2r_cfg.val_unseen_traj_files, r2r_cfg.img_ft_file, 
        r2r_cfg.scanvp_cands_file, r2r_cfg.connectivity_dir, 
        image_prob_size=0,
        image_feat_size=768, 
        angle_feat_size=4,
        max_txt_len=args.max_words, max_act_len=100,
        hist_enc_pano=True, 
        ob_cand_pano_view=False,
        val_sample_num=None, in_memory=True,
        tokenizer_path=llama_tokenzier_path,
        bboxes_file=r2r_cfg.bboxes_file
    )
    train_dataloaders = create_dataloaders(
        r2r_cfg, train_nav_db, None, True, device, args
    )
    val2_dataloaders = create_dataloaders(
        r2r_cfg, val2_nav_db, None, False, device, args
    )
    meta_loader = MetaLoader(
        train_dataloaders,
        accum_steps=1,
        distributed=args.local_rank != -1,
        device=device,
        num_iters=500
    )
    # meta_loader = PrefetchLoader(train_dataloaders['sap'][0], device)
    data_loader_train = meta_loader

    # SummaryWrite
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)

        # TODO: comment out above

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # train_stats = train_one_epoch_img(
        #     model, data_loader_train,
        #     optimizer, device, epoch, loss_scaler,
        #     log_writer=log_writer,
        #     args=args
        # )

        if args.output_dir and (epoch % 2 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            
        val_stats_sap = eval_one_epoch(model, val2_dataloaders['sap'], device, epoch, log_writer=log_writer)
        val_stats_itm = eval_one_epoch(model, val2_dataloaders['itm'], device, epoch, log_writer=log_writer)
        val_stats_lmp = eval_one_epoch(model, val2_dataloaders['lmp'], device, epoch, log_writer=log_writer)

        # val_stats = eval_one_epoch_img(model, data_loader_val, device, epoch, log_writer=log_writer)

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_sap_{k}': v for k, v in val_stats_sap.items()},
                     **{f'val_itm_{k}': v for k, v in val_stats_itm.items()},
                     **{f'val_lmp_{k}': v for k, v in val_stats_lmp.items()}}

        # log_stats = {'epoch': epoch,
        #              **{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'val_itm_{k}': v for k, v in val_stats_itm.items()}}

        # log_stats = {'epoch': epoch,
        #              **{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'val_{k}': v for k, v in val_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
