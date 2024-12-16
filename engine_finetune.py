import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from llama import LLaMA_adapter


def train_one_epoch(
    model: LLaMA_adapter,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    # model.module.set_default_trainability()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # examples, labels, example_mask, imgs
        examples = batch["txt_ids"]
        labels = batch["txt_labels"]
        imgs = batch["hist_img_fts"]
        ang_feats = batch["hist_ang_fts"]
        pano_img_feats = None
        pano_ang_feats = None
        if "hist_pano_img_fts" in batch:
            pano_img_feats = batch["hist_pano_img_fts"]
            pano_ang_feats = batch["hist_pano_ang_fts"]

        ob_img_feats = None
        ob_ang_feats = None
        # ob_attn_mask = None
        ob_nav_types = None
        ob_id_seps = None
        ob_action_viewindex = None
        if "ob_img_fts" in batch:
            ob_img_feats = batch["ob_img_fts"]
            ob_ang_feats = batch["ob_ang_fts"]
            # ob_attn_mask = batch['ob_attn_mask']
            ob_nav_types = batch["ob_nav_types"]
            ob_id_seps = batch["ob_id_seps"]
            ob_action_viewindex = batch["ob_action_viewindex"]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if imgs is not None:
            imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            c_loss, m_loss = model(
                examples,
                labels,
                imgs,
                ang_feats,
                pano_img_feats,
                pano_ang_feats,
                ob_img_feats,
                ob_ang_feats,
                ob_nav_types,
                ob_id_seps,
                ob_action_viewindex,
            )
        loss = c_loss + m_loss * 0
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        m_loss_value = m_loss
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("m_train_loss", m_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_one_epoch(model: LLaMA_adapter, data_loader: Iterable, device: torch.device, epoch: int, log_writer=None):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # examples, labels, example_mask, imgs
        examples = batch["txt_ids"]
        labels = batch["txt_labels"]
        imgs = batch["hist_img_fts"]
        ang_feats = batch["hist_ang_fts"]
        pano_img_feats = None
        pano_ang_feats = None
        if "hist_pano_img_fts" in batch:
            pano_img_feats = batch["hist_pano_img_fts"]
            pano_ang_feats = batch["hist_pano_ang_fts"]

        ob_img_feats = None
        ob_ang_feats = None
        # ob_attn_mask = None
        ob_nav_types = None
        ob_id_seps = None
        ob_action_viewindex = None
        if "ob_img_fts" in batch:
            ob_img_feats = batch["ob_img_fts"]
            ob_ang_feats = batch["ob_ang_fts"]
            # ob_attn_mask = batch['ob_attn_mask']
            ob_nav_types = batch["ob_nav_types"]
            ob_id_seps = batch["ob_id_seps"]
            ob_action_viewindex = batch["ob_action_viewindex"]

        if imgs is not None:
            imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                c_loss, m_loss = model(
                    examples,
                    labels,
                    imgs,
                    ang_feats,
                    pano_img_feats,
                    pano_ang_feats,
                    ob_img_feats,
                    ob_ang_feats,
                    ob_nav_types,
                    ob_id_seps,
                    ob_action_viewindex,
                )
            loss = c_loss + m_loss * 0
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            m_loss_value = m_loss

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if "ob_img_fts" in batch:
                log_writer.add_scalar("c_val_loss_sap", c_loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("m_val_loss_sap", m_loss_value_reduce, epoch_1000x)
            else:
                log_writer.add_scalar("c_val_loss_itm", c_loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("m_val_loss_itm", m_loss_value_reduce, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_img(
    model: LLaMA_adapter,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    # model.module.set_default_trainability()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        examples, labels, example_mask, imgs, gt_id, ori_prompt, gt_caption = batch

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if imgs is not None:
            imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            c_loss, m_loss = model(
                examples,
                labels,
                imgs,
            )
        loss = c_loss + m_loss * 0
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        m_loss_value = m_loss
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("m_train_loss", m_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_one_epoch_img(model: LLaMA_adapter, data_loader: Iterable, device: torch.device, epoch: int, log_writer=None):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        examples, labels, example_mask, imgs, gt_id, ori_prompt, gt_caption = batch

        if imgs is not None:
            imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                c_loss, m_loss = model(
                    examples,
                    labels,
                    imgs
                )
            loss = c_loss + m_loss * 0
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            m_loss_value = m_loss

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_val_loss", c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("m_val_loss", m_loss_value_reduce, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
