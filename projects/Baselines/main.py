import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Any, Callable, Dict, Iterable, List, Set, Type, Union, Optional
from yacs.config import CfgNode
import time
import GPUtil
from collections import defaultdict

from connectomics.utils.system import get_args, init_devices
from connectomics.config import load_cfg, save_all_cfg
from connectomics.engine import Trainer
from connectomics.engine.solver import *
from connectomics.model import *
from connectomics.utils.monitor import build_monitor
from connectomics.data.augmentation import build_train_augmentor, TestAugmentor
from connectomics.data.dataset import build_dataloader, get_dataset
from connectomics.model.build import make_parallel, MODEL_MAP

from projects.Baselines.models import *
from projects.Baselines.models.config import add_custom_config


class CustomTrainer(Trainer):
    r"""Trainer class for supervised learning.

    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        device (torch.device): model running device. GPUs are recommended for model training and inference.
        mode (str): running mode of the trainer (``'train'`` or ``'test'``). Default: ``'train'``
        rank (int, optional): node rank for distributed training. Default: `None`
        checkpoint (str, optional): the checkpoint file to be loaded. Default: `None`
    """

    def __init__(self,
                 cfg: CfgNode,
                 device: torch.device,
                 mode: str = 'train',
                 rank: Optional[int] = None,
                 checkpoint: Optional[str] = None):
        self.init_basics(cfg, device, mode, rank)

        self.model = build_model(self.cfg, self.device, rank)
        if True:
            print(self.model)
            import torch.distributed as dist

            total_params = sum(p.numel() for p in self.model.parameters())
            
            print(f"总参数数: {total_params}")

            try:
                from mmcv.cnn.utils import get_model_complexity_info
                input_shape = tuple([1] + cfg.MODEL.INPUT_SIZE)
                flops, params = get_model_complexity_info(self.model, input_shape)
                split_line = '=' * 30
                print(f'{split_line}\nInput shape: {input_shape}\nFlops: {flops}\nParams: {params}\n{split_line}')
            except ImportError:
                print('Please install mmcv to get model complexity information.')

        if self.mode == 'train':
            self.optimizer = self._build_optimizer()
            self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
            self.scaler = GradScaler() if cfg.MODEL.MIXED_PRECESION else None
            self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
            self.update_checkpoint(checkpoint)

            # stochastic weight averaging
            if self.cfg.SOLVER.SWA.ENABLED:
                self.swa_model, self.swa_scheduler = build_swa_model(
                    self.cfg, self.model, self.optimizer)

            self.augmentor = build_train_augmentor(self.cfg)
            self.criterion = Criterion.build_from_cfg(self.cfg, self.device)
            if self.is_main_process:
                self.monitor = build_monitor(self.cfg)
                self.monitor.load_info(self.cfg, self.model)

            self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
            self.total_time = 0
        else:
            self.update_checkpoint(checkpoint)
            # PEA 推理输出 3 通道短程 affinity，覆盖 TARGET_OPT 避免 SplitActivation 不匹配
            if cfg.MODEL.ARCHITECTURE == 'pea':
                cfg.defrost()
                cfg.MODEL.TARGET_OPT = ["2-1-1-1-v1"]
                cfg.INFERENCE.OUTPUT_ACT = ["none"]
                cfg.MODEL.OUT_PLANES = 3
                cfg.freeze()
            elif cfg.MODEL.ARCHITECTURE == 'cad':
                cfg.defrost()
                cfg.MODEL.TARGET_OPT = ["2-1-1-1-v1"]
                cfg.INFERENCE.OUTPUT_ACT = ["none"]
                cfg.MODEL.OUT_PLANES = 3
                cfg.freeze()
            # build test-time augmentor and update output filename
            self.augmentor = TestAugmentor.build_from_cfg(cfg, activation=True)
            if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
                self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME
                self.test_filename = self.augmentor.update_name(self.test_filename)

        self.dataset, self.dataloader = None, None
        if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
            self.dataloader = build_dataloader(
                self.cfg, self.augmentor, self.mode, rank=rank,
                # dataset_options={"ensure_single_connected": True, "return_clean_input": True})
                dataset_options={"ensure_single_connected": True})
            self.dataloader = iter(self.dataloader)
            if self.mode == 'train' and cfg.DATASET.VAL_IMAGE_NAME is not None:
                self.val_loader = build_dataloader(
                    self.cfg, None, mode='val', rank=rank,
                dataset_options={"ensure_single_connected": True})

    def _build_optimizer(self):
        """Build optimizer, applying extra Adam params from custom config."""
        # CAD: two param groups with separate LRs
        if self.cfg.SOLVER.NAME in ('Adam', 'AdamW') and self.cfg.MODEL.ARCHITECTURE == 'cad':
            model = self.model.module if hasattr(self.model, 'module') else self.model
            param_groups = model.get_param_groups()
            base_lr = self.cfg.SOLVER.BASE_LR
            optimizer = torch.optim.Adam(
                [{'params': pg['params'], 'lr': base_lr * pg['lr_ratio']}
                 for pg in param_groups],
                lr=base_lr,
                betas=self.cfg.SOLVER.BETAS,
                eps=self.cfg.SOLVER.ADAM_EPS,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
                amsgrad=self.cfg.SOLVER.AMSGRAD,
            )
            return optimizer

        optimizer = build_optimizer(self.cfg, self.model)
        name = self.cfg.SOLVER.NAME
        eps = self.cfg.SOLVER.ADAM_EPS
        amsgrad = self.cfg.SOLVER.AMSGRAD
        if name in ('Adam', 'AdamW') and (eps != 1e-8 or amsgrad):
            # Rebuild with per-group params preserved (lr, weight_decay already
            # set per-group by build_optimizer) plus custom eps/amsgrad.
            optimizer = getattr(torch.optim, name)(
                optimizer.param_groups,
                lr=self.cfg.SOLVER.BASE_LR,
                betas=self.cfg.SOLVER.BETAS,
                eps=eps,
                amsgrad=amsgrad,
            )
        return optimizer

    def _in_write_list(self, model):
        write_list = ['unet_3d', 'unet_2d', 'fpn_3d', 'unet_plus_3d', 'unet_plus_2d', 
            'deeplabv3a', 'deeplabv3b', 'deeplabv3c', 'unetr', 'swinunetr']
        rt = False 
        model = model.module if isinstance(model, nn.DataParallel) \
            or isinstance(model, nn.parallel.DistributedDataParallel) else model
        for w in write_list:
            if isinstance(model, MODEL_MAP[w]):
                rt = True
                break
        return rt

    def train(self):
        r"""Training function of the trainer class.
        """
        self.model.train()

        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            # load data
            sample = next(self.dataloader)
            volume = sample.out_input
            target, weight = sample.out_target_l, sample.out_weight_l
            self.data_time = time.perf_counter() - self.start_time

            # prediction
            volume = volume.to(self.device, non_blocking=True)
            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                if self._in_write_list(self.model):     # predefined models in pytorch_connectomics
                    pred = self.model(volume)
                    loss, losses_vis = self.criterion(pred, target, weight)
                else:       # custom models
                    pred, loss, losses_vis = self.model(
                        volume, target, weight, 
                    criterion=self.criterion)

            # Crop volume/target/weight to match pred spatial size (e.g. MALA valid conv)
            if volume.shape[2:] != pred.shape[2:]:
                out_size = pred.shape[2:]
                off = [(v - o) // 2 for v, o in zip(volume.shape[2:], out_size)]
                slc = (slice(None), slice(None),
                       slice(off[0], off[0]+out_size[0]),
                       slice(off[1], off[1]+out_size[1]),
                       slice(off[2], off[2]+out_size[2]))
                def _crop(x):
                    if isinstance(x, list):
                        return [_crop(i) for i in x]
                    if hasattr(x, 'ndim') and x.ndim >= len(slc):
                        return x[slc]
                    return x
                volume = volume[slc]
                target = _crop(target)
                weight = _crop(weight)

            self._train_misc(loss, pred, volume, target, weight,
                             iter_total, losses_vis)

        self.maybe_save_swa_model()

    def validate(self, iter_total):
        if not hasattr(self, 'val_loader'):
            return

        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i, sample in enumerate(self.val_loader):
                volume = sample.out_input
                target, weight = sample.out_target_l, sample.out_weight_l

                volume = volume.to(self.device, non_blocking=True)
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    if self._in_write_list(self.model):
                        pred = self.model(volume)
                        loss, _ = self.criterion(pred, target, weight)
                    else:
                        pred, loss, _ = self.model(
                            volume, target, weight,
                            criterion=self.criterion)
                    val_loss += loss.data

        if hasattr(self, 'monitor'):
            self.monitor.logger.log_tb.add_scalar(
                'Validation_Loss', val_loss, iter_total)
            if self._in_write_list(self.model):
                self.monitor.visualize(volume, target, pred,
                                       weight, iter_total, suffix='Val')

        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = val_loss

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(iter_total, is_best=True)

        del pred, loss, val_loss
        self.model.train()


def main():
    args = get_args()
    cfg = load_cfg(args, add_cfg_func=add_custom_config)
    device = init_devices(args, cfg)

    if args.local_rank == 0 or args.local_rank is None:
        # In distributed training, only print and save the configurations 
        # using the node with local_rank=0.
        print("PyTorch: ", torch.__version__)
        print(cfg)

        if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
            print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
            os.makedirs(cfg.DATASET.OUTPUT_PATH)
            save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    # start training or inference
    mode = 'test' if args.inference else 'train'
    trainer = CustomTrainer(cfg, device, mode,
                      rank=args.local_rank,
                      checkpoint=args.checkpoint)

    # Start training or inference:
    if cfg.DATASET.DO_CHUNK_TITLE == 0:
        test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
        test_func() if args.inference else trainer.train()
    else:
        trainer.run_chunk(mode)

    print("Rank: {}. Device: {}. Process is finished!".format(
          args.local_rank, device))


if __name__ == "__main__":
    main()

