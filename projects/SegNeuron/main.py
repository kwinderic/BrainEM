"""SegNeuron supervised train / inference entry point.

MNet is registered as arch "segneuron" and computes its own dual-head loss
(affinity + foreground mask) inside forward, mirroring the SAM2AFF CustomTrainer
convention. The framework dataloader supplies both targets via
MODEL.TARGET_OPT = ['2', '0'] (affinity + binary foreground).
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional
from yacs.config import CfgNode

from connectomics.utils.system import get_args, init_devices
from connectomics.config import load_cfg, save_all_cfg
from connectomics.engine import Trainer
from connectomics.engine.solver import *
from connectomics.model import *
from connectomics.model.build import make_parallel, MODEL_MAP
from connectomics.utils.monitor import build_monitor
from connectomics.data.augmentation import build_train_augmentor, TestAugmentor
from connectomics.data.dataset import build_dataloader, get_dataset, VolumeDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import add_segneuron_config
import segneuron  # noqa: F401  (registers "segneuron" arch)


class SegNeuronTrainer(Trainer):
    def __init__(self, cfg, device, mode='train', rank=None, checkpoint=None):
        self.init_basics(cfg, device, mode, rank)
        self._ssl = (mode == 'train' and bool(self.cfg.MODEL.SSL_PRETRAIN))

        if self._ssl:
            self._init_ssl(rank, checkpoint)
            return

        self.model = build_model(self.cfg, self.device, rank)

        if self.mode == 'train' and self.cfg.MODEL.SEGNEURON_PRETRAIN:
            self._load_pretrain_encoder(self.cfg.MODEL.SEGNEURON_PRETRAIN)

        if self.mode == 'train':
            self.optimizer = build_optimizer(self.cfg, self.model)
            self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
            self.scaler = GradScaler() if cfg.MODEL.MIXED_PRECESION else None
            self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
            self.update_checkpoint(checkpoint)
            self.augmentor = build_train_augmentor(self.cfg)
            self.criterion = None  # model computes its own dual-head loss
            if self.is_main_process:
                self.monitor = build_monitor(self.cfg)
                self.monitor.load_info(self.cfg, self.model)
            self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
            self.total_time = 0
        else:
            self.update_checkpoint(checkpoint)
            self.augmentor = TestAugmentor.build_from_cfg(cfg, activation=True)
            if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
                self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME
                self.test_filename = self.augmentor.update_name(self.test_filename)

        self.dataset, self.dataloader = None, None
        if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
            dataset_class, dataset_options = VolumeDataset, {}
            # Cross-volume mixing (official freq_mix / spa_mix) needs a second
            # patch from another source volume, which a per-sample augmentor
            # can't provide -- inject the dataset subclass that does it.
            if self.mode == 'train' and (self.cfg.AUGMENTOR.FREQ_MIX_PROB > 0 or
                                         self.cfg.AUGMENTOR.SPA_MIX_PROB > 0):
                from segneuron.mix_dataset import SegNeuronVolumeDataset
                dataset_class = SegNeuronVolumeDataset
                dataset_options = {
                    'freq_mix_prob': self.cfg.AUGMENTOR.FREQ_MIX_PROB,
                    'freq_mix_l': self.cfg.AUGMENTOR.FREQ_MIX_L,
                    'spa_mix_prob': self.cfg.AUGMENTOR.SPA_MIX_PROB,
                    'spa_mix_window_frac': self.cfg.AUGMENTOR.SPA_MIX_WINDOW_FRAC,
                }
            self.dataloader = build_dataloader(
                self.cfg, self.augmentor, self.mode, rank=rank,
                dataset_class=dataset_class, dataset_options=dataset_options)
            self.dataloader = iter(self.dataloader)

    def _init_ssl(self, rank, checkpoint):
        """Self-supervised pretraining setup: pretrain MNet (dual recon heads),
        multi-volume masked-inpainting loader, no labels, no criterion. DDP is
        handled by make_parallel (same as build_model)."""
        from segneuron.model.mnet_pretrain import MNet as MNetPretrain

        kn = tuple(self.cfg.MODEL.FILTERS)
        model = MNetPretrain(1, kn=kn, FMU=self.cfg.MODEL.FMU)
        # MNetPretrain's `outputs`/`outputs2` ModuleLists carry 7 conv heads
        # each (deep-supervision scaffolding from the supervised MNet), but
        # forward() only reads index 0 of each — the other 6x2 heads never
        # get gradients, which DDP's default strict all-params-used check
        # rejects.
        self.model = make_parallel(model, self.cfg, self.device, rank,
                                   find_unused_parameters=True)

        self.optimizer = build_optimizer(self.cfg, self.model)
        self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
        self.scaler = GradScaler() if self.cfg.MODEL.MIXED_PRECESION else None
        self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
        self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
        self.total_time = 0
        if self.is_main_process:
            self.monitor = build_monitor(self.cfg)
            self.monitor.load_info(self.cfg, self.model)

        # Unlabeled crops come from the framework's own dataloader, which
        # already handles multi-volume pools, h5/tif mixes, DDP sharding and
        # worker seeding -- SSL just drops the targets. Requires
        # DATASET.LABEL_NAME None, so VolumeDataset returns (pos, volume) and
        # collate_fn_test packs it into .out_input.
        #
        # augmentor is None on purpose: the framework augmentations assume a
        # label is present (misalign/rescale call .copy() on it and crash on
        # None). The reference SimpleAugment (flips + xy transpose) is applied
        # per sample in _train_ssl_recon instead.
        #
        # No StopIteration risk: VolumeDataset sizes itself as
        # max(ITERATION_TOTAL * batch, total sliding-window positions), and for
        # a multi-volume pool the latter dwarfs any DDP shard.
        from connectomics.data.dataset.collate import collate_fn_test
        if self.is_main_process:
            print(f"[ssl] building unlabeled dataloader (crop="
                  f"{tuple(self.cfg.MODEL.INPUT_SIZE)}, "
                  f"batch={self.cfg.SOLVER.SAMPLES_PER_BATCH}/rank)")
        self.dataloader = iter(build_dataloader(
            self.cfg, None, 'train', rank=rank, cf=collate_fn_test))
        self._ssl_hist = []

    def _load_pretrain_encoder(self, path):
        """Load encoder (down*) from a self-supervised pretrain checkpoint.

        The pretrain ckpt uses key 'model_weights' and contains up*/outputs*
        which we drop (decoder/heads differ between pretrain and supervised).
        The supervised model wraps MNet under `.mnet`, so pretrain keys like
        'down11...' map to 'mnet.down11...'.
        """
        print(f"[SegNeuron] loading pretrain encoder from {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("model_weights", ck.get("state_dict", ck))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        net = self.model.module if hasattr(self.model, "module") else self.model
        mdict = net.state_dict()
        transfer = {}
        for k, v in sd.items():
            if "up" in k or "outputs" in k:
                continue
            mk = k if k in mdict else ("mnet." + k if "mnet." + k in mdict else None)
            if mk is not None and mdict[mk].shape == v.shape:
                transfer[mk] = v
        mdict.update(transfer)
        net.load_state_dict(mdict)
        print(f"[SegNeuron] transferred {len(transfer)} encoder params")

    def train(self):
        self.model.train()
        if self._ssl:
            self._train_ssl_recon()
            return

        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            sample = next(self.dataloader)
            volume = sample.out_input
            target, weight = sample.out_target_l, sample.out_weight_l
            self.data_time = time.perf_counter() - self.start_time

            volume = volume.to(self.device, non_blocking=True)
            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                pred, loss, losses_vis = self.model(volume, target, weight)

            self._train_misc(loss, pred, volume, target, weight, iter_total, losses_vis)
        self.maybe_save_swa_model()

    def _train_ssl_recon(self):
        """Self-supervised masked-inpainting + HOG-reconstruction loop.

        Each step: sample (masked_img, clean_img, hog_gt) from
        SegNeuronPretrainDataset, forward through the pretrain MNet (dual
        1-channel recon heads), loss = SSL_IMG_WEIGHT * MSE(img) + MSE(hog)
        (official: 0.2 * loss1 + loss2).
        Mirrors Arch/main.py's _train_mae() structure (scaler/monitor/
        save_checkpoint/scheduler_step), but data comes from self.dataloader
        instead of a patch loader since masking/HOG happen in the dataset.
        """
        img_weight = self.cfg.MODEL.SSL_IMG_WEIGHT
        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            # Framework loader yields a TestBatch of [B,1,Z,Y,X] in [0,1];
            # build the pretext targets (block mask + noise fill + HOG) here.
            volume = next(self.dataloader).out_input
            img_mask, img_gt, hog_gt = self._make_ssl_batch(volume)
            img_mask = img_mask.to(self.device, non_blocking=True)
            img_gt = img_gt.to(self.device, non_blocking=True)
            hog_gt = hog_gt.to(self.device, non_blocking=True)
            self.data_time = time.perf_counter() - self.start_time

            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                rec_img, rec_hog = self.model(img_mask)
                loss_img = F.mse_loss(rec_img, img_gt)
                loss_hog = F.mse_loss(rec_hog, hog_gt)
                loss = img_weight * loss_img + loss_hog

            losses_vis = {'loss_img': loss_img.detach(),
                          'loss_hog': loss_hog.detach()}

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if hasattr(self, 'monitor'):
                self.monitor.update(iter_total, loss.detach(), losses_vis,
                                    self.optimizer.param_groups[0]['lr'])
            if (iter_total + 1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
                self.save_checkpoint(iter_total)
            if self.is_main_process and \
                    (iter_total + 1) % self.cfg.MODEL.SSL_VIS_INTERVAL == 0:
                self._save_ssl_vis(img_mask, img_gt, rec_img, hog_gt, rec_hog, iter_total)
            self.scheduler_step(iter_total, loss.detach())

            if self.is_main_process:
                self.iter_time = time.perf_counter() - self.start_time
                self.total_time += self.iter_time
                avg_it = self.total_time / (iter_total + 1 - self.start_iter)
                left = avg_it * (self.total_iter_nums + self.start_iter -
                                 iter_total - 1) / 3600.0
                print('[SSL Iter %05d] data=%.3fs iter=%.3fs avg=%.3fs '
                      'left=%.2fh loss=%.4f' % (
                          iter_total, self.data_time,
                          time.perf_counter() - self.start_time,
                          avg_it, left, loss.item()))

            del img_mask, img_gt, hog_gt, rec_img, rec_hog, loss, losses_vis

        # SSL has no validation loop; the final encoder is what finetune loads.
        self.save_checkpoint(iter_total, is_best=True)

    def _make_ssl_batch(self, volume):
        """Turn a framework image batch into the SSL pretext triplet.

        volume: [B,1,Z,Y,X] float in [0,1] from the framework dataloader.
        Applies the reference SimpleAugment (flips + xy transpose) then the
        block-mask / noise-fill / HOG construction, per sample.
        """
        from segneuron.pretrain_data import SimpleAugment, make_ssl_targets
        aug = SimpleAugment()
        masked, clean, hogs = [], [], []
        for b in range(volume.shape[0]):
            img = volume[b, 0].cpu().numpy()
            if self.cfg.AUGMENTOR.ENABLED:
                img = aug(img)
            m, c, h = make_ssl_targets(np.ascontiguousarray(img, dtype=np.float32))
            masked.append(m)
            clean.append(c)
            hogs.append(h)
        return torch.stack(masked), torch.stack(clean), torch.stack(hogs)

    def _save_ssl_vis(self, img_mask, img_gt, rec_img, hog_gt, rec_hog, iter_total):
        """Save a 5-panel PNG: masked input / reconstructed img / clean GT /
        reconstructed HOG / HOG GT, for the middle z-slice of sample 0."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        vis_dir = os.path.join(self.cfg.DATASET.OUTPUT_PATH, 'vis_ssl')
        os.makedirs(vis_dir, exist_ok=True)

        z = img_gt.shape[2] // 2
        panels = [
            (img_mask[0, 0, z].detach().float().cpu().numpy(), 'masked input', 'gray'),
            (rec_img[0, 0, z].detach().float().cpu().numpy(), 'reconstructed img', 'gray'),
            (img_gt[0, 0, z].detach().float().cpu().numpy(), 'clean GT', 'gray'),
            (rec_hog[0, 0, z].detach().float().cpu().numpy(), 'reconstructed HOG', 'magma'),
            (hog_gt[0, 0, z].detach().float().cpu().numpy(), 'HOG GT', 'magma'),
        ]
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
        for ax, (im, title, cmap) in zip(axes, panels):
            ax.imshow(im, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        fig.suptitle(f'SSL iter {iter_total}', fontsize=11)
        fig.tight_layout()
        out = os.path.join(vis_dir, f'ssl_recon_{iter_total + 1:06d}.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f'[ssl-vis] saved {out}')


def main():
    args = get_args()
    cfg = load_cfg(args, add_cfg_func=add_segneuron_config)
    device = init_devices(args, cfg)

    if args.local_rank in (0, None):
        print("PyTorch: ", torch.__version__)
        if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
            os.makedirs(cfg.DATASET.OUTPUT_PATH)
            save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    mode = 'test' if args.inference else 'train'
    trainer = SegNeuronTrainer(cfg, device, mode, rank=args.local_rank, checkpoint=args.checkpoint)
    if cfg.DATASET.DO_CHUNK_TITLE == 0:
        test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
        test_func() if args.inference else trainer.train()
    else:
        trainer.run_chunk(mode)
    print("Rank: {}. Device: {}. Finished!".format(args.local_rank, device))


if __name__ == "__main__":
    main()
