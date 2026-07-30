"""dbMiM train / inference entry point.

Two modes on one Trainer, selected by MODEL.SSL_PRETRAIN:

  * SSL pretraining (masked image modeling): builds DBMIM3DMAE, skips the
    labeled dataloader, samples crops from an unlabeled volume pool, and runs
    _train_ssl_mim. Goes through the standard run_train torchrun DDP path, the
    same convention as projects/SegNeuron.
  * Supervised finetuning: the "dbmim_unetr_aniso_em" arch (UNETREMAffinityNet
    plus a built-in MSE+MAWS loss) trained on framework affinity targets
    (MODEL.TARGET_OPT ['2']), initialized from the pretrained ViT encoder.

The reference uses a constant LR with no warmup and no scheduler, and a lower
LR for the encoder -- both reproduced here (see _build_dbmim_optimizer).
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from connectomics.utils.system import get_args, init_devices
from connectomics.config import load_cfg, save_all_cfg
from connectomics.engine import Trainer
from connectomics.engine.solver import build_lr_scheduler
from connectomics.model import *
from connectomics.model.build import make_parallel
from connectomics.utils.monitor import build_monitor
from connectomics.data.augmentation import build_train_augmentor, TestAugmentor
from connectomics.data.dataset import build_dataloader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import add_dbmim_config
import dbmim  # noqa: F401  (registers "dbmim_unetr_aniso_em")


class DBMiMTrainer(Trainer):
    def __init__(self, cfg, device, mode='train', rank=None, checkpoint=None):
        self.init_basics(cfg, device, mode, rank)
        self._ssl = (mode == 'train' and bool(self.cfg.MODEL.SSL_PRETRAIN))

        if self._ssl:
            self._init_ssl(rank, checkpoint)
            return

        self.model = build_model(self.cfg, self.device, rank)

        if self.mode == 'train' and self.cfg.MODEL.DBMIM_PRETRAIN:
            self._load_pretrain_encoder(self.cfg.MODEL.DBMIM_PRETRAIN)

        if self.mode == 'train':
            self.optimizer = self._build_dbmim_optimizer()
            self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
            self.scaler = GradScaler() if cfg.MODEL.MIXED_PRECESION else None
            self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
            self.update_checkpoint(checkpoint)
            self.augmentor = build_train_augmentor(self.cfg)
            self.criterion = None   # model computes its own MSE+MAWS loss
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
            self.dataloader = build_dataloader(self.cfg, self.augmentor, self.mode, rank=rank)
            self.dataloader = iter(self.dataloader)

    # ------------------------------------------------------------------
    # Optimizer: constant LR, encoder at DBMIM_ENCODER_LR
    # ------------------------------------------------------------------
    def _build_dbmim_optimizer(self):
        """AdamW with a separate, lower-LR group for the ViT encoder.

        Port of train_finetune.py:1273-1304. Params are split by name prefix;
        the wrapper nests the net under `net.`, so prefixes are matched on the
        name with that (and DDP's `module.`) stripped.
        """
        prefixes = tuple(self.cfg.SOLVER.DBMIM_ENCODER_PARAM_PREFIXES)
        enc_lr = float(self.cfg.SOLVER.DBMIM_ENCODER_LR)
        base_lr = float(self.cfg.SOLVER.BASE_LR)

        encoder_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            bare = name
            for strip in ("module.", "net."):
                if bare.startswith(strip):
                    bare = bare[len(strip):]
            (encoder_params if bare.startswith(prefixes) else other_params).append(param)

        if self.is_main_process:
            print(f"[dbMiM] optimizer: {len(encoder_params)} encoder params @ lr={enc_lr}, "
                  f"{len(other_params)} params @ lr={base_lr}")
        groups = [{"params": other_params, "lr": base_lr}]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": enc_lr})
        return torch.optim.AdamW(groups, lr=base_lr,
                                 weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)

    def _load_pretrain_encoder(self, path):
        """Load the ViT encoder from a dbMiM pretraining checkpoint.

        Uses the reference load_pretrained_backbone (models.py:1472-1509):
        non-strict, shape-filtered, with pos_embed interpolation when the token
        grid differs. Our checkpoints come from the framework's save_checkpoint,
        so the tensors live under 'state_dict'; the reference reads 'model'.
        Keys are also unwrapped from the DDP/wrapper prefixes.
        """
        from dbmim.models import load_pretrained_backbone

        print(f"[dbMiM] loading pretrain encoder from {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck.get("model", ck))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # SSL saves the MAE wrapper (keys under 'mae.'); strip to bare net names.
        sd = {(k[4:] if k.startswith("mae.") else k): v for k, v in sd.items()}

        net = self.model.module if hasattr(self.model, "module") else self.model
        target = net.net if hasattr(net, "net") else net
        loaded = load_pretrained_backbone(target, {"model": sd})
        print(f"[dbMiM] transferred {len(loaded)} pretrain tensors")

    # ------------------------------------------------------------------
    # SSL: masked image modeling
    # ------------------------------------------------------------------
    def _init_ssl(self, rank, checkpoint):
        from dbmim.models import DBMIM3DMAE

        crop = tuple(self.cfg.MODEL.INPUT_SIZE)
        mae = DBMIM3DMAE(
            in_channels=self.cfg.MODEL.IN_PLANES,
            volume_size=crop,
            patch_size=tuple(self.cfg.MODEL.DBMIM_PATCH_SIZE),
            embed_dim=self.cfg.MODEL.DBMIM_EMBED_DIM,
            depth=self.cfg.MODEL.DBMIM_DEPTH,
            num_heads=self.cfg.MODEL.DBMIM_NUM_HEADS,
            decoder_dim=self.cfg.MODEL.SSL_DECODER_DIM,
            mask_ratio=self.cfg.MODEL.SSL_MASK_RATIO,
            structure_weight=self.cfg.MODEL.SSL_STRUCTURE_WEIGHT,
            structure_axis_weights=list(self.cfg.MODEL.SSL_STRUCTURE_AXIS_WEIGHTS),
            membrane_weight=self.cfg.MODEL.SSL_MEMBRANE_WEIGHT,
            membrane_axis_weights=list(self.cfg.MODEL.SSL_MEMBRANE_AXIS_WEIGHTS),
            membrane_clip=self.cfg.MODEL.SSL_MEMBRANE_CLIP,
        )
        self.model = make_parallel(mae, self.cfg, self.device, rank,
                                   find_unused_parameters=True)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.SOLVER.BASE_LR, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
        self.scaler = GradScaler() if self.cfg.MODEL.MIXED_PRECESION else None

        # Learnable mask policy: a separate module with its own optimizer. It
        # reads detached tokens and is rewarded by the detached reconstruction
        # loss, so the two never share a gradient path.
        self.decision_module, self.policy_optimizer = None, None
        if self.cfg.MODEL.SSL_POLICY_ENABLED:
            from dbmim.models import DecisionModule
            dm = DecisionModule(
                embed_dim=self.cfg.MODEL.DBMIM_EMBED_DIM,
                hidden_dim=self.cfg.MODEL.SSL_POLICY_HIDDEN_DIM,
                min_mask_ratio=self.cfg.MODEL.SSL_POLICY_MIN_RATIO,
                max_mask_ratio=self.cfg.MODEL.SSL_POLICY_MAX_RATIO,
                entropy_coef=self.cfg.MODEL.SSL_POLICY_ENTROPY_COEF,
                value_coef=self.cfg.MODEL.SSL_POLICY_VALUE_COEF,
                ratio_coef=self.cfg.MODEL.SSL_POLICY_RATIO_COEF,
                reward_clip=self.cfg.MODEL.SSL_POLICY_REWARD_CLIP,
                advantage_normalize=self.cfg.MODEL.SSL_POLICY_ADV_NORMALIZE,
            )
            self.decision_module = make_parallel(dm, self.cfg, self.device, rank,
                                                 find_unused_parameters=True)
            self.policy_optimizer = torch.optim.AdamW(
                [p for p in self.decision_module.parameters() if p.requires_grad],
                lr=self.cfg.MODEL.SSL_POLICY_LR,
                weight_decay=self.cfg.MODEL.SSL_POLICY_WEIGHT_DECAY)
            if self.is_main_process:
                print(f"[dbMiM-ssl] mask policy ON: target_ratio="
                      f"{self.cfg.MODEL.SSL_POLICY_TARGET_RATIO}, range=["
                      f"{self.cfg.MODEL.SSL_POLICY_MIN_RATIO},"
                      f"{self.cfg.MODEL.SSL_POLICY_MAX_RATIO}], lr="
                      f"{self.cfg.MODEL.SSL_POLICY_LR}, weight="
                      f"{self.cfg.MODEL.SSL_POLICY_WEIGHT}, warmup="
                      f"{self.cfg.MODEL.SSL_POLICY_WARMUP_STEPS}, freeze_after="
                      f"{self.cfg.MODEL.SSL_POLICY_FREEZE_AFTER}")
        self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
        self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
        self.total_time = 0
        if self.is_main_process:
            self.monitor = build_monitor(self.cfg)
            self.monitor.load_info(self.cfg, self.model)

        # Unlabeled crops come from the framework's own dataloader: the
        # supervised sampler already handles multi-volume pools, h5/tif mixes,
        # DDP sharding and worker seeding, so SSL just drops the targets.
        # Requires DATASET.LABEL_NAME None -- VolumeDataset then returns
        # (pos, volume) and collate_fn_test packs it into .out_input.
        #
        # augmentor is None on purpose: the framework augmentations assume a
        # label is present (misalign/rescale/... call .copy() on it and crash
        # on None), and dbMiM's reference pipeline applies its own image-only
        # augmentation, which _augment_ssl_batch below reproduces.
        #
        # No StopIteration risk here: VolumeDataset sizes itself as
        # max(ITERATION_TOTAL * batch, total sliding-window positions), and for
        # a 47-volume pool the latter is ~1e9, far beyond any DDP shard.
        from connectomics.data.dataset.collate import collate_fn_test
        if self.is_main_process:
            print(f"[dbMiM-ssl] building unlabeled dataloader "
                  f"(crop={crop}, batch={self.cfg.SOLVER.SAMPLES_PER_BATCH}/rank)")
        self.dataloader = iter(build_dataloader(
            self.cfg, None, 'train', rank=rank, cf=collate_fn_test))

    def train(self):
        self.model.train()
        if self._ssl:
            self._train_ssl_mim()
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

    def _train_ssl_mim(self):
        """dbMiM masked-image-modeling loop.

        DBMIM3DMAE.forward computes the full objective internally and returns a
        MAEOutput (loss = membrane-weighted pixel loss + structure_weight *
        gradient loss), so there is no criterion here.

        With MODEL.SSL_POLICY_ENABLED the mask comes from the DecisionModule
        actor-critic instead of a fixed-ratio random draw. Both are trained in
        the same step but by separate optimizers (mirrors
        dbMiM/train_pretrain.py:277-331):
          * total = mae_loss + SSL_POLICY_WEIGHT * policy_loss
          * the policy sees detached tokens, and its reward is the detached
            per-sample reconstruction MSE, so no gradient crosses over;
          * the policy learns only in
            [SSL_POLICY_WARMUP_STEPS, SSL_POLICY_FREEZE_AFTER). After freezing,
            and with SSL_POLICY_USE_FROZEN False (the reference default), the
            mask reverts to the fixed-ratio random mask.
        """
        policy_hist = []   # (iter, mean_mask_ratio) trace for the learned policy
        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            # Framework loader yields a TestBatch; SSL uses only the image.
            image = next(self.dataloader).out_input.to(self.device, non_blocking=True)
            if self.cfg.AUGMENTOR.ENABLED:
                image = self._augment_ssl_batch(image)
            self.data_time = time.perf_counter() - self.start_time

            freeze_after = self.cfg.MODEL.SSL_POLICY_FREEZE_AFTER
            update_policy = (
                self.decision_module is not None
                and self.policy_optimizer is not None
                and iter_total >= self.cfg.MODEL.SSL_POLICY_WARMUP_STEPS
                and (freeze_after <= 0 or iter_total < freeze_after))
            if self.decision_module is not None:
                self.decision_module.train(update_policy)
            if update_policy:
                self.policy_optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                active_dm = None
                if self.decision_module is not None and (
                        update_policy or
                        (self.cfg.MODEL.SSL_POLICY_USE_FROZEN and
                         iter_total >= self.cfg.MODEL.SSL_POLICY_WARMUP_STEPS)):
                    active_dm = self.decision_module
                deterministic = (self.cfg.MODEL.SSL_POLICY_DETERMINISTIC_FROZEN
                                 and active_dm is not None and not update_policy)
                out = self.model(
                    image, decision_module=active_dm,
                    target_mask_ratio=self.cfg.MODEL.SSL_POLICY_TARGET_RATIO
                    if active_dm is not None else None,
                    deterministic_policy=deterministic)
                loss = out.loss

                policy_loss = image.new_tensor(0.0)
                if update_policy and out.decision is not None:
                    # Reward is the detached full-volume per-sample MSE.
                    per_sample = (out.pred.detach() - out.target).pow(2) \
                        .flatten(1).mean(dim=1)
                    dm = self.decision_module.module \
                        if hasattr(self.decision_module, "module") else self.decision_module
                    policy_loss = dm.policy_loss(out.decision, per_sample)
                    loss = loss + float(self.cfg.MODEL.SSL_POLICY_WEIGHT) * policy_loss

            # losses_vis feeds the monitor's loss-ratio PIE chart, so it may only
            # contain non-negative, additive loss components. The RL policy loss
            # is signed (advantage-weighted) and mask_ratio is not a loss, so
            # both are reported separately below instead.
            losses_vis = {'loss_pixel': out.pixel_loss.detach(),
                          'loss_structure': out.structure_loss.detach()}

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                if update_policy:
                    self.scaler.step(self.policy_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                if update_policy:
                    self.policy_optimizer.step()

            # Trace where the policy is driving the mask ratio -- the only
            # observable output of the learned masking.
            if out.decision is not None and self.is_main_process:
                ratio = float(out.mask.float().mean().item())
                policy_hist.append((iter_total, ratio))
                log_iv = max(1, int(self.cfg.MODEL.SSL_POLICY_LOG_INTERVAL))
                if (iter_total + 1) % log_iv == 0:
                    recent = [r for _, r in policy_hist[-log_iv:]]
                    dr = out.decision.get('mask_ratio')
                    print('[dbMiM-policy iter %06d] mask_ratio now=%.4f '
                          'mean@%d=%.4f min=%.4f max=%.4f | target=%.2f '
                          'policy_loss=%.4f %s' % (
                              iter_total, ratio, log_iv,
                              float(np.mean(recent)), float(np.min(recent)),
                              float(np.max(recent)),
                              self.cfg.MODEL.SSL_POLICY_TARGET_RATIO,
                              float(policy_loss.detach().item()),
                              '' if update_policy else '(frozen)'))
                    if dr is not None:
                        print('            per-sample policy ratio: %s' %
                              np.round(dr.detach().float().cpu().numpy(), 4).tolist())

            if hasattr(self, 'monitor'):
                self.monitor.update(iter_total, loss.detach(), losses_vis,
                                    self.optimizer.param_groups[0]['lr'])
                if out.decision is not None:
                    tb = self.monitor.logger.log_tb
                    tb.add_scalar('policy/mask_ratio',
                                  out.mask.float().mean().item(), iter_total)
                    tb.add_scalar('policy/loss',
                                  float(policy_loss.detach().item()), iter_total)
                    tb.add_scalar('policy/updating', float(update_policy), iter_total)
            if (iter_total + 1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
                self.save_checkpoint(iter_total)
            if self.is_main_process and \
                    (iter_total + 1) % self.cfg.MODEL.SSL_VIS_INTERVAL == 0:
                self._save_ssl_vis(image, out, iter_total)
            self.scheduler_step(iter_total, loss.detach())

            if self.is_main_process:
                self.iter_time = time.perf_counter() - self.start_time
                self.total_time += self.iter_time
                avg_it = self.total_time / (iter_total + 1 - self.start_iter)
                left = avg_it * (self.total_iter_nums + self.start_iter -
                                 iter_total - 1) / 3600.0
                print('[dbMiM-SSL Iter %05d] data=%.3fs iter=%.3fs avg=%.3fs '
                      'left=%.2fh loss=%.4f (pixel=%.4f struct=%.4f)' % (
                          iter_total, self.data_time,
                          time.perf_counter() - self.start_time, avg_it, left,
                          loss.item(), out.pixel_loss.item(), out.structure_loss.item()))

            del image, out, loss, losses_vis

        # No validation loop in SSL; the final encoder is what finetune loads.
        self.save_checkpoint(iter_total, is_best=True)
        if policy_hist and self.is_main_process:
            self._save_policy_trace(policy_hist)

    def _save_policy_trace(self, policy_hist):
        """Persist and plot the learned mask ratio over training."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        arr = np.asarray(policy_hist, dtype=np.float32)
        out_dir = self.cfg.DATASET.OUTPUT_PATH
        np.save(os.path.join(out_dir, 'policy_mask_ratio.npy'), arr)

        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(arr[:, 0], arr[:, 1], lw=0.6, alpha=0.4, label='mask ratio')
        if len(arr) > 20:
            k = max(1, len(arr) // 100)
            sm = np.convolve(arr[:, 1], np.ones(k) / k, mode='valid')
            ax.plot(arr[k - 1:, 0], sm, lw=1.8, label=f'smoothed (k={k})')
        ax.axhline(self.cfg.MODEL.SSL_POLICY_TARGET_RATIO, ls='--', c='gray',
                   lw=1.0, label='target')
        ax.axhline(self.cfg.MODEL.SSL_POLICY_MIN_RATIO, ls=':', c='lightgray', lw=0.8)
        ax.axhline(self.cfg.MODEL.SSL_POLICY_MAX_RATIO, ls=':', c='lightgray', lw=0.8)
        if self.cfg.MODEL.SSL_POLICY_FREEZE_AFTER > 0:
            ax.axvline(self.cfg.MODEL.SSL_POLICY_FREEZE_AFTER, c='crimson',
                       lw=1.0, ls='-.', label='policy frozen')
        ax.set_xlabel('iteration')
        ax.set_ylabel('mask ratio')
        ax.set_title('dbMiM learned mask policy')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(out_dir, 'policy_mask_ratio.png')
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f'[dbMiM-policy] wrote {p} (and policy_mask_ratio.npy)')

    def save_checkpoint(self, iteration: int, is_best: bool = False):
        """Framework save_checkpoint plus the mask policy weights.

        The DecisionModule is a separate module with its own optimizer, so it is
        absent from self.model.state_dict(); the reference stores it under its
        own 'decision_module' key and we do the same. Finetuning ignores it --
        it only reads the ViT encoder tensors.
        """
        super().save_checkpoint(iteration, is_best=is_best)
        # getattr: the finetune path never builds a decision module.
        if getattr(self, 'decision_module', None) is None or not self.is_main_process:
            return
        filename = 'checkpoint_best.pth.tar' if is_best \
            else 'checkpoint_%05d.pth.tar' % (iteration + 1)
        path = os.path.join(self.output_dir, filename)
        dm = self.decision_module.module \
            if hasattr(self.decision_module, "module") else self.decision_module
        state = torch.load(path, map_location='cpu', weights_only=False)
        state['decision_module'] = dm.state_dict()
        state['policy_optimizer'] = self.policy_optimizer.state_dict() \
            if self.policy_optimizer is not None else None
        torch.save(state, path)

    def _augment_ssl_batch(self, image):
        """Reference dbMiM image-only augmentation, applied per sample.

        The framework augmentor is bypassed for SSL (it requires labels), so
        this reproduces augment_image_and_label's image path from
        dbMiM/dbmim/datasets.py:60-132 -- rot90 xy, flips (x/y 0.5, z 0.2),
        intensity gain/bias 0.35, gamma 0.35, gaussian noise 0.25.
        """
        from dbmim.ssl_data import augment_volume
        return torch.stack([augment_volume(image[i]) for i in range(image.shape[0])])

    def _save_ssl_vis(self, image, out, iter_total):
        """5-panel PNG: input / masked patches / reconstruction / composite / |error|.

        Note on what "masked" means here: dbMiM masks at the TOKEN level --
        `encoded = where(mask, mask_token + pos_embed, tokens)` (models.py:610)
        -- so no zeroed-out image is ever fed to the encoder. Showing
        `img * (1 - mask)` would therefore depict an input that does not exist.
        Instead the mask is drawn as a red overlay on the real input, and the
        composite panel shows the reconstruction only where it had to be
        inferred (masked patches) with the visible patches left untouched.

        vmin/vmax are pinned: with autoscaling, imshow renormalizes each panel
        independently, which hides a collapsed (near-constant) reconstruction.
        The reconstruction title reports its std as a collapse indicator -- a
        value near 0 means the model is emitting the global mean.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        vis_dir = os.path.join(self.cfg.DATASET.OUTPUT_PATH, 'vis_ssl')
        os.makedirs(vis_dir, exist_ok=True)

        net = self.model.module if hasattr(self.model, "module") else self.model
        with torch.no_grad():
            # MAEOutput.pred is already a full volume (models.py: pred_volume =
            # self.unpatchify(pred)); only the per-token mask needs expanding.
            recon = out.pred.detach().float()
            patch_dim = int(np.prod(net.patch_size)) * net.in_channels
            mask_vox = net.unpatchify(
                out.mask.detach().float().unsqueeze(-1).expand(-1, -1, patch_dim))
        z = image.shape[2] // 2
        img = image[0, 0, z].detach().float().cpu().numpy()
        rec = np.clip(recon[0, 0, z].float().cpu().numpy(), 0.0, 1.0)
        m = mask_vox[0, 0, z].float().cpu().numpy()
        mb = m > 0.5

        # Input with masked patches tinted red (the encoder sees mask_token
        # there, not a hole -- this only marks WHERE information was withheld).
        overlay = np.repeat(img[..., None], 3, axis=2)
        overlay[mb] = 0.6 * overlay[mb] + 0.4 * np.array([1.0, 0.0, 0.0])
        # Reconstruction pasted only into the masked patches.
        composite = img.copy()
        composite[mb] = rec[mb]

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
        panels = [
            (img, 'input', 'gray', 0.0, 1.0),
            (overlay, f'masked patches ({m.mean():.0%} of slice)', None, None, None),
            (rec, f'reconstruction (std={rec.std():.3f})', 'gray', 0.0, 1.0),
            (composite, 'input + recon@masked', 'gray', 0.0, 1.0),
            (np.abs(rec - img) * mb, '|error| on masked', 'magma', 0.0, None),
        ]
        for ax, (arr, title, cm, vmin, vmax) in zip(axes, panels):
            ax.imshow(arr, cmap=cm, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        fig.suptitle(f'dbMiM SSL iter {iter_total} '
                     f'(mask_ratio={out.mask.float().mean().item():.3f}, '
                     f'input std={img.std():.3f})', fontsize=11)
        fig.tight_layout()
        p = os.path.join(vis_dir, f'ssl_recon_{iter_total + 1:06d}.png')
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f'[dbmim-ssl-vis] saved {p}')


def main():
    args = get_args()
    cfg = load_cfg(args, add_cfg_func=add_dbmim_config)
    device = init_devices(args, cfg)

    if args.local_rank in (0, None):
        print("PyTorch: ", torch.__version__)
        if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
            os.makedirs(cfg.DATASET.OUTPUT_PATH)
            save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    mode = 'test' if args.inference else 'train'
    trainer = DBMiMTrainer(cfg, device, mode, rank=args.local_rank,
                           checkpoint=args.checkpoint)
    if cfg.DATASET.DO_CHUNK_TITLE == 0:
        test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
        test_func() if args.inference else trainer.train()
    else:
        trainer.run_chunk(mode)
    print("Rank: {}. Device: {}. Finished!".format(args.local_rank, device))


if __name__ == "__main__":
    main()
