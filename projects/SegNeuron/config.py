"""SegNeuron-specific config knobs."""
from yacs.config import CfgNode


def add_segneuron_config(_C: CfgNode):
    _C.MODEL.FMU = "sub"               # feature merging unit: sum|sub|cat
    _C.MODEL.FUSE_MASK_IN_INFER = True # affinity = min(affinity, fg_mask) at test
    # Self-supervised pretrain checkpoint (model_weights format). Only the
    # encoder (down*) is transferred; up*/outputs* are dropped.
    _C.MODEL.SEGNEURON_PRETRAIN = ""

    # ---- Self-supervised pretraining (masked inpainting + HOG reconstruction) ----
    # When True, SegNeuronTrainer builds the pretrain MNet (dual 1-ch recon heads),
    # skips the labeled dataloader, and runs the _train_ssl_recon loop (goes
    # through the standard Trainer + run_train DDP path).
    _C.MODEL.SSL_PRETRAIN = False
    _C.MODEL.SSL_VOLUMES_TXT = ""       # unlabeled volume list (relative to root)
    _C.MODEL.SSL_VOLUMES_ROOT = "datasets/EM"
    # total = SSL_IMG_WEIGHT * MSE_img + MSE_hog, matching the official
    # Pretrain/pretrain.py: `loss = 0.2 * loss1 + loss2`. Weighting the image
    # term down (rather than the HOG term up by 5x) keeps the loss magnitude --
    # and therefore the effective step size -- identical to the reference.
    _C.MODEL.SSL_IMG_WEIGHT = 0.2
    _C.MODEL.SSL_VIS_INTERVAL = 2000

    # ---- Supervised-only cross-volume mixing augmentations ----
    # Official Train_and_Inference config sets freq_mix_prob / spa_mix_prob to
    # 0.25 (the pretrain config has neither). Both need a second patch from a
    # different source volume, so they self-disable on single-volume datasets.
    _C.AUGMENTOR.FREQ_MIX_PROB = 0.0    # FDA style mixing (targets unchanged)
    _C.AUGMENTOR.FREQ_MIX_L = 0.001     # low-freq band fraction swapped
    _C.AUGMENTOR.SPA_MIX_PROB = 0.0     # xy-window composite of image + targets
    _C.AUGMENTOR.SPA_MIX_WINDOW_FRAC = 0.546875   # official 70/128
