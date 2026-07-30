"""dbMiM-specific config knobs.

dbMiM (masked-image pretraining + anisotropic UNETR affinity finetuning) keeps
its own loss and normalization because both differ from the framework defaults:

  * the finetune loss is MSE-on-sigmoid plus MAWS (membrane-aware spatial
    weighting), whose weight map is derived from the raw IMAGE gradient rather
    than from labels -- so it cannot be expressed via MODEL.WEIGHT_OPT, which
    computes weights from the target;
  * inputs use a per-crop 1/99-percentile contrast stretch, not the
    mean/std normalization in DATASET.MEAN / DATASET.STD.
"""
from yacs.config import CfgNode


def add_dbmim_config(_C: CfgNode):
    # ---- Architecture (UNETREMAffinityNet / DBMIM3DMAE share the ViT encoder) ----
    _C.MODEL.DBMIM_PATCH_SIZE = [4, 16, 16]
    _C.MODEL.DBMIM_EMBED_DIM = 192
    _C.MODEL.DBMIM_DEPTH = 6
    _C.MODEL.DBMIM_NUM_HEADS = 6
    _C.MODEL.DBMIM_FEATURE_SIZE = 32
    _C.MODEL.DBMIM_DROPOUT = 0.05
    # Number of ResidualAnisotropicBlock3D refinement blocks in the affinity
    # head. Clamped to >=1 by EMAffinityHead3D, so 0 still builds one block.
    _C.MODEL.DBMIM_EM_REFINE_DEPTH = 2
    # Per-channel output bias init (z, y, x). The negative z bias starts the
    # anisotropic z-affinity channel more conservative.
    _C.MODEL.DBMIM_CHANNEL_BIAS_INIT = [-0.2, 0.0, 0.0]

    # Encoder checkpoint from dbMiM pretraining. Only tensors whose name and
    # shape match are loaded (pos_embed is interpolated when the grid differs).
    _C.MODEL.DBMIM_PRETRAIN = ""

    # ---- Self-supervised pretraining (masked image modeling) ----
    # When True, the trainer builds DBMIM3DMAE, skips the labeled dataloader,
    # and runs the MIM loop through the standard run_train DDP path.
    _C.MODEL.SSL_PRETRAIN = False
    _C.MODEL.SSL_VOLUMES_TXT = ""      # unlabeled volume list (relative to root)
    _C.MODEL.SSL_VOLUMES_ROOT = "datasets/EM"
    _C.MODEL.SSL_MASK_RATIO = 0.75
    _C.MODEL.SSL_DECODER_DIM = 192
    # dbMiM term 1: pixel loss is weighted by an image-gradient membrane proxy.
    _C.MODEL.SSL_MEMBRANE_WEIGHT = 1.35
    _C.MODEL.SSL_MEMBRANE_AXIS_WEIGHTS = [0.25, 1.0, 1.0]
    _C.MODEL.SSL_MEMBRANE_CLIP = 5.0
    # dbMiM term 2: additive gradient/structure-consistency term.
    _C.MODEL.SSL_STRUCTURE_WEIGHT = 0.2
    _C.MODEL.SSL_STRUCTURE_AXIS_WEIGHTS = [0.5, 1.0, 1.0]
    _C.MODEL.SSL_VIS_INTERVAL = 2000

    # ---- Learnable mask policy (DecisionModule actor-critic) ----
    # Jointly trained with the MAE but through a SEPARATE optimizer: the policy
    # reads detached tokens and its reward is the detached reconstruction loss,
    # so no gradient flows between the two. This is the reference's headline
    # recipe (decision.enabled: true in pretrain_public_em_membrane_r16.yaml,
    # the pretraining behind the best-VOI R48 finetune).
    _C.MODEL.SSL_POLICY_ENABLED = False
    _C.MODEL.SSL_POLICY_HIDDEN_DIM = 256
    _C.MODEL.SSL_POLICY_LR = 5.0e-04
    _C.MODEL.SSL_POLICY_WEIGHT_DECAY = 1.0e-04
    _C.MODEL.SSL_POLICY_WEIGHT = 0.05      # policy_loss coefficient in the total
    _C.MODEL.SSL_POLICY_TARGET_RATIO = 0.75
    _C.MODEL.SSL_POLICY_MIN_RATIO = 0.4
    _C.MODEL.SSL_POLICY_MAX_RATIO = 0.9
    # Actor-critic coefficients (reference defaults). The reward is the raw
    # per-sample reconstruction MSE, so with reward_clip 0 and
    # advantage_normalize False the critic term can diverge -- the two knobs
    # below are the stabilizers to reach for if the policy saturates at a
    # ratio bound.
    _C.MODEL.SSL_POLICY_ENTROPY_COEF = 0.01
    _C.MODEL.SSL_POLICY_VALUE_COEF = 0.5
    _C.MODEL.SSL_POLICY_RATIO_COEF = 0.25
    _C.MODEL.SSL_POLICY_REWARD_CLIP = 0.0        # >0 clamps reward to +-value
    _C.MODEL.SSL_POLICY_ADV_NORMALIZE = False    # standardize advantage in-batch
    _C.MODEL.SSL_POLICY_WARMUP_STEPS = 0   # steps before the policy starts learning
    # After this step the policy stops being updated. With
    # SSL_POLICY_USE_FROZEN False (the reference default) the mask also reverts
    # to the fixed-ratio random mask from then on.
    _C.MODEL.SSL_POLICY_FREEZE_AFTER = 40000
    _C.MODEL.SSL_POLICY_USE_FROZEN = False
    _C.MODEL.SSL_POLICY_DETERMINISTIC_FROZEN = True
    # How often to log the policy's learned mask-ratio statistics.
    _C.MODEL.SSL_POLICY_LOG_INTERVAL = 200

    # ---- Finetune loss: MSE on sigmoid + MAWS ----
    _C.MODEL.DBMIM_MEMBRANE_WEIGHT = 0.75
    _C.MODEL.DBMIM_MEMBRANE_AXIS_WEIGHTS = [0.25, 1.0, 1.0]
    _C.MODEL.DBMIM_MEMBRANE_CLIP = 4.0
    _C.MODEL.DBMIM_MEMBRANE_NORMALIZE = True
    # Relative weight per affinity channel (z, y, x) inside the weighted mean.
    _C.MODEL.DBMIM_CHANNEL_WEIGHTS = [1.35, 1.0, 1.0]

    # ---- Optimizer: constant LR, encoder at a lower rate ----
    # The reference has no LR scheduler and no warmup; set
    # SOLVER.LR_SCHEDULER_NAME accordingly in the config/script.
    _C.SOLVER.DBMIM_ENCODER_LR = 1.0e-05
    _C.SOLVER.DBMIM_ENCODER_PARAM_PREFIXES = [
        "pos_embed", "patch_embed", "encoder_blocks", "norm"]

    # ---- Data ----
    # Per-crop percentile contrast stretch (reference normalize_volume).
    _C.DATASET.DBMIM_PERCENTILE_NORM = True
    _C.DATASET.DBMIM_PERCENTILE_LO = 1.0
    _C.DATASET.DBMIM_PERCENTILE_HI = 99.0
