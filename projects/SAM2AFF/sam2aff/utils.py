from connectomics.model.build import MODEL_MAP
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging
import os
import torch

def return_loss(model_forward):
    """warper for loss calculation.
    For compatibility with the Trainer in main_ori.py (forward then calculate loss) 
    and CustomTrainer in main.py (calculate loss in forward function).
    """
    def forward(self, inputs, target=None, weight=None, criterion=None):
        pred = model_forward(self, inputs)
        if criterion is None:       # test mode
            return pred
        loss, losses_vis = criterion(pred, target, weight)
        return pred, loss, losses_vis
    return forward 


def register_model(name):
    def register_model_cls(cls):
        MODEL_MAP[name] = cls
        return cls
    return register_model_cls

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


def build_sam2aff(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="train",
    net= "sam2.sam2_video_predictor.SAM2VideoPredictor",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    image_size=256,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_="+net,
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    # if apply_postprocessing:
    #     hydra_overrides_extra = hydra_overrides_extra.copy()
    #     hydra_overrides_extra += [
    #         # dynamically fall back to multi-mask if the single mask is not stable
    #         "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
    #         "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
    #         "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
    #         # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
    #         "++model.binarize_mask_from_pts_for_mem_enc=true",
    #         # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
    #         "++model.fill_hole_area=8",
    #     ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    cfg.model.image_size = image_size
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model