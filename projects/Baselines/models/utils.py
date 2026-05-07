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


