import os
import sys
from dyn_model.models.language_encoder import LanguageEncoder
from dyn_model.models.resnet_encoder import ResNetEncoder
import hydra
import random
import torch
import logging
import warnings


warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "proprio_encoder",
    "action_encoder",
]

def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        ckpt = torch.load(f, map_location=device)
    return ckpt

def load_model(model_ckpt, train_cfg, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print('result keys in load_model:', result.keys())
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")
    
    encoder = ResNetEncoder(
        policy_ckpt_path=train_cfg.policy_ckpt_path,
        view_names=train_cfg.view_names,
    )

    if "encoder" in result:
        encoder.load_state_dict(result["encoder"])
        print(f'loaded encoder from checkpoint {model_ckpt}')
    elif not train_cfg.model.train_encoder:
        print('using pretrained encoder')
    else:
        raise ValueError("Encoder not found in model checkpoint")
    
    language_encoder = None
    if 'libero' in train_cfg.train_data_path:
        language_encoder = LanguageEncoder(policy_ckpt_path=train_cfg.policy_ckpt_path)
        for param in language_encoder.parameters():
            param.requires_grad = False

    action_dim = 10 if train_cfg.abs_action else 7
    prior_in_chans = 9
    if 'transport' in train_cfg.train_data_path:
        action_dim = 20
        prior_in_chans = 18
    elif 'pusht' in train_cfg.train_data_path:
        action_dim = 2
        prior_in_chans = 2
    elif 'libero' in train_cfg.train_data_path:
        action_dim = 10
        prior_in_chans = 14
    total_action_dim = action_dim * train_cfg.frameskip

    language_emb_dim = 0
    if 'libero' in train_cfg.train_data_path:
        language_emb_dim = 32

    action_encoder = hydra.utils.instantiate(
        train_cfg.action_encoder,
        in_chans=total_action_dim,
        emb_dim=train_cfg.action_emb_dim,
    )
    if "action_encoder" in result:
        action_encoder.load_state_dict(result["action_encoder"])
        print(f'loaded action encoder from checkpoint {model_ckpt}')
    else:
        raise ValueError("Action encoder not found in model checkpoint")
    
    proprio_encoder = hydra.utils.instantiate(
        train_cfg.proprio_encoder,
        in_chans=prior_in_chans,
        emb_dim=train_cfg.proprio_emb_dim,
    )
    if "proprio_encoder" in result:
        proprio_encoder.load_state_dict(result["proprio_encoder"])
        print(f'loaded proprio encoder from checkpoint {model_ckpt}')
    else:
        raise ValueError("Proprio encoder not found in model checkpoint")
    
    predictor = hydra.utils.instantiate(
        train_cfg.predictor,
        num_patches=1,
        num_frames=train_cfg.num_hist,
        dim=encoder.emb_dim * len(train_cfg.view_names)
        + (proprio_encoder.emb_dim + action_encoder.emb_dim + language_emb_dim),
        visual_dim=encoder.emb_dim * len(train_cfg.view_names),
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        )
    if "predictor" in result:
        predictor.load_state_dict(result["predictor"])
        print(f'loaded predictor from checkpoint {model_ckpt}')
    else:
        raise ValueError("Predictor not found in model checkpoint")

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        view_names=train_cfg.view_names,
        use_layernorm=train_cfg.use_layernorm,
        language_encoder=language_encoder,
    )
    if train_cfg.has_predictor:
        # print(hasattr(model, 'per_view_norm'), 'per_view_norm' in result)
        if hasattr(model, 'per_view_norm') and 'per_view_norm' in result:
            model.per_view_norm.load_state_dict(result['per_view_norm'])
            print(f'loaded per_view_norm from checkpoint {model_ckpt}')
        # print(hasattr(model, 'fusion_norm'), 'fusion_norm' in result)
        if hasattr(model, 'fusion_norm') and 'fusion_norm' in result:
            model.fusion_norm.load_state_dict(result['fusion_norm'])
            print(f'loaded fusion_norm from checkpoint {model_ckpt}')


    model.to(device)
    return model
