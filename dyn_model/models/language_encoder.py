import torch
import argparse
import sys
import os
import pathlib
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
import hydra
from omegaconf import DictConfig, OmegaConf
import dill
from einops import rearrange, repeat
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class LanguageEncoder(torch.nn.Module):
    def __init__(self, policy_ckpt_path):
        super().__init__()
        self.policy_ckpt_path = policy_ckpt_path

        with open(self.policy_ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill)
            cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(payload['cfg'], output_dir='debug_obs_encoder')
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy.to(device)
        policy.eval()

        self.text_model = policy.text_model
        self.text_model.eval()  

        del workspace
        del policy
        torch.cuda.empty_cache()

    def forward(self, text_tokens):
        text_latents = self.text_model(text_tokens)  # (batch_size, 32)
        text_latents = text_latents.unsqueeze(1) # dummy patch dim

        return text_latents

if __name__ == "__main__":
    ckpt = 'data/outputs/2025.06.04/03.33.53_train_diffusion_unet_hybrid_libero_image/checkpoints/0.ckpt'
    encoder = LanguageEncoder(ckpt)
    vocab_size = 49408
    text_tokens = {
        "input_ids": torch.randint(0, vocab_size, (8, 30)).to('cuda'),  # batch of 8, sequence length 30
        "attention_mask": torch.ones(8, 30).to('cuda')
    }
    features = encoder(text_tokens)  # shape: (8, 32)
    print("Text features shape:", features.shape)  # should be (8, 32)