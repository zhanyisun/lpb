import os
from pathlib import Path 
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.robomimic_replay_image_dataset import _convert_actions, undo_transform_action
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from dyn_model.datasets.robomimic_dset import RobomimicImageDynamicsModelDataset
import robomimic.utils.file_utils as FileUtils
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from dyn_model.datasets.img_transforms import default_transform, get_eval_crop_transform, get_eval_crop_transform_resnet
from dyn_model.plan import load_model
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

import hydra
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
import copy
import numpy as np
from einops import rearrange


class Planner:
    def __init__(
        self,
        demo_dataset_config,
        dynamics_model_ckpt,
        action_step=8,
        output_dir='debug/',
        demo_dataset_path=None
    ):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # demo dataset
        self.demo_dataset_config = demo_dataset_config
        self.demo_dataset_path = demo_dataset_path
        dynamics_model_dir = os.path.dirname(os.path.dirname(dynamics_model_ckpt))
        with open(os.path.join(dynamics_model_dir, "hydra.yaml"), "r") as f:
            model_cfg = OmegaConf.load(f)
            assert model_cfg.abs_action == self.demo_dataset_config.abs_action
        self.dyn_model = load_model(Path(dynamics_model_ckpt), model_cfg, device=self.device)
        if not model_cfg.model.train_encoder and model_cfg.encoder_ckpt_path is not None:
            encoder_ckpt = torch.load(model_cfg.encoder_ckpt_path, map_location='cuda')
            self.dyn_model.encoder.load_state_dict(encoder_ckpt['encoder'])
            print('loaded encoder from ', model_cfg.encoder_ckpt_path)

        self.dyn_model = self.accelerator.prepare(self.dyn_model)
        self.dyn_model.eval()
        
        # Store shape_obs from model config for dynamic observation extraction
        if hasattr(model_cfg, 'env') and hasattr(model_cfg.env, 'shape_obs'):
            self.shape_obs = model_cfg.env.shape_obs
        else:
            self.shape_obs = None

        # normalizer
        wm_normalizer = LinearNormalizer()
        wm_normalizer.load_state_dict(torch.load(os.path.join(dynamics_model_dir, "normalizer.pth")))
        self.dyn_model_normalizer = wm_normalizer.to(self.device)
        self.policy_action_normalizer = LinearNormalizer()

        # eval_env
        self.abs_action = self.demo_dataset_config.abs_action
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=demo_dataset_config.dataset_path)
            
        self.use_crop = model_cfg.use_crop
        self.original_img_size = model_cfg.original_img_size
        self.cropped_img_size = model_cfg.cropped_img_size
        if self.use_crop:
            self.img_transform = get_eval_crop_transform_resnet(original_img_size=self.original_img_size, 
                                                        cropped_img_size=self.cropped_img_size)

        self.view_names = model_cfg.view_names

        self.env_name = env_meta["env_name"]
        print('env_name ', self.env_name)

        self.frameskip = model_cfg.frameskip
        self.exec_step = action_step
        print('self.demo_dataset_config.horizon ', self.demo_dataset_config.horizon)
        self.horizon = self.demo_dataset_config.horizon // 16

        self.get_demo_latents()
        self.timestep = 0
        self.rotation_transformer = RotationTransformer(
                from_rep='axis_angle', to_rep='rotation_6d')
        self.output_dir = output_dir
        self.idx = 0

    def set_policy_action_normalizer(self, policy_action_normalizer):
        self.policy_action_normalizer = policy_action_normalizer

    def get_demo_latents(self,):
        demo_dataset: BaseImageDataset
        
        # Use demo_dataset_path from config if provided, otherwise fall back to hardcoded paths
        if self.demo_dataset_path:
            self.demo_dataset_config.dataset_path = self.demo_dataset_path

        self.demo_dataset_config.val_ratio = 0
        self.demo_dataset_config.horizon = 1
        self.demo_dataset_config.n_obs_steps = 1
        self.demo_dataset_config.pad_before = 0
        self.demo_dataset_config.pad_after = 0

        demo_dataset = hydra.utils.instantiate(self.demo_dataset_config)
        print('len(demo_dataset) ', len(demo_dataset))
        demo_loader = DataLoader(demo_dataset, batch_size=64, shuffle=False, num_workers=4)

        demo_visual_latents = []
        demo_proprio_latents = []
        demo_images = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(demo_loader):
                obs = batch['obs']
                obs = dict_apply(obs, lambda x: x[:, -1:, ...])
                proprio_arrays = []
                for key, meta in self.shape_obs.items():
                    if meta.get('type') == 'rgb' or 'image' in key:
                        continue
                    if key in obs:
                        proprio_arrays.append(obs[key].to(self.device))
                
                proprio = torch.cat(proprio_arrays, dim=-1) if proprio_arrays else torch.zeros((obs[list(obs.keys())[0]].shape[0], 0)).to(self.device)
                visual = {}

                for view_name in self.view_names:
                    bs, h = obs[view_name].shape[:2]
                    visual[view_name] = obs[view_name].to(self.device)
                    visual[view_name] = self.dyn_model_normalizer[view_name].normalize(visual[view_name])
                    visual[view_name] = self.img_transform(visual[view_name].view(-1, 3, self.original_img_size, self.original_img_size))
                    visual[view_name] = visual[view_name].view(-1, 1, 3, self.cropped_img_size, self.cropped_img_size)
                visual_cpu = {k: v.cpu() for k, v in visual.items()}
                demo_images.append(visual_cpu)

                obs_wm = {'visual': visual, 'proprio': proprio}
                obs_wm['proprio'] = self.dyn_model_normalizer['state'].normalize(obs_wm['proprio'])

                encode_obs = self.dyn_model.encode_obs(obs_wm)
                demo_visual_latents.append(encode_obs['visual'].cpu())
                demo_proprio_latents.append(encode_obs['proprio'].cpu())
                torch.cuda.empty_cache()

        self.demo_visual_latents = torch.cat(demo_visual_latents, dim=0)
        self.demo_proprio_latents = torch.cat(demo_proprio_latents, dim=0)

        if len(self.demo_visual_latents.shape) > 2:
            self.demo_visual_latents = self.demo_visual_latents.reshape(self.demo_visual_latents.size(0), -1)
            if 'ToolHang' in self.env_name:
                self.demo_visual_latents = self.demo_visual_latents[...,512:]
            elif 'Transport' in self.env_name:
                self.demo_visual_latents = self.demo_visual_latents[...,:1024]

        print('demo_visual_latents shape ', self.demo_visual_latents.shape)
        print('demo_proprio_latents shape ', self.demo_proprio_latents.shape)
        self.demo_images = {
            key: torch.cat([d[key] for d in demo_images], dim=0)
            for key in demo_images[0].keys()
        }
        for key in self.demo_images.keys():
            print(key, self.demo_images[key].shape)

        del demo_dataset
 
    def compute_current_reward(self, current_obs):
        with torch.no_grad():
            current_obs_wm = self.prepare_obs(current_obs, 1)
            encode_obs = self.dyn_model.encode_obs(current_obs_wm)
            current_visual_latent = encode_obs['visual'] # (1, 1, 196, 382*2)
            reward, _ = self.compute_nn_reward(current_visual_latent.squeeze(1))
        return reward

    def compute_nn_reward(self, current_visual_latent):
        if len(current_visual_latent.shape) > 2:
            current_visual_latent = current_visual_latent.reshape(current_visual_latent.size(0), -1)
            if 'ToolHang' in self.env_name or 'Square' in self.env_name:
                current_visual_latent = current_visual_latent[..., 512:]
            elif 'Transport' in self.env_name:
                current_visual_latent = current_visual_latent[...,:1024]

        device = current_visual_latent.device
        chunk_size = 2048 
        
        global_min_cost = None
        global_min_idx = None
        
        # Process the demo_visual_latents in chunks.
        for start in range(0, self.demo_visual_latents.shape[0], chunk_size):
            demo_chunk = self.demo_visual_latents[start:start+chunk_size].to(device, non_blocking=True)
            dist = torch.cdist(current_visual_latent, demo_chunk, p=2)  # shape: (B, chunk_size)
            cost, idx = dist.min(dim=-1)  # cost: (B,), idx: (B,)

            if global_min_cost is None:
                global_min_cost = cost
                global_min_idx = idx + start  # Adjust index offset.
            else:
                # Update the global minimum for each sample.
                mask = cost < global_min_cost
                global_min_cost[mask] = cost[mask]
                global_min_idx[mask] = idx[mask] + start
        
        reward = -global_min_cost
        return reward, global_min_idx

    def compute_loss(self, sample, current_obs):
        init_actions_normalized = sample[:, 1:1+self.horizon * self.frameskip]
        init_actions_unnormalized = self.policy_action_normalizer.unnormalize(init_actions_normalized)
        init_actions = self.dyn_model_normalizer['act'].normalize(init_actions_unnormalized)
        action_batch = rearrange(init_actions, 'b (h f) a -> b h (f a)', f=self.frameskip, h=self.horizon)
        batch_size = init_actions.shape[0]
        current_obs_wm = self.prepare_obs(current_obs, batch_size)
        
        act_0 = action_batch[:, :1, :]  # use the first action
        z = self.dyn_model.encode(current_obs_wm, act_0)

        total_rew = torch.zeros(batch_size, device=sample.device, dtype=torch.float)
        t = 0
        inc = 1

        latent_list = [z]

        while t < 1:
            z_pred = self.dyn_model.predict(latent_list[-1])
            z_new = z_pred[:, -inc:, ...]
            z_obs, _ = self.dyn_model.separate_emb(z_new)
            rew, idx = self.compute_nn_reward(z_obs['visual'])
            total_rew += rew
            t += inc
        cost = -1 * total_rew
        loss = cost.mean()
        self.idx += 1
        return loss

    def prepare_obs(self, current_obs, action_shape):
        # Dynamically extract low-dimensional observations
        # Process observations in the same order as in the dataset's shape_obs
        proprio_arrays = []

        for key, meta in self.shape_obs.items():
            if meta.get('type') == 'rgb' or 'image' in key:
                continue
            if key in current_obs:
                proprio_arrays.append(current_obs[key])

        proprio = torch.cat(proprio_arrays, dim=-1).to('cuda') if proprio_arrays else torch.zeros((1, 0)).to('cuda')

        visual = {}
        for view_name in self.view_names:
            visual[view_name] = current_obs[view_name].to('cuda')
            visual[view_name] = self.dyn_model_normalizer[view_name].normalize(visual[view_name].to('cuda'))
            visual[view_name] = self.img_transform(visual[view_name].view(-1, 3, self.original_img_size, self.original_img_size))
            visual[view_name] = visual[view_name].view(-1, 1, 3, self.cropped_img_size, self.cropped_img_size)

        current_obs_wm = {'visual': visual, 'proprio': proprio}
        current_obs_wm['proprio'] = self.dyn_model_normalizer['state'].normalize(current_obs_wm['proprio'])

        current_obs_wm['proprio'] = current_obs_wm['proprio'].expand(action_shape, -1, -1)
        current_obs_wm['visual'] = {key: value.expand(action_shape, -1, -1, -1, -1) for key, value in current_obs_wm['visual'].items()}
        return current_obs_wm
