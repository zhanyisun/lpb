from diffusion_policy.common.normalize_util import array_to_stats, get_identity_normalizer_from_stat, get_image_range_normalizer, get_range_normalizer_from_stat, robomimic_abs_action_only_normalizer_from_stat
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from dyn_model.datasets.img_transforms import default_transform, get_train_crop_transform_resnet, get_eval_crop_transform_resnet
from diffusion_policy.common.replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F

import os
from torch.utils.data import Dataset
import numpy as np


class PushTImageDynamicsModelDataset(Dataset):
    def __init__(self, 
                 zarr_path, 
                 num_hist=1, 
                 num_pred=1, 
                 frameskip=8,
                 view_names=['image'],
                 abs_action=False,
                 use_crop=False,
                 train=True,
                 original_img_size=140,
                 cropped_img_size=128,
                 action_dim=2):
        """
        Initializes the dataset by loading data from a Zarr file and precomputing valid anchor indices.
        
        Args:
            zarr_path (str): Path to the Zarr dataset.
            horizon (int): Number of steps for history and future.
            val_ratio (float): Fraction of episodes to use for validation.
            n_neg (int): Number of negative samples (unused in this implementation).
        """
        self.abs_action = abs_action
        self.original_action_dim = action_dim
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action', 'n_contacts'])
        # Extract episode ends (1-indexed)
        self.episode_ends = self.replay_buffer.episode_ends[:]

        # Extract and process states and actions
        self.view_names = view_names
        self.imgs = {}
        for view_name in self.view_names:
            self.imgs[view_name] = np.array(self.replay_buffer['img'])

        self.states = self.replay_buffer['state'][:,:2].astype(np.float32) # agent_pos
        self.actions = self.replay_buffer['action'].astype(np.float32) # action
        self.action_dim = self.original_action_dim * frameskip
        self.proprio_dim = self.states.shape[1]

        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip
        self.num_frames = num_hist + num_pred
        self.use_crop = use_crop
        self.train = train
        
        # Convert episode_ends to zero-indexed format and store the start and end indices of each trajectory
        self.episode_start_indices = np.concatenate(([0], self.episode_ends[:-1]))
        self.episode_end_indices = self.episode_ends - 1  # last index of each trajectory
        
        # Precompute valid anchor indices
        self.valid_anchor_indices = []
        for start, end in zip(self.episode_start_indices, self.episode_end_indices):
            # Valid anchors are from start + horizon_history to end - horizon_future
            anchor_start = start
            anchor_end = end - num_pred * self.frameskip
            if anchor_end >= anchor_start:
                anchors = np.arange(anchor_start, anchor_end)
                self.valid_anchor_indices.extend(anchors)
        self.valid_anchor_indices = np.array(self.valid_anchor_indices)
        self.num_valid = len(self.valid_anchor_indices)
        self.transform = default_transform()
        self.original_img_size = original_img_size
        self.cropped_img_size = cropped_img_size
        if self.use_crop:
            if self.train:
                self.transform = get_train_crop_transform_resnet(original_img_size, cropped_img_size)
            else:
                self.transform = get_eval_crop_transform_resnet(original_img_size, cropped_img_size)
        
    def __len__(self):
        """
        Returns the number of valid anchor samples.
        """
        return self.num_valid
    
    def __getitem__(self, idx):
        start = self.valid_anchor_indices[idx]
        end = start + (self.num_frames) * self.frameskip
        obs_indices = list(range(start, end, self.frameskip))
        action_indices = list(range(start, end))
        action_indices[-self.frameskip:] = [obs_indices[-1] - 1] * self.frameskip
        obs = {}
        obs['visual'] = {}
        for view_name in self.view_names:
            obs['visual'][view_name] = self.imgs[view_name][obs_indices]
            obs['visual'][view_name] = np.moveaxis(obs['visual'][view_name],-1,1).astype(np.float32)/255
            obs['visual'][view_name] = torch.from_numpy(obs['visual'][view_name])
            
        obs['proprio'] = self.states[obs_indices]
        obs['proprio'] = torch.from_numpy(obs['proprio'].astype(np.float32))
        act = self.actions[action_indices]
        state = self.states[obs_indices]
        act = torch.from_numpy(act.astype(np.float32))
        state = torch.from_numpy(state.astype(np.float32))

        return tuple([obs, act, state])

    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer:
        data = {
            'act': self.actions,
            'state': self.states,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        for view_name in self.view_names:
            normalizer[view_name] = get_image_range_normalizer()
        return normalizer

