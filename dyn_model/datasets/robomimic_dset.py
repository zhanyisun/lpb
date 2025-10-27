from diffusion_policy.common.normalize_util import array_to_stats, get_identity_normalizer_from_stat, get_image_range_normalizer, get_range_normalizer_from_stat, robomimic_abs_action_only_normalizer_from_stat
from diffusion_policy.dataset.robomimic_replay_image_dataset import _convert_robomimic_to_replay
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from dyn_model.datasets.img_transforms import default_transform, get_train_crop_transform_resnet, get_eval_crop_transform_resnet
from diffusion_policy.common.replay_buffer import ReplayBuffer
import torch
from filelock import FileLock
import os
from torch.utils.data import Dataset
import numpy as np
import copy
import shutil

import zarr
from einops import rearrange
import random
from torch.utils.data import DataLoader

class RobomimicImageDynamicsModelDataset(Dataset):
    def __init__(self, 
                 zarr_path, 
                 num_hist=1, 
                 num_pred=1, 
                 frameskip=8,
                 view_names=['agentview', 'robot0_eye_in_hand'],
                 abs_action=False,
                 use_crop=False,
                 train=True,
                 shape_obs=None,
                 original_img_size=140,
                 cropped_img_size=128,
                 action_dim=10):
        """
        Initializes the dataset by loading data from a Zarr file and precomputing valid anchor indices.
        
        Args:
            zarr_path (str): Path to the Zarr dataset.
            horizon (int): Number of steps for history and future.
            val_ratio (float): Fraction of episodes to use for validation.
            n_neg (int): Number of negative samples (unused in this implementation).
        """
        self.abs_action = abs_action
        self.original_img_size = original_img_size
        self.cropped_img_size = cropped_img_size
        
        # Use action_dim from config
        self.original_action_dim = action_dim
        
        # Build shape_meta from provided shape_obs
        shape_meta = {
            'obs': shape_obs,
            'action': {'shape': [action_dim]}
        }
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep='rotation_6d')
        
        cache_zarr_path = zarr_path + '.zarr.zip'
        cache_lock_path = cache_zarr_path + '.lock'
        print('Acquiring lock on cache.')
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                try:
                    print('Cache does not exist. Creating!')
                    replay_buffer = _convert_robomimic_to_replay(
                        store=zarr.MemoryStore(), 
                        shape_meta=shape_meta, 
                        dataset_path=zarr_path, 
                        abs_action=abs_action, 
                        rotation_transformer=rotation_transformer)
                    print('Saving cache to disk.')
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(
                            store=zip_store
                        )
                except Exception as e:
                    shutil.rmtree(cache_zarr_path)
                    raise e
            else:
                print('Loading cached ReplayBuffer from Disk.')
                with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore())
                print('Loaded!')
                
        # Extract episode ends (1-indexed)
        self.episode_ends = replay_buffer.episode_ends[:]
        
        state_arrays = []
        for key, meta in shape_obs.items():
            if meta.get('type') == 'rgb' or 'image' in key:
                continue
            if key in replay_buffer.data:
                print(f'Adding state key: {key} with shape {replay_buffer[key].shape}')
                state_arrays.append(np.array(replay_buffer[key]))
        
        # Concatenate all state arrays
        self.states = np.concatenate(state_arrays, axis=1) if state_arrays else np.zeros((len(replay_buffer), 0))

        # Extract and process states and actions
        self.view_names = view_names
        self.imgs = {}
        for view_name in self.view_names:
            self.imgs[view_name] = np.array(replay_buffer[view_name])
        self.states_dim = self.states.shape[1]
        self.proprio_dim = self.states.shape[1]
        self.action_dim = self.original_action_dim * frameskip
        if self.abs_action:
            self.actions = np.array(replay_buffer['abs_action'])
        else:
            self.actions = np.array(replay_buffer['action'])

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

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # action
        act_stat = array_to_stats(self.actions)

        if self.abs_action:
            act_normalizer = robomimic_abs_action_only_normalizer_from_stat(act_stat)
        else:
            # already normalized
            act_normalizer = get_identity_normalizer_from_stat(act_stat)
        normalizer['act'] = act_normalizer
        # state
        state_stat = array_to_stats(self.states)
        normalizer['state'] = get_range_normalizer_from_stat(state_stat)
        for view_name in self.view_names:
            normalizer[view_name] = get_image_range_normalizer()
        return normalizer

