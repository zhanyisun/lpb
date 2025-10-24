from diffusion_policy.common.normalize_util import array_to_stats, get_identity_normalizer_from_stat, get_image_range_normalizer, get_range_normalizer_from_stat, robomimic_abs_action_only_normalizer_from_stat
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from dyn_model.datasets.img_transforms import default_transform, get_train_crop_transform, get_eval_crop_transform, get_train_crop_transform_resnet, get_eval_crop_transform_resnet
from diffusion_policy.common.replay_buffer import ReplayBuffer
import torch
from filelock import FileLock
import os
from torch.utils.data import Dataset
import numpy as np
import copy
import shutil
import h5py
import multiprocessing

from tqdm import tqdm
import concurrent.futures
from transformers import AutoTokenizer
import glob
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.pose_util import axisangle2quat_batch

from dyn_model.datasets.language_goals import language_goals_list

import zarr
from einops import rearrange
import random
from torch.utils.data import DataLoader

class LiberoImageDynamicsModelDataset(Dataset):
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
                 original_img_size=128,
                 cropped_img_size=114,
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
        
        cache_zarr_path = zarr_path
        cache_lock_path = cache_zarr_path + '.lock'
        print('Acquiring lock on cache.')
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                try:
                    print("Cache does not exist. Creating!")

                    replay_buffer = _convert_robomimic_to_replay(
                        store=zarr.MemoryStore(),
                        shape_meta=shape_meta,
                        abs_action=abs_action,
                        rotation_transformer=rotation_transformer,
                    )
                    print("Saving cache to disk.")
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                except Exception as e:
                    shutil.rmtree(cache_zarr_path)
                    raise e
            else:
                print("Loading cached ReplayBuffer from Disk.")
                with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore()
                    )
                print("Loaded!")
                
        self.episode_ends = replay_buffer.episode_ends[:]
        ee_ori = np.array(replay_buffer['ee_ori'])
        ee_pos = np.array(replay_buffer['ee_pos'])
        joint_states = np.array(replay_buffer['joint_states'])
        self.states = np.concatenate((ee_ori, ee_pos, joint_states), axis=1)
        self.proprio_dim = self.states.shape[1]

        self.replay_buffer = replay_buffer
        self.view_names = view_names

        self.language = np.array(replay_buffer['language'])
        self.action_dim = self.original_action_dim * frameskip
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
            obs['visual'][view_name] = self.replay_buffer[view_name].get_orthogonal_selection((obs_indices,))
            obs['visual'][view_name] = np.moveaxis(obs['visual'][view_name],-1,1).astype(np.float32)/255
            obs['visual'][view_name] = np.rot90(obs['visual'][view_name], k=2, axes=(2, 3)).copy()
            obs['visual'][view_name] = np.flip(obs['visual'][view_name], axis=3).copy()
            obs['visual'][view_name] = torch.from_numpy(obs['visual'][view_name])
            
        obs['proprio'] = self.states[obs_indices]
        obs['proprio'] = torch.from_numpy(obs['proprio'].astype(np.float32))
        obs['language'] = self.language[obs_indices]
        obs['language'] = torch.from_numpy(obs['language'].astype(np.float32))
        act = self.actions[action_indices]
        state = self.states[obs_indices]
        act = torch.from_numpy(act.astype(np.float32))
        state = torch.from_numpy(state.astype(np.float32))

        return tuple([obs, act, state])

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # action
        act_stat = array_to_stats(self.actions)

        act_normalizer = robomimic_abs_action_only_normalizer_from_stat(act_stat)
        normalizer['act'] = act_normalizer
        # state
        state_stat = array_to_stats(self.states)
        normalizer['state'] = get_range_normalizer_from_stat(state_stat)
        for view_name in self.view_names:
            normalizer[view_name] = get_image_range_normalizer()
        return normalizer



def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    language_emb_model='clip',
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    file_handles = []  # Store file handles if you need to keep them open
    demos_all = {}
    language_all = {}
    count = 0

    if language_emb_model == "clip":
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
    else:
        raise NotImplementedError(f"Language model {language_emb_model} not implemented")

    from pathlib import Path
    dataset_paths = []
    # data_dirs = [
    #     '/store/real/zhanyis/diffusion_policy/data/libero/libero_10',
    #     '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.06_07.49.26_ckpt_100',
    #     '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.06_07.49.26_ckpt_140',
    #     '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.06_07.49.26_ckpt_170',
    #     '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.06_07.49.26_ckpt_200',
    #     '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.06_07.49.26_ckpt_250',
    #     '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.06_07.49.26_ckpt_280',
    #     '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.06_07.49.26_ckpt_310',
    #     '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.06_07.49.26_ckpt_340',
    # ]
    data_dirs = [
        '/store/real/zhanyis/diffusion_policy/data/libero/libero_10',
        '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.04.03.33.53_ckpt_40',
        '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.04.03.33.53_ckpt_80',
        '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.04.03.33.53_ckpt_120',
        '/store/real/zhanyis/diffusion_policy/data/libero/libero_10_2025.06.04.03.33.53_ckpt_160',
    ]

    for d in data_dirs:
        root_path = Path(d)
        # rglob will recurse into subdirectories
        for p in root_path.rglob("*.hdf5"):
            print(f"Found dataset file: {p}")
            dataset_paths.append(str(p))
    
    for dataset_path_each in dataset_paths:
        language_goal = " ".join(dataset_path_each.split("/")[-1][:-10].split("_"))
        assert language_goal in language_goals_list, f"Language goal {language_goal} not found in language_goals"
        print(f"Loading {dataset_path_each}")
        file = h5py.File(
            dataset_path_each, "r"
        )  # Open the file without closing it immediately
        file_handles.append(
            file
        )  # Keep track of the file handle to avoid it being closed
        demos = file["data"]

        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            demos_all[f"demo_{count}"] = demo
            language_all[f"demo_{count}"] = language_goal
            count += 1
    print("Total demos:", count)

        
    seq_max_len = 30

    if language_emb_model == "clip":
        language_all_tokens = [
            tokenizer(
                language_all[f"demo_{i}"],
                padding="max_length",
                max_length=seq_max_len,
                return_tensors="pt",
            )
            for i in range(len(language_all))
        ]
        language_input_ids = [
            item.input_ids.unsqueeze(1) for item in language_all_tokens
        ]
        language_attention_mask = [
            item.attention_mask.unsqueeze(1) for item in language_all_tokens
        ]
    else:
        raise NotImplementedError(f"Language model {language_emb_model} not implemented")

    demos = demos_all
    episode_ends = list()
    prev_end = 0
    for i in range(len(demos)):
        demo = demos[f"demo_{i}"]
        episode_length = demo["actions"].shape[0]
        episode_end = prev_end + episode_length
        prev_end = episode_end
        episode_ends.append(episode_end)
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    # save lowdim data
    for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
        data_key = "obs/" + key
        if key == "action":
            data_key = "actions"
            this_language_data = list()
        if key == "language":
            continue
        this_data = list()
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            demo_key_data = demo[data_key][:].astype(np.float32)
            if 'ori' in key:
                if demo_key_data.shape[-1] == 3:
                    demo_key_data = axisangle2quat_batch(demo_key_data)
                    assert demo_key_data.shape[-1] == 4, f"Expected quaternion shape, got {demo_key_data.shape}"
                else:
                    assert demo_key_data.shape[-1] == 4, f"Expected quaternion shape, got {demo_key_data.shape}"
            this_data.append(demo_key_data)

            if key == "action":
                if language_emb_model == "clip":
                    language_tokens = torch.cat(
                        [language_input_ids[i], language_attention_mask[i]], dim=1
                    )
                    this_language_data.append(
                        language_tokens.repeat(this_data[-1].shape[0], 1, 1)
                    )
                else:
                    raise NotImplementedError(f"Language model {language_emb_model} not implemented")

        this_data = np.concatenate(this_data, axis=0)

        if key == "action":
            this_data = _convert_actions(
                raw_actions=this_data,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
            )

            assert this_data.shape == (n_steps,) + tuple(shape_meta["action"]["shape"])

            this_language_data = np.concatenate(this_language_data, axis=0)
            if language_emb_model == "clip":
                assert this_language_data.shape == (n_steps,) + tuple([2, seq_max_len])
            else:
                raise NotImplementedError(f"Language model {language_emb_model} not implemented")
        else:
            print('key ', key, ' shape ', this_data.shape)
            assert this_data.shape == (n_steps,) + tuple(
                shape_meta["obs"][key]["shape"]
            )
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype,
        )

        if key == "action":
            _ = data_group.array(
                name="language",
                data=this_language_data,
                shape=this_language_data.shape,
                chunks=this_language_data.shape,
                compressor=None,
                dtype=this_language_data.dtype,
            )

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    with tqdm(
        total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0
    ) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()
            for key in rgb_keys:
                data_key = "obs/" + key
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                this_compressor = Jpeg2k(level=50)
                img_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w, c),
                    chunks=(1, h, w, c),
                    compressor=this_compressor,
                    dtype=np.uint8,
                )

                for episode_idx in range(len(demos)):
                    demo = demos[f"demo_{episode_idx}"]
                    hdf5_arr = demo["obs"][key]
                    for hdf5_idx in range(hdf5_arr.shape[0]):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError("Failed to encode image!")
                            pbar.update(len(completed))

                        zarr_idx = episode_starts[episode_idx] + hdf5_idx
                        futures.add(
                            executor.submit(
                                img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                            )
                        )
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to encode image!")
            pbar.update(len(completed))

    # Ensure you close all files when you're done with them
    for file in file_handles:
        file.close()

    replay_buffer = ReplayBuffer(root)
    return replay_buffer
