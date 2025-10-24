import os
import sys
from dyn_model.datasets.language_goals import language_goals_list
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

## here we just use the same env wrapper as robomimic
from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)
from diffusion_policy.env_runner.libero_bddl_mapping import bddl_file_name_dict

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils



current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
libero_path = os.path.join(parent_dir, "LIBERO")
sys.path.append(libero_path)
from libero.libero.envs.bddl_base_domain import TASK_MAPPING
from diffusion_policy.env_runner.libero_bddl_mapping import bddl_file_name_dict


def create_env(env_meta, shape_meta, enable_render=True, render_obs_key='agentview_image',
               fps=10, crf=22, n_obs_steps=2, n_action_steps=8, max_steps=400):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    if env_meta["bddl_file"] not in bddl_file_name_dict.values():
        env_meta["bddl_file"] = bddl_file_name_dict[env_meta["bddl_file"]]
        env_meta["env_kwargs"]["bddl_file_name"] = env_meta["bddl_file"]

    raw_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    raw_env.env.hard_reset = False

    video_recorder = VideoRecorder.create_h264(
        fps=fps,
        codec='h264',
        input_pix_fmt='rgb24',
        crf=crf,
        thread_type='FRAME',
        thread_count=1
    )
    video_wrapper = VideoRecordingWrapper(
        env=RobomimicImageWrapper(
            env=raw_env,
            shape_meta=shape_meta,
            init_state=None,
            render_obs_key=render_obs_key,
        ),
        video_recoder=video_recorder,
        file_path=None,
        steps_per_render=max(20 // fps, 1)
    )

    env_wrapped = MultiStepWrapper(
        env=video_wrapper,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
    )
    return env_wrapped


class SequentialLiberoImageRunner(BaseImageRunner):
    """
    Sequential version of LiberoImageRunner. Each rollout:
      - calls create_env(...) to build MultiStepWrapper(VideoRecording(RobomimicImageWrapper))
      - sets init_state or seed, sets video file path
      - performs rollout exactly like the parallel runner’s inner loop
    """

    def __init__(
        self,
        task_dir,
        output_dir,
        dataset_path,
        shape_meta: dict,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="agentview_image",
        fps=10,
        crf=22,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,

    ):
        super().__init__(output_dir)

        # Create folder for recorded videos
        self.media_dir = pathlib.Path(output_dir).joinpath("media")
        self.media_dir.mkdir(parents=True, exist_ok=True)

        # Use task_dir as dataset path
        self.dataset_path = os.path.expanduser(task_dir)
        self.env_meta = FileUtils.get_env_metadata_from_dataset(self.dataset_path)
        self.shape_meta = shape_meta

        # Fix BDDL filename if needed
        if self.env_meta["bddl_file"] not in bddl_file_name_dict.values():
            self.env_meta["bddl_file"] = bddl_file_name_dict[self.env_meta["bddl_file"]]
            self.env_meta["env_kwargs"]["bddl_file_name"] = self.env_meta["bddl_file"]

        # Rotation transformer if absolute actions
        self.rotation_transformer = None
        if abs_action:
            self.env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            from diffusion_policy.model.common.rotation_transformer import (
                RotationTransformer,
            )

            self.rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        # Build list of rollout configurations
        self.env_configs = []
        with h5py.File(self.dataset_path, "r") as f:
            # Train inits
            if 'data' in f:
                data_group = f['data']
                # total = data_group.attrs.get('total', None)
                self.env_args = data_group.attrs.get('env_args', None)

            for i in range(n_train):
                idx = train_start_idx + i
                init_state = f[f"data/demo_{idx}/states"][0]
                enable_render = (i < n_train_vis)
                self.env_configs.append({
                    "prefix": f"train/{self.env_meta['bddl_file'].split('/')[-1][:-5]}_",
                    "init_state": init_state,
                    "seed": None,
                    "enable_render": enable_render
                })
            # Test inits
            for i in range(n_test):
                seed = test_start_seed + i
                print('seed:', seed)
                enable_render = (i < n_test_vis)
                self.env_configs.append({
                    "prefix": f"test/{self.env_meta['bddl_file'].split('/')[-1][:-5]}_",
                    "init_state": None,
                    "seed": seed,
                    "enable_render": enable_render
                })

        self.fps = fps
        self.crf = crf
        self.render_obs_key = render_obs_key
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.past_action = past_action
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.test_start_seed = test_start_seed
        self.output_path = None

        # Language goal and task name (for logging)
        self.language_goal = " ".join(task_dir.split("/")[-1][:-10].split("_"))
        assert self.language_goal in language_goals_list, f"Language goal {self.language_goal} not found in language_goals"

        self.task_name = self.env_meta["bddl_file"].split("/")[-1][:-5]

    def _initialize_env(self, env, prefix, init_state, seed, enable_render, idx):
        """
        Exactly like the parallel version’s init_fn:
          - Stop previous recording
          - Set file_path if enable_render
          - Set RobomimicImageWrapper.init_state or env.seed(seed)
        """
        assert isinstance(env.env, VideoRecordingWrapper)
        env.env.video_recoder.stop()
        env.env.file_path = None

        if enable_render:
            filename = self.media_dir.joinpath(f"eval_video_{idx}.mp4")
            env.env.file_path = str(filename)

        assert isinstance(env.env.env, RobomimicImageWrapper)
        if prefix.startswith("train"):
            env.env.env.init_state = init_state
        else:
            env.env.env.init_state = None
            env.seed(seed)

    def run(self, policy: BaseImagePolicy, **kwargs):
        device = policy.device

        n_inits = len(self.env_configs)
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for i, cfg in enumerate(self.env_configs):
            prefix = cfg["prefix"]
            init_state = cfg["init_state"]
            seed = cfg["seed"]
            enable_render = cfg["enable_render"]
            # 1) Create + wrap a fresh env (MultiStepWrapper(VideoRecordingWrapper(...)))
            env = create_env(
                env_meta=self.env_meta,
                shape_meta=self.shape_meta,
                enable_render=True,
                render_obs_key=self.render_obs_key,
                fps=self.fps,
                crf=self.crf,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.n_action_steps,
                max_steps=self.max_steps,
            )

            # 2) Initialize per-config (sets init_state or seed, sets video file path)
            self._initialize_env(env, prefix, init_state, seed, enable_render, i)

            # 3) Reset and run policy
            obs = env.reset()
            policy.reset()
            past_action_list = []
            rewards = []

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {self.task_name} {i+1}/{n_inits}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            timestep = 0

            while not done:
                np_obs_dict = dict(obs)
                if self.past_action and len(past_action_list) > 0:
                    np_obs_dict["past_action"] = past_action_list[-1].astype(np.float32)

                # Expand batch dim
                np_obs_dict = dict_apply(np_obs_dict, lambda x: np.expand_dims(x, axis=0))
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device))

                # Rename keys exactly as parallel version did
                if "agentview_image" in obs_dict:
                    obs_dict["agentview_rgb"] = obs_dict.pop("agentview_image")
                if "robot0_eye_in_hand_image" in obs_dict:
                    obs_dict["eye_in_hand_rgb"] = obs_dict.pop("robot0_eye_in_hand_image")
                if "robot0_joint_pos" in obs_dict:
                    obs_dict["joint_states"] = obs_dict.pop("robot0_joint_pos")
                if "robot0_eef_pos" in obs_dict:
                    obs_dict["ee_pos"] = obs_dict.pop("robot0_eef_pos")
                if "robot0_eef_quat" in obs_dict:
                    obs_dict["ee_ori"] = obs_dict.pop("robot0_eef_quat")

                action_dict = policy.predict_action_dyn_guided(
                    obs_dict,
                    language_goal=[self.language_goal] * obs_dict["agentview_rgb"].size(0),
                )

                np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy().squeeze(0))
                action = np_action_dict["action"]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                # Step env
                env_action = self.undo_transform_action(action) if self.abs_action else action
                obs, reward, done_flag, info = env.step(env_action)

                # # Check Libero-specific success signals
                if reward == 1.0:
                    done_flag = np.array([True])
                done = np.all(done_flag)
                
                past_action_list.append(action[np.newaxis, ...])
                if len(past_action_list) > 2:
                    past_action_list.pop(0)

                rewards.append(reward)
                timestep += 1

                # **Update pbar by n_action_steps** (action.shape[0])
                pbar.update(action.shape[0])
            pbar.close()

            # 5) Collect videos & rewards
            video_path = env.render()
            if isinstance(video_path, list) and len(video_path) > 0:
                video_path = video_path[0]
            all_video_paths[i] = video_path
            all_rewards[i] = rewards
            env.close()
            del env

        # 6) Logging
        max_rewards = collections.defaultdict(list)
        log_data = {}

        for i, cfg in enumerate(self.env_configs):
            prefix = cfg["prefix"]
            reward_arr = np.array(all_rewards[i])
            max_reward = reward_arr.max() if reward_arr.size > 0 else 0.0
            max_rewards[prefix].append(max_reward)

            if prefix.startswith("train"):
                key = f"{prefix}sim_max_reward_{i}"
            else:
                key = f"{prefix}sim_max_reward_{cfg['seed']}"
            log_data[key] = max_reward

            vp = all_video_paths[i]
            if vp is not None and os.path.exists(vp):
                if prefix.startswith("train"):
                    log_data[f"{prefix}sim_video_{i}"] = wandb.Video(vp)
                else:
                    log_data[f"{prefix}sim_video_{cfg['seed']}"] = wandb.Video(vp)

        for prefix, vals in max_rewards.items():
            log_data[f"{prefix}mean_score"] = np.mean(vals)

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
