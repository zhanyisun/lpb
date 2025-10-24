import json
import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import wandb.sdk.data_types.video as wv

from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

def create_env(env_meta, shape_meta, enable_render=True, render_obs_key='agentview_image',
               fps=10, crf=22, n_obs_steps=2, n_action_steps=8, max_steps=400):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    # disable Robosuite's hard reset to avoid large memory usage
    env.env.hard_reset = False

    # Create a VideoRecordingWrapper with H.264
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
            env=env,
            shape_meta=shape_meta,
            init_state=None,         # will be set later
            render_obs_key=render_obs_key,
        ),
        video_recoder=video_recorder,
        file_path=None,            # will be set later
        steps_per_render=max(20 // fps, 1)  # robosuite default fps is 20
    )

    # Finally, wrap with MultiStepWrapper
    env_wrapped = MultiStepWrapper(
        env=video_wrapper,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
    )
    return env_wrapped


class SequentialRobomimicImageRunner(BaseImageRunner):
    """
    A sequential runner with minimal changes but no function pickling.
    We store *raw* initialization data (like init_state or seed) and apply
    it inline in `run()`, thus avoiding 'ctypes pointer' pickling errors.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
        ):
        super().__init__(output_dir)
        if 'square' in dataset_path:
            max_steps = 600
        # BUG
        # max_steps = 50
        # Prepare folder for recorded videos
        self.media_dir = pathlib.Path(output_dir).joinpath('media')
        self.media_dir.mkdir(parents=True, exist_ok=True)

        # Load environment metadata
        dataset_path = os.path.expanduser(dataset_path)
        self.dataset_path = dataset_path
        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        self.env_configs = []  # each item is dict with keys: mode, init_state_or_seed, enable_render

        # Read from dataset for train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                init_state = f[f'data/demo_{train_idx}/states'][0]
                enable_render = (i < n_train_vis)

                self.env_configs.append({
                    'prefix': 'train/',
                    'init_state': init_state,  # for train
                    'seed': None,
                    'enable_render': enable_render
                })

        # Add test config
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = (i < n_test_vis)

            self.env_configs.append({
                'prefix': 'test/',
                'init_state': None,  # for test, we do random reset
                'seed': seed,
                'enable_render': enable_render
            })

        # Store everything as fields
        self.output_dir = output_dir
        self.dataset_path = dataset_path
        self.env_meta = env_meta
        self.shape_meta = shape_meta
        self.n_train = n_train
        self.n_test = n_test
        self.render_obs_key = render_obs_key
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.past_action = past_action
        self.abs_action = abs_action
        self.rotation_transformer = rotation_transformer
        self.tqdm_interval_sec = tqdm_interval_sec

        self.output_path = None

    def run(self, policy: BaseImagePolicy):
        device = policy.device

        n_inits = len(self.env_configs)  # total number of train + test inits
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for i, env_cfg in enumerate(self.env_configs):
            prefix = env_cfg['prefix']
            init_state = env_cfg['init_state']  # either actual array or None
            seed = env_cfg['seed']             # either int or None
            enable_render = env_cfg['enable_render']
            # Create the environment fresh each time
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
            
            self._initialize_env(env, prefix, init_state, seed, enable_render, i)

            # Now do the rollout
            obs = env.reset()
            policy.reset()
            past_action = None
            rewards = []

            # Build a progress bar for steps
            # Use environment name from self.env_meta
            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}Image {i+1}/{n_inits}",
                leave=False,
                mininterval=self.tqdm_interval_sec
            )
        
            done = False
            timestep = 0

            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                    
                np_obs_dict = dict_apply(np_obs_dict, lambda x: np.expand_dims(x, axis=0))
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                action_dict = policy.predict_action_dyn_guided(obs_dict)
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy().squeeze(0))
                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                success = env.env.get_success_label()

                done = np.all(done)
                past_action = action
                rewards.append(reward)

                timestep += 1
                # update pbar
                pbar.update(action.shape[0])
                if success: break
            pbar.close()

            # store results for logging
            video_path = env.render()
            if isinstance(video_path, list) and len(video_path) > 0:
                video_path = video_path[0]
            all_video_paths[i] = video_path
            all_rewards[i] = rewards
            env.close()
            del env

        # Log results
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i, env_cfg in enumerate(self.env_configs):
            prefix = env_cfg['prefix']
            reward_array = np.array(all_rewards[i])
            max_reward = np.max(reward_array) if len(reward_array) > 0 else 0.0
            max_rewards[prefix].append(max_reward)

            # seed vs train_idx
            if prefix.startswith('train'):
                # you might store train_idx in env_cfg if you want
                # or just index i
                key = prefix + f'sim_max_reward_{i}'
            else:
                # if you store a separate `seed` for test, you can do:
                key = prefix + f'sim_max_reward_{env_cfg["seed"]}'
            log_data[key] = max_reward

            # video
            video_path = all_video_paths[i]
            if video_path is not None and os.path.exists(video_path):
                sim_video = wandb.Video(video_path)
                # similarly for logging key
                if prefix.startswith('train'):
                    log_data[prefix + f'sim_video_{i}'] = sim_video
                else:
                    log_data[prefix + f'sim_video_{env_cfg["seed"]}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            log_data[name] = np.mean(value)

        return log_data

    def _initialize_env(self, env, prefix, init_state, seed, enable_render, i):
        """
        A helper function to set up each environment with the correct init.
          - For 'train', we set RobomimicImageWrapper.init_state = init_state
          - For 'test', we set init_state=None and env.seed(...)
          - We also set the file path for the video if enable_render is True
        """
        # Stop any previous recording just in case
        assert isinstance(env.env, VideoRecordingWrapper)
        env.env.video_recoder.stop()
        env.env.file_path = None

        if enable_render:
            # filename = self.media_dir.joinpath(wv.util.generate_id() + ".mp4")
            filename = self.media_dir.joinpath(f"eval_video_{i}.mp4")
            env.env.file_path = str(filename)

        # If 'train/' prefix => we're using a dataset init_state
        # If 'test/' prefix => we do random reset with a seed
        assert isinstance(env.env.env, RobomimicImageWrapper)
        if prefix.startswith('train'):
            env.env.env.init_state = init_state
        else:
            env.env.env.init_state = None
            env.seed(seed)

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
