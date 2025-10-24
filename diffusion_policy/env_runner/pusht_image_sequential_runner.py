import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import collections
import torch
import numpy as np
import wandb.sdk.data_types.video as wv
import zarr
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class SequentialPushTImageRunner(BaseImageRunner):
    def __init__(self,
                 output_dir,
                 n_train=10,
                 n_train_vis=3,
                 train_start_seed=0,
                 n_test=22,
                 n_test_vis=6,
                 legacy_test=False,
                 test_start_seed=10000,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=140,
                 past_action=False,
                 tqdm_interval_sec=5.0,
                 n_envs=None):
        super().__init__(output_dir)
        self.media_dir = pathlib.Path(output_dir).joinpath('media')
        self.media_dir.mkdir(parents=True, exist_ok=True)
        # max_steps = 100
        def env_fn(seed, enable_render):
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=max(10 // fps, 1)
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        self.envs = []
        self.env_seeds = []
        self.env_prefixs = []
        self.env_init_fn_dills = []
        self.output_dir = output_dir
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        # Setup train environments
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis
            env = env_fn(seed, enable_render)
            self.envs.append(env)
            self.env_seeds.append(seed)
            self.env_prefixs.append('train/')
            self.env_init_fn_dills.append(dill.dumps(lambda e, s=seed: e.seed(s)))

        # Setup test environments
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis
            env = env_fn(seed, enable_render)
            self.envs.append(env)
            self.env_seeds.append(seed)
            self.env_prefixs.append('test/')
            self.env_init_fn_dills.append(dill.dumps(lambda e, s=seed: e.seed(s)))

        self.output_path = None

    def run(self, policy: BaseImagePolicy):
        
        device = policy.device
        dtype = policy.dtype

        all_video_paths = []
        all_rewards = []

        max_rewards = collections.defaultdict(list)
        n_inits = len(self.envs)
        log_data = dict()

        for env, init_fn_dill, seed, prefix in tqdm.tqdm(zip(self.envs, self.env_init_fn_dills, self.env_seeds, self.env_prefixs), 
                                                         total=len(self.envs),
                                                         desc="Running environments sequentially"):
            # Initialize environment
            env.run_dill_function(init_fn_dill)
            obs = env.reset()
            past_action = None
            policy.reset()

            # Dynamically set the video file path
            # video_filename = self.media_dir.joinpath(f'{wv.util.generate_id()}.mp4')
            video_filename = self.media_dir.joinpath(f'{seed}.mp4')
            env.env.file_path = str(video_filename)

            # Initialize variables for recording
            done = False
            rewards = []

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval Environment {prefix} Seed {seed}", leave=False)
            info = None
            while not done:
                # Prepare observations
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[:, -(self.n_obs_steps - 1):].astype(np.float32)

                # Add batch dimension to observations
                np_obs_dict = dict_apply(np_obs_dict, lambda x: np.expand_dims(x, axis=0))

                # Convert observations to tensors
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device, dtype=dtype))

                # Get actions from the policy
                action_dict = policy.predict_action_dyn_guided(obs_dict)
                
                # Convert actions back to numpy and remove batch dimension
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy().squeeze(0))

                action = np_action_dict['action']

                # Step environment
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # Record rewards
                rewards.append(reward)
                if reward > 0.98:
                    break
                # break
                pbar.update(1)
            pbar.close()

            # Ensure video is saved after each episode
            env.env.video_recoder.stop()  # Stop video recorder explicitly to save the file
            video_path = env.env.file_path  # Use the file path that was set for video saving
            all_video_paths.append(video_path)
            all_rewards.append(np.max(rewards))

            # Logging rewards and video paths
            max_reward = np.max(rewards)
            max_rewards[prefix].append(max_reward)
            print(f"Environment {prefix} Seed {seed} Max Reward: {max_reward}")
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward
            if video_path is not None and pathlib.Path(video_path).exists():
                # Video file exists, prepare video logging
                sim_video = wandb.Video(str(video_path), fps=self.fps)
                log_data[prefix + f'sim_video_{seed}'] = sim_video

        # Aggregate metrics across all environments and log them
        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data