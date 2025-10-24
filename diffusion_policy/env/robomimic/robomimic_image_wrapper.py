from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import h5py
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite
from diffusion_policy.common.robomimic_util import get_robomimic_model_file

class RobomimicImageWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        # perturb=False,
        # perturb_prob=0.0,
        # perturb_mag=0.0,
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        
        print('env._env_name:', self.env._env_name)
        
        # Use helper function to get model file
        self.model_file = get_robomimic_model_file(self.env._env_name)
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        
        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def get_flattened_state(self):
        return self.env.env.sim.get_state().flatten()
    
    def get_success_label(self):
        return self.env.env._check_success()

    def get_check_tool_on_frame(self):
        return self.env.env._check_tool_on_frame()
    
    def get_check_frame_assembled(self):
        return self.env.env._check_frame_assembled()
    
    def get_trash_in_trash_bin(self):
        return self.env.env.transport.trash_in_trash_bin
    
    def get_payload_in_target_bin(self):
        return self.env.env.transport.payload_in_target_bin
    
    def set_init_state(self, state):
        self.init_state = state
    
    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            if self.model_file is None:
                raw_obs = self.env.reset_to({'states': self.init_state})
            else:
                raw_obs = self.env.reset_to({'states': self.init_state, 'model': self.model_file})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                if self.model_file is None:
                    raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
                else:
                    raw_obs = self.env.reset_to({'states': self.seed_state_map[seed], 'model': self.model_file})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
                if self.model_file is not None:
                    raw_obs = self.env.reset_to({'states': state, 'model': self.model_file})
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()
            state = self.env.get_state()['states']
            if self.model_file is not None:
                raw_obs = self.env.reset_to({'states': state, 'model': self.model_file})

        # return obs
        obs = self.get_observation(raw_obs)
        self.idx = 0
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self.idx += 1
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img

