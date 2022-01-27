from abc import ABC, abstractmethod
import gym_super_mario_bros as gym_smb
from gym_super_mario_bros import actions
from nes_py.wrappers import JoypadSpace
from action import Action
import time
import numpy as np

# gym
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack



import torch
import torchvision.transforms as T


class Environment(ABC):
    @abstractmethod
    def step(self, action: Action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass


class Smb(Environment):
    def __init__(
        self,
        action_set: list[list[str]] = actions.COMPLEX_MOVEMENT,
        env: str = 'SuperMarioBros-v0'
    ):
        smb_env = gym_smb.make(env)
        self.env: JoypadSpace = JoypadSpace(smb_env, action_set)
        self.done: bool = True
        self.state: np.ndarray
        self.last_reward: int
        self.last_info: dict

        # Apply Wrappers to environment
        self.env = SkipFrame(self.env, skip=4)
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=84)
        self.env = FrameStack(self.env, num_stack=4)

    def step(self, action: Action):
        next_state, reward, done, info = self.env.step(action)
        self.state = next_state
        self.last_reward = reward
        self.done = done
        self.last_info = info
        return next_state, reward, done, info

    def reset(self) -> np.ndarray:
        self.state = self.env.reset()
        self.done = False
        return self.state

    def render(self, fps: int = 60):
        self.env.render()
        time.sleep(1/fps)

    def close(self):
        self.env.close()

    def is_done(self) -> bool:
        return self.done
















class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation
