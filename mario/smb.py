from abc import ABC, abstractmethod
import gym_super_mario_bros as gym_smb
from gym_super_mario_bros import actions
from nes_py.wrappers import JoypadSpace
from action import Action
import time
import numpy as np
from dataclasses import dataclass
# gym
import gym
from gym.wrappers import Monitor
from gym.spaces import Box
from gym.wrappers import FrameStack, LazyFrames, RecordVideo
#torch
import torch
import torchvision.transforms as T

from wrappers import wrapper



@dataclass
class Memory:
    """ A dataclass representing a single memory
    """
    state:      LazyFrames
    next_state: LazyFrames
    action:     Action
    reward:     int
    done:       bool



class Environment(ABC):
    """ 
    Abstract class wrapping an environment
    """
    @abstractmethod
    def step(self, action: Action, render: bool = False) -> Memory:
        """ Perform an action and create a memory

        Args:
            action (Action): the action to perform
            render (bool, optional) [False]. Whether to render the environment after

        Returns:
            Memory: a memory of the action performed
        """
        pass

    @abstractmethod
    def reset(self) -> LazyFrames:
        """ Reset the environment

        Returns:
            LazyFrames: the initial state
        """
        pass

    @abstractmethod
    def render(self):
        """ 
        Render the environment
        """
        pass

    @abstractmethod
    def close(self):
        """ 
        Close the environment
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """ Return whether the episode is finished

        Returns:
            bool: true if the episode is over
        """
        pass


class Smb(Environment):
    def __init__(
        self,
        action_set = actions.RIGHT_ONLY,
        env: str = 'SuperMarioBros-v0',
        record: bool = False,
    ):
        smb_env = gym_smb.make(env)
        self.env: JoypadSpace = JoypadSpace(smb_env, action_set)
        self.done: bool = True
        self.state: LazyFrames
        self.last_reward: int
        self.last_info: dict
        self.last_memory: Memory

        # Apply Wrappers to environment
        # self.env = SkipFrame(self.env, skip=4)
        # self.env = GrayScaleObservation(self.env)
        # self.env = ResizeObservation(self.env, shape=84)
        # self.env = FrameStack(self.env, num_stack=4)
        self.env = wrapper(self.env)
        if record:
            self.env = RecordVideo(self.env, 'video' )

    def step(self, action: Action, render: bool = False) -> Memory:
        next_state, reward, done, info = self.env.step(action)
        self.last_memory = Memory(self.state, next_state, action, reward, done)
        self.state = next_state
        self.last_reward = reward
        self.last_info = info
        self.done = done

        if render:
            self.render()
        return self.last_memory

    def reset(self) -> gym.wrappers.LazyFrames:
        self.state = self.env.reset()
        self.done = False
        return self.state

    def render(self, fps: int = 60):
        self.env.render()
        time.sleep(1/fps)

    def close(self):
        self.env.close()

    def is_done(self) -> bool:
        return self.done or self.last_info['flag_get']

