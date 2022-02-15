from smb import Smb, Memory
from action import Action
from pprint import pprint
from agent import Agent, Mario
from pathlib import Path
from datetime import datetime
from logger import MetricLogger
import torch
import os 
import sys
from gym_super_mario_bros import actions


def test(initial_weights: str, n_episodes: int = 1, record: bool = False):
    action_set = [
        ['right'],
        ['right', 'A']
    ]
    action_set = actions.RIGHT_ONLY
    smb = Smb(action_set=action_set, env='SuperMarioBros-v0', record = record)
    mario = Mario(state_shape=(84,84,4), n_actions=smb.env.action_space.n, savestates_path=None)
    mario.restore_weights(initial_weights)

    for e in range(n_episodes):
        state = smb.reset()

        # play a level
        while True:
            # choose an action
            action = mario.choose_action(smb.state)
            # execute the chosen action and gather the memory
            memory: Memory = smb.step(action, render=True)
            # log reward, q, loss
            if smb.is_done():
                break

        # print(smb.last_info)
        if smb.last_info['flag_get']:

            print('Level complete')
            print(f'episode : {e}')
        # else:
        #     os.remove(f'video/rl-video-episode-{e}.meta.json')
        #     os.remove(f'video/rl-video-episode-{e}.mp4')
            

        
    smb.close()
    


def main():

    # parameters
    action_set = actions.RIGHT_ONLY
    episodes = 40000
    log = False
    initial_weights = None 
    render_every = None # set to none to disable rendering during training 
    log_every = 100

    smb = Smb(action_set=action_set, env='SuperMarioBros-v0')
    save_dir = Path('checkpoints_nobk') / datetime.now().strftime("%Y-%m-%dT%H-%M")
    save_dir.mkdir(parents=True)
    mario = Mario(state_shape=(84,84,4), n_actions=smb.env.action_space.n, savestates_path=save_dir)
    logger = MetricLogger(save_dir)


    if initial_weights:
        mario.restore_weights(initial_weights)

    try:
        # Training loop
        for e in range(episodes):

            # set up level
            state = smb.reset()
            if render_every and (e % render_every) == 0:
                render = True
                # smb.render()
            else:
                render = False

            # play a level
            while True:
                # choose an action
                action = mario.choose_action(smb.state)
                # execute the chosen action and gather the memory
                memory: Memory = smb.step(action, render=render)
                # memorize the action
                mario.memorize(memory)

                # execute an update step
                q, loss = mario.learn(verbose=False)
                # log reward, q, loss
                logger.log_step(smb.last_reward, loss, q)
                if smb.is_done():
                    break

            logger.log_episode()
            if e % log_every == 0:
                logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.current_step)

    except KeyboardInterrupt: 
        print('\nKeyboard interrupt detected: saving weights...')
        
    finally:
        mario.save()
        try:
            smb.close()
        except ValueError:
            pass


if __name__ == '__main__':
    # main() 
    test('checkpoints_nobk/2022-01-28T15-12/mario_net_1.chkpt', record = True, n_episodes=1)


'''
- To run with visuals or to test, we need to disable matplotlib logging. 
This is done by commenting the last lines in logger.py

- To run with matplotlib, we need to disable visuals.
This is done by setting render_every to None in main.py.

'''