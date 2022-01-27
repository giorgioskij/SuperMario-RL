from smb import Smb, Memory
from action import Action
from pprint import pprint
from agent import Agent, Mario
from pathlib import Path
from datetime import datetime
from logger import MetricLogger
import torch
import sys
from gym_super_mario_bros import actions


def test(initial_weights: str, n_episodes: int = 1):
    smb = Smb(action_set=actions.SIMPLE_MOVEMENT, env='SuperMarioBros-v2')
    mario = Mario(state_shape=(4,84,84), n_actions=smb.env.action_space.n, savestates_path=None)
    mario.restore_weights(initial_weights)

    for e in range(n_episodes):
        state = smb.reset()

        # play a level
        while True:
            # choose an action
            action = mario.choose_action(state)
            # execute the chosen action and gather the memory
            memory: Memory = smb.step(action, render=True)
            # log reward, q, loss
            if smb.is_done():
                break
        
    smb.close()
    


def main():

    smb = Smb(action_set=actions.SIMPLE_MOVEMENT, env='SuperMarioBros-v2')
    save_dir = Path('checkpoints_nobk') / datetime.now().strftime("%Y-%m-%dT%H-%M")
    save_dir.mkdir(parents=True)
    mario = Mario(state_shape=(4,84,84), n_actions=smb.env.action_space.n, savestates_path=save_dir)
    logger = MetricLogger(save_dir)

    # parameters
    episodes = 100
    log = False
    initial_weights = None 
    render_every = 20
    log_every = 2


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
                action = mario.choose_action(state)
                # execute the chosen action and gather the memory
                memory: Memory = smb.step(action, render=render)
                # memorize the action
                mario.memorize(memory)

                # execute an update step
                q, loss = mario.learn(verbose=True)
                # log reward, q, loss
                logger.log_step(smb.last_reward, loss, q)
                if smb.is_done():
                    break

            logger.log_episode()
            if e % log_every == 0:
                logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.current_step)

    except KeyboardInterrupt: 
        print('\nKeyboard interrupt detected: saving weights...')
        mario.save()
        
    finally:
        mario.save()
        try:
            smb.close()
        except ValueError:
            pass


if __name__ == '__main__':
    main()
    # test('checkpoints_nobk/2022-01-27T17-00/mario_net_0.chkpt')