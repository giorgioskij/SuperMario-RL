from smb import Smb
from action import Action
from pprint import pprint
from agent import Agent, Mario
from pathlib import Path
from datetime import datetime
from logger import MetricLogger
from gym_super_mario_bros import actions

import torch


def main():

    save_dir = Path('checkpoints') / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    logger = MetricLogger(save_dir)
    smb = Smb(action_set=actions.RIGHT_ONLY, env='SuperMarioBros-v2')
    mario = Mario(state_shape=(4,84,84), n_actions=smb.env.action_space.n, savestates_path=save_dir)

    episodes = 1000


    for e in range(episodes):
        state = smb.reset()

        while True:

            action = mario.choose_action(state)

            next_state, reward, done, info = smb.step(action)

            mario.memorize(state, next_state, action, reward, done)

            q, loss = mario.learn()

            logger.log_step(reward, loss, q)

            state = next_state

            if done or info['flag_get']:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.current_step)

    torch.save(mario.dqn.online.state_dict(), save_dir / 'online.pth')
    torch.save(mario.dqn.target.state_dict(), save_dir / 'target.pth')
    # for step in range(1):
    #     if env.is_done(): 
    #         env.reset()
    #     env.step(Action.RIGHT_B)
    #     print(env.state.shape)
    #     print(type(env.state))

    #     pprint(env.state[:, :, 0])
    #     # env.render()
    # env.close()


if __name__ == '__main__':
    main()