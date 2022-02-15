from collections import deque
from action import Action
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random
import numpy as np
from abc import ABC, abstractmethod
import copy
from smb import Memory

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Set pytorch device to {DEVICE}')

class Agent(ABC):

    @abstractmethod
    def choose_action(self, state: np.ndarray) -> Action: 
        """ Implements epsilon-greedy algorithm to balance exploration and exploitation

        Args:
            state (np.ndarray): current state of the game

        Returns:
            Action: chosen action
        """
        pass


    @abstractmethod
    def memorize(self, memory: Memory):
        """ Stores the memory of an action taken and its result 

        Args:
            memory (Memory): the memory to store
        """
        pass

    @abstractmethod
    def recall(self):
        """ Recalls a random memory from its storage 

        Returns:
            tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]: the memory
        """
        pass


class Mario(Agent):

    def __init__(self, state_shape: tuple, n_actions: int, savestates_path: str):
        
        # Game related
        self.state_shape: tuple = state_shape
        self.n_actions: int          = n_actions
        self.savestates_path: str    = savestates_path
        self.save_every: int         = 500000
        # self.save_every: int         = 10000

        # Policy related
        self.device: str                 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.dqn: nn.Module              = MarioNet(self.state_shape, self.n_actions).float()
        self.dqn.to(self.device)

        self.optimizer                   = optim.Adam(self.dqn.parameters(), lr=0.00025)
        # self.loss_fn                     = nn.SmoothL1Loss()
        self.loss_fn                     = nn.HuberLoss()
        self.exploration_rate: int       = 1
        self.exploration_decay: float    = 0.99999975
        self.exploration_rate_min: float = 0.1
        self.current_step: int           = 0
        self.burnin: int                 = 100  # min. experiences before training
        self.learn_every: int            = 4    # no. of experiences between updates to Q_online
        self.sync_every: int             = 10000  # no. of experiences between Q_target & Q_online sync

        # Memory related
        self.batch_size = 32
        self.memories   = deque(maxlen=50000)
        self.gamma      = 0.9


    def choose_action(self, state) -> Action:
        # exploration
        if np.random.rand() < self.exploration_rate:
            action = Action(np.random.randint(self.n_actions))
        # exploitation
        else:
            state = state.__array__()
            state = torch.tensor(state).to(self.device).unsqueeze(0)
            action_values = self.dqn(state, model='online')
            action = Action(torch.argmax(action_values, axis=1).item())
        # decrease exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        # increment step
        self.current_step += 1
        return action


    # stores a memory
    def memorize(self, memory: Memory):

        state = memory.state.__array__()
        next_state = memory.next_state.__array__()
        state      = torch.tensor(state).to(self.device)
        next_state = torch.tensor(next_state).to(self.device)
        action     = torch.tensor([memory.action]).to(self.device)
        reward     = torch.tensor([memory.reward]).to(self.device)
        done       = torch.tensor([memory.done]).to(self.device)
        self.memories.append((state, next_state, action, reward, done,))



    def recall(self):
        batch = random.sample(self.memories, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()



    def td_estimate(self, state: torch.Tensor, action: Action):
        current_Q = self.dqn(state, model='online')[
            np.arange(0, self.batch_size), action
        ]
        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward: int, next_state: torch.Tensor, done: bool):
        next_state_Q = self.dqn(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.dqn(next_state, model='target')[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float() 


    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    
    def sync_Q_target(self):
        self.dqn.target.load_state_dict(self.dqn.online.state_dict())


    def save(self):
        save_path = (
            self.savestates_path / f'mario_net_{int(self.current_step // self.save_every)}.chkpt'
        )
        torch.save(
            dict(model=self.dqn.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f'MarioNet saved to {save_path} at step {self.current_step}')


    def restore_weights(self, path):
        data = torch.load(path)
        self.dqn.load_state_dict(data['model'])
        self.exploration_rate = data['exploration_rate']

    

    def learn(self, verbose: bool = False):

        if self.current_step % self.sync_every == 0:
            if verbose:
                print(f'Step {self.current_step}: copying online weights to target')
            self.sync_Q_target()
        
        if self.current_step % self.save_every == 0:
            if verbose:
                print(f'Step {self.current_step}: saving network')
            self.save()

        if self.current_step < self.burnin: 
            return None, None
        
        if self.current_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)




class MarioNet(nn.Module):
    def __init__(self, input_shape: tuple, n_actions: int):
        super().__init__()
        h, w, c = input_shape

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False
        
        self.init_weights()
        
    def forward(self, x: torch.Tensor, model: str):
        x = x.permute(0,3,1,2)
        x = x.float()/255.
        if model == 'online':
            return self.online(x)
        elif model == 'target':
            return self.target(x)

    def init_weights(self):
        def _init_layer_weight(layer):
            if type(layer) == nn.Linear:
                nn.init.kaiming_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
        self.apply(_init_layer_weight)


class AnotherAgent():
    def __init__(self, h: int, w: int, n_actions: int, epsilon: float, lr: float = 0.01):

        self.n_actions = n_actions
        self.dqn = DQN(h, w, n_actions)
        self.loss_fn = nn.HuberLoss()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr)


    def update(self, state: np.ndarray, y: np.ndarray):
        y_pred = self.dqn(torch.Tensor(state))
        loss = self.loss_fn(y_pred, torch.autograd.Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, state: np.ndarray):
        with torch.no_grad():
            return self.dqn(torch.Tensor(state))

    
    def choose_action(self, state: np.ndarray):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        else:
            q_values = self.predict(state)
            return torch.argmax(q_values).item()




class DQN(nn.Module):

    def __init__(self, h, w, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, n_actions)


    def forward(self, x):
        x = x.to(DEVICE)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class VGGDQN(nn.Module):
    '''
        A VGG-11 convolutional network with custom initialization
    '''

    def __init__(self, n_actions: int, lr: float = 0.01, n_channels: int = 3):

        super().__init__()
    


        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_actions, bias=True)
        )

        self.init_weights()


    def init_weights(self):
        
        def _init_layer_weight(layer):
            if type(layer) == nn.Linear:
                nn.init.kaiming_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

        self.apply(_init_layer_weight)


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
