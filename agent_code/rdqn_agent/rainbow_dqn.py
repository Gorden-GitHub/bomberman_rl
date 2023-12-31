import torch
import torch.nn as nn
import numpy as np
import copy
from collections import deque
import torch.nn.functional as F
import math

class DQN(object):
    def __init__(self, state_dim, hidden_dim = 64, action_dim = 6, batch_size = 1e5, max_train_steps = 2e5, lr = 1e-4,
                gamma = 0.99,  tau = 0.0005,grad_clip = 10.0, target_update_freq = 200, n_steps = 5, use_noisy = True,
                device = 'cpu'):
        self.action_dim = action_dim
        self.batch_size = batch_size  # batch size
        self.max_train_steps = max_train_steps
        self.lr = lr  # learning rate
        self.gamma = gamma  # discount factor
        self.tau = tau  # Soft update
        self.use_soft_update = True
        self.target_update_freq = target_update_freq  # hard update
        self.update_count = 0
        self.grad_clip = grad_clip
        self.use_lr_decay = False
        self.use_double = True
        self.use_dueling = True
        self.use_per = True
        self.use_n_steps = True
        self.device = device
        
        if self.use_n_steps:
            self.gamma = self.gamma ** n_steps

        if self.use_dueling:  # Whether to use the 'dueling network'
            self.net = Dueling_Net(state_dim, hidden_dim, action_dim, use_noisy).to(self.device)
        else:
            self.net = Net(state_dim, hidden_dim, action_dim, use_noisy).to(self.device)

        self.target_net = copy.deepcopy(self.net)  # Copy the online_net to the target_net

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, state, epsilon):
        self.net.to(self.device)
        with torch.no_grad():
            state = state
            state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(self.device)
            q = self.net(state)
            if np.random.uniform() > epsilon:
                action = q.argmax(dim=-1).item()
            else:
                action = np.random.randint(0, self.action_dim)
            return action, q.reshape(-1).cpu().numpy()

    def update(self, replay_buffer, total_steps):
        self.net.to(self.device)
        self.target_net.to(self.device)
        batch, batch_index, IS_weight = replay_buffer.sample(total_steps)
        IS_weight = IS_weight.to(self.device)
        with torch.no_grad():  # q_target has no gradient
            if self.use_double:  # Whether to use the 'double q-learning'
                # Use online_net to select the action
                a_argmax = self.net(batch['next_state'].to(self.device)).argmax(dim=-1, keepdim=True)  # shape：(batch_size,1)
                # Use target_net to estimate the q_target
                q_target = batch['reward'].to(self.device) + self.gamma * (1 - batch['terminal'].to(self.device)) * self.target_net(batch['next_state'].to(self.device)).gather(-1, a_argmax).squeeze(-1)  # shape：(batch_size,)
            else:
                q_target = batch['reward'].to(self.device) + self.gamma * (1 - batch['terminal'].to(self.device)) * self.target_net(batch['next_state'].to(self.device)).max(dim=-1)[0]  # shape：(batch_size,)

        q_current = self.net(batch['state'].to(self.device)).gather(-1, batch['action'].to(self.device)).squeeze(-1)  # shape：(batch_size,)
        td_errors = q_current - q_target  # shape：(batch_size,)

        if self.use_per:
            loss = (IS_weight * (td_errors ** 2)).mean()
            replay_buffer.update_batch_priorities(batch_index, td_errors.cpu().detach().numpy())
        else:
            loss = (td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_soft_update:  # soft update
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:  # hard update
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        if self.use_lr_decay:  # learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now




class Dueling_Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, use_noisy):
        super(Dueling_Net, self).__init__()
        self.CNN = CNN(3, 1, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        if use_noisy:
            self.V = NoisyLinear(64, 1)
            self.A = NoisyLinear(64, action_dim)
        else:
            self.V = nn.Linear(64, 1)
            self.A = nn.Linear(64, action_dim)

    def forward(self, s):
        # s = self.CNN(s)
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        V = self.V(s)  # batch_size X 1
        A = self.A(s)  # batch_size X action_dim
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, out_dim):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.out_layers = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        
        self.out = nn.Linear(32 * 4 * 4, out_dim)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        h = h.view(h.shape[0],-1)
        out = self.out(h)
        return out

class Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, use_noisy, action_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if use_noisy:
            self.fc3 = NoisyLinear(hidden_dim, action_dim)
        else:
            self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        Q = self.fc3(s)
        return Q


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  # mul是对应元素相乘
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))  # 这里要除以out_features

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size)  # torch.randn产生标准高斯分布
        x = x.sign().mul(x.abs().sqrt())
        return x

class N_Steps_Prioritized_ReplayBuffer(object):
    def __init__(self, state_dim, max_train_steps = 2e5, alpha = 0.6, beta_init = 0.4, gamma  = 0.99, batch_size = 1e5, buffer_capacity = 1e5,
                 n_steps = 4 ):
        self.max_train_steps = max_train_steps
        self.alpha = alpha
        self.beta_init = beta_init
        self.beta = beta_init
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.sum_tree = SumTree(self.buffer_capacity)
        self.n_steps = n_steps
        self.n_steps_deque = deque(maxlen=self.n_steps)
        self.buffer = {'state': np.zeros((self.buffer_capacity, 3, 17, 17)),
                       'action': np.zeros((self.buffer_capacity, 1)),
                       'reward': np.zeros(self.buffer_capacity),
                       'next_state': np.zeros((self.buffer_capacity, 3, 17, 17)),
                       'terminal': np.zeros(self.buffer_capacity),
                       }
        self.current_size = 0
        self.count = 0

    def store_transition(self, state, action, reward, next_state, terminal, done):
        transition = (state, action, reward, next_state, terminal, done)
        self.n_steps_deque.append(transition)
        if len(self.n_steps_deque) == self.n_steps:
            state, action, n_steps_reward, next_state, terminal = self.get_n_steps_transition()
            self.buffer['state'][self.count] = state
            self.buffer['action'][self.count] = action
            self.buffer['reward'][self.count] = n_steps_reward
            self.buffer['next_state'][self.count] = next_state
            self.buffer['terminal'][self.count] = terminal
            # 如果是buffer中的第一条经验，那么指定priority为1.0；否则对于新存入的经验，指定为当前最大的priority
            priority = 1.0 if self.current_size == 0 else self.sum_tree.priority_max
            self.sum_tree.update(data_index=self.count, priority=priority)  # 更新当前经验在sum_tree中的优先级
            self.count = (self.count + 1) % self.buffer_capacity  # When 'count' reaches buffer_capacity, it will be reset to 0.
            self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self, total_steps):
        batch_index, IS_weight = self.sum_tree.get_batch_index(current_size=self.current_size, batch_size=self.batch_size, beta=self.beta)
        self.beta = self.beta_init + (1 - self.beta_init) * (total_steps / self.max_train_steps)  # beta：beta_init->1.0
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key == 'action':
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.float32)

        return batch, batch_index, IS_weight

    def get_n_steps_transition(self):
        state, action = self.n_steps_deque[0][:2]  # 获取deque中第一个transition的s和a
        next_state, terminal = self.n_steps_deque[-1][3:5]  # 获取deque中最后一个transition的s'和terminal
        n_steps_reward = 0
        for i in reversed(range(self.n_steps)):  # 逆序计算n_steps_reward
            r, s_, ter, d = self.n_steps_deque[i][2:]
            n_steps_reward = r + self.gamma * (1 - d) * n_steps_reward
            if d:  # 如果done=True，说明一个回合结束，保存deque中当前这个transition的s'和terminal作为这个n_steps_transition的next_state和terminal
                next_state, terminal = s_, ter

        return state, action, n_steps_reward, next_state, terminal

    def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update(data_index=index, priority=priority)


class SumTree(object):
    """
    Story data with its priority in the tree.
    Tree structure and array storage:

    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions

    Array type for storing:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity  # buffer的容量
        self.tree_capacity = 2 * buffer_capacity - 1  # sum_tree的容量
        self.tree = np.zeros(self.tree_capacity)

    def update(self, data_index, priority):
        # data_index表示当前数据在buffer中的index
        # tree_index表示当前数据在sum_tree中的index
        tree_index = data_index + self.buffer_capacity - 1  # 把当前数据在buffer中的index转换为在sum_tree中的index
        change = priority - self.tree[tree_index]  # 当前数据的priority的改变量
        self.tree[tree_index] = priority  # 更新树的最后一层叶子节点的优先级
        # then propagate the change through the tree
        while tree_index != 0:  # 更新上层节点的优先级，一直传播到最顶端
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_index(self, v):
        parent_idx = 0  # 从树的顶端开始
        while True:
            child_left_idx = 2 * parent_idx + 1  # 父节点下方的左右两个子节点的index
            child_right_idx = child_left_idx + 1
            if child_left_idx >= self.tree_capacity:  # reach bottom, end search
                tree_index = parent_idx  # tree_index表示采样到的数据在sum_tree中的index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[child_left_idx]:
                    parent_idx = child_left_idx
                else:
                    v -= self.tree[child_left_idx]
                    parent_idx = child_right_idx

        data_index = tree_index - self.buffer_capacity + 1  # tree_index->data_index
        return data_index, self.tree[tree_index]  # 返回采样到的data在buffer中的index,以及相对应的priority

    def get_batch_index(self, current_size, batch_size, beta):
        batch_index = np.zeros(batch_size, dtype=np.int32)
        IS_weight = torch.zeros(batch_size, dtype=torch.float32)
        segment = self.priority_sum / batch_size  # 把[0,priority_sum]等分成batch_size个区间，在每个区间均匀采样一个数
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            index, priority = self.get_index(v)
            batch_index[i] = index
            prob = priority / self.priority_sum  # 当前数据被采样的概率
            IS_weight[i] = (current_size * prob) ** (-beta)
        IS_weight /= IS_weight.max()  # normalization

        return batch_index, IS_weight

    @property
    def priority_sum(self):
        return self.tree[0]  # 树的顶端保存了所有priority之和

    @property
    def priority_max(self):
        return self.tree[self.buffer_capacity - 1:].max()  # 树的最后一层叶节点，保存的才是每个数据对应的priority
