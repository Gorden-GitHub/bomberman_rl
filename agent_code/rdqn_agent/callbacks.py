import os
import pickle
import random
import numpy as np
# from .model import ActorCritic
from .rainbow_dqn import *
import torch
import argparse
from ..rule_based_agent.callbacks import act as chosen_by_rule
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # if self.train or not os.path.isfile("my-saved-model.pt"):
    self.action_by_rule = False
    self.random_use_rule = True
    self.state_dim = 81
    self.batch_size = int(1e5)
    self.buffer_capacity = int(8192 * 4)
    self.use_noisy = True
    if self.use_noisy:  # 如果使用Noisy net，就不需要epsilon贪心策略了
        self.epsilon = 0
    else:
        self.epsilon = 0.5
        self.epsilon_min = 0.1
        self.epsilon_decay = (0.5 - 0.1) / self.batch_size

    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = DQN(state_dim = self.state_dim,  # 状态数
                    hidden_dim = 64,  # 隐含层数
                    action_dim = 6,  # 动作数
                    use_noisy = self.use_noisy,
                    batch_size = self.batch_size,
                    device = device)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    self.danger_zone = []


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .0
    rule_prob = .4
    a_index = self.model.choose_action(state_to_features(self,game_state), epsilon = self.epsilon)
    # purely use rule 
    if self.train and self.action_by_rule:
        rule_based_action = chosen_by_rule(self, game_state)
        if rule_based_action == 'None' or rule_based_action == None:
                # again = chosen_by_rule(self, game_state)
                # print("rule can do nothing")
                rule_based_action = 'WAIT'
        rule_based_index =ACTIONS.index(rule_based_action)
        self.logger.debug(f'Training, chosing action from rule set. Action{rule_based_index}: {rule_based_action}')
        return rule_based_action
    # randomly use rule
    if self.train and self.random_use_rule:
        if random.random() < rule_prob:
            rule_based_action = chosen_by_rule(self, game_state)
            if rule_based_action == 'None' or rule_based_action == None:
                # again = chosen_by_rule(self, game_state)
                # print("rule can do nothing")
                rule_based_action = 'WAIT'
            rule_based_index =ACTIONS.index(rule_based_action)
            self.logger.debug(f'Training, chosing action from rule set. Action{rule_based_index}: {rule_based_action}')
            return rule_based_action
    if self.train:
        # 80%: walk in any direction. 10% wait. 10% bomb.
        # randomly choose at random
        if random.random() < random_prob:
            random_choice = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
            self.logger.debug(f'Training, choosing action purely at random with random prob{random_prob}.')
            return random_choice
        else:
        # accroding to model with prob
            self.logger.debug(f'Training, querying model for action, action: {ACTIONS[a_index]}')
        return ACTIONS[a_index]

    chosen_act = ACTIONS[a_index]

    self.logger.debug(f'Evaluating, querying model for action , action:{chosen_act}')
    
    # 防止扔炸弹，此时已经学会了如何收集金币，权重保留在coin-collector里
    # coin heaven only
    # while chosen_act == 'BOMB':
    #     chosen_act = np.random.choice(ACTIONS, p=probs)
    return chosen_act


def generate_field_from_state(self,game_state):
        # Get important informations:
    field = game_state['field']

    field = np.array(field,dtype=np.float64)
    bombs_data = game_state['bombs']
    # 找到每一个爆弹
    for ((x,y),z) in bombs_data:
        field[x][y] = -1
        danger_zone = find_danger_zone(self,field,x,y,count=z)
        # 爆弹的每一个危险区域，直到碰到-1或者达到3距离为止
        for ((a,b),c) in danger_zone:
            field[a][b] += -0.1 * c - 0.1


    explosion_map = game_state['explosion_map']
    explosion_map = explosion_map * -0.1
    field += explosion_map
    
    coin_pos = game_state['coins']
    for (x,y) in coin_pos:
        field[x][y] += 2
    
    opponents_info = game_state['others']
    opponents_pos = []
    opponents_bombs = []
    for i in opponents_info:
        opponents_pos.append(i[3])
        opponents_bombs.append(i[2])
    my_info = game_state['self']
    my_pos = my_info[3]
    my_bombs = my_info[2]
    for (index,(x,y)) in enumerate(opponents_pos):
        if opponents_bombs[index] ==True:
            field[x][y] += -4.5
        else:
            field[x][y] += -2.5

    if my_bombs == True:
        field[my_pos[0]][my_pos[1]] += -5
    else:
        field[my_pos[0]][my_pos[1]] += -3
    return field

def state_to_features(self, game_state: dict) -> np.array:
    """
    *
    取四格视野范围内的东西

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    field = generate_field_from_state(self,game_state)
    # channels.append(agent_map)

    my_info = game_state['self']
    my_pos = my_info[3]
    
    padded_field = np.pad(field,((5,5),(5,5)),mode='constant',constant_values=-1)
    feature_vector = []
    padded_x = my_pos[0] + 5
    padded_y = my_pos[1] + 5
    for j in range(padded_field.shape[0]):
        for k in range(padded_field.shape[1]):
            if abs(j - padded_x) <= 4 and abs(k - padded_y) <= 4:
                feature_vector.append(padded_field[j][k])
    # # calculate dangerous zone
    # for bomb in self.bombs:
    #     if bomb.timer <= 0:
    #         # Explode when timer is finished
    #         self.logger.info(f'Agent <{bomb.owner.name}>\'s bomb at {(bomb.x, bomb.y)} explodes')
    #         bomb.owner.add_event(e.BOMB_EXPLODED)
    #         blast_coords = bomb.get_blast_coords(self.arena)

    #         # Clear crates
    #         for (x, y) in blast_coords:

    # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels).reshape(-1)
    # and return them as a vector    x  

    # field = field.reshape(-1)
    
    feature_vector = np.array(feature_vector).reshape(-1)
    
    # state normalization
    # state_norm = Normalization(shape=stacked_channels.shape[0])
    if(feature_vector.shape[0] != 81):

        print("feature_vector shape", feature_vector.shape)
    # normed_states = state_norm(stacked_channels)
    self.field = field
    return feature_vector

def find_danger_zone(self, field, x, y, count):
    '''
        输入炸弹和爆炸倒数，返回危险区域的坐标以及爆炸倒数，return ((x,y),count)
    '''
    danger_zone = []
    for i in range(3):
        i += 1
        if y + i >= field.shape[1]:
                break
        if field[x][y + i] == -1 or field[x][y + i] == 1:
            break
        else:
            danger_zone.append(((x,y+i),count + 1))
    for i in range(3):
        i += 1
        if y - i < 0:
            break
        if field[x][y - i] == -1 or field[x][y - i] == 1:
            break
        else:
            danger_zone.append(((x,y-i),count + 1))
    for i in range(3):
        i += 1
        if x + i >= field.shape[0]:
            break
        if field[x + i][y] == -1 or field[x + i][y] == 1:
            break
        else:
            danger_zone.append(((x+i,y),count + 1))
    for i in range(3):
        i += 1
        if x - i < 0:
            break
        if field[x - i][y] == -1 or field[x - i][y]  == 1:
            break
        else:
            danger_zone.append(((x-i,y),count + 1))
    return danger_zone
# def state_to_features(self, game_state: dict):
#     """
#     *This is not a required function, but an idea to structure your code.*

#     Converts the game state to the input of your model, i.e.
#     a feature vector.

#     You can find out about the state of the game environment via game_state,
#     which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
#     what it contains.

#     :param game_state:  A dictionary describing the current game board.
#     :return: np.array
#     """
#     # This is the dict before the game begins and after it ends
#     if game_state is None:
#         return None

#     features = []

#     _, score, bombs_left, (x, y) = game_state['self']
#     arena = game_state['field']
#     coins = game_state['coins']
#     bombs = game_state['bombs']
#     others = [xy for (n, s, b, xy) in game_state['others']]
#     directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
#     if bombs:
#         for (bx, by), t in bombs:
#             arena[bx, by] = -1
#     if others:
#         for xy in others:
#             arena[xy[0], xy[1]] = -1

#     # BFS Coin Feature
#     if len(coins) == 0:
#         features.append("none")
#     else:
#         cfield = arena.copy()
#         for cx, cy in coins:
#             cfield[cx, cy] = 2
#         features.append(BFS((x, y), cfield, 2))

#     # BFS Crate Feature
#     features.append("none")
#     for d in directions[-4:]:
#         if arena[d] == 1:
#             features[1] = 'met'
#             break
#     if features[1] != 'met':
#         features[1] = BFS((x, y), arena, 1)
 
#     # whether Bomb is Possible and to bomb something Feature
#     features.append('false')
#     for d in directions[-4:]:
#         if (game_state["self"][2]) and (arena[d] == 1 or d in others):
#             features[2] = 'true'
#             break
        
    
#     # BFS Safe Zone Feature
#     features.append('none')
#     bfield = arena.copy()
#     for (xb, yb), t in bombs:
#         bfield[xb, yb] = -2
#         for (i, j) in [(xb - h, yb) for h in range(1, 4)]:
#             if (0 < i < bfield.shape[0]) and (0 < j < bfield.shape[1]):
#                 if bfield[(i, j)] == -1:
#                     break
#                 else:
#                     bfield[i, j] = -2
#         for (i, j) in [(xb + h, yb) for h in range(1, 4)]:
#             if (0 < i < bfield.shape[0]) and (0 < j < bfield.shape[1]):
#                 if bfield[(i, j)] == -1:
#                     break
#                 else:
#                     bfield[i, j] = -2
        
#         for (i, j) in [(xb, yb - h) for h in range(1, 4)]:
#             if (0 < i < bfield.shape[0]) and (0 < j < bfield.shape[1]):
#                 if bfield[(i, j)] == -1:
#                     break
#                 else:
#                     bfield[i, j] = -2
#         for (i, j) in [(xb, yb + h) for h in range(1, 4)]:
#             if (0 < i < bfield.shape[0]) and (0 < j < bfield.shape[1]):
#                 if bfield[(i, j)] == -1:
#                     break
#                 else:
#                     bfield[i, j] = -2
#     for d in directions[-4:]:
#         if bfield[d] == -2:
#             features[3] = 'stop'
#             break
#     if bfield[x, y] == -2:
#         features[3] = BFS((x, y), bfield, 0, -2, arena)
#     for d in directions[-4:]:
#         if game_state['explosion_map'][d] >= 1:
#             features[3] = 'stop'
#             break

    
#     try:
#         self.model[", ".join(features)]
#     except Exception as ex:
#         self.model[", ".join(features)] = list(np.zeros(6))
#     return features


def BFS(self_pos, field, goal, ignore = 100, hfield = np.array(0)):

    around = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    queue = [self_pos]
    parents = np.ones((*field.shape, 2), dtype=int) * -1
    current_pos = queue.pop(0)

    while field[current_pos[0], current_pos[1]] != goal:
        for i, j in around:
            neighbor = current_pos + np.array([i, j])
            n_x, n_y = neighbor
            if hfield.any() != 0:
                if not -1 < hfield[n_x, n_y] <= 0:
                    continue
            if not ( -1 < field[n_x, n_y] <= 0
                    or field[n_x, n_y] == goal
                    or field[n_x, n_y] == ignore):
                continue
            if parents[n_x, n_y][0] == -1:
                parents[n_x, n_y] = current_pos
            else:
                continue
            queue.append(neighbor)
        if len(queue) == 0:
            break
        else:
            current_pos = queue.pop(0)
    
    if field[current_pos[0], current_pos[1]] != goal:
        return 'none'
    
    if np.all(current_pos == self_pos):
        return 'none'

    while np.any(parents[current_pos[0], current_pos[1]] != self_pos):
        current_pos = parents[current_pos[0], current_pos[1]]
    
    diff = current_pos - self_pos

    # X coordinate to the left
    if diff[0] < 0:
        return 'LEFT'

    # X coordinate to the right
    if diff[0] > 0:
        return 'RIGHT'

    # Y coordinate Up
    if diff[1] < 0:
        return 'UP'

    # Y coordinate Down
    if diff[1] > 0:
        return 'DOWN'