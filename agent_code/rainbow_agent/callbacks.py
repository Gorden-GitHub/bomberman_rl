import os
import pickle
import random

import numpy as np
import torch
from .rainbow import DQNAgent

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    self.saved_flag = True

    if self.train or not os.path.isfile("my-saved-model.pt"):
        if self.saved_flag:
            self.logger.info("Loading model from saved state.")
            with open("my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)
            self.num_steps = 40000
        else:
            self.logger.info("Setting up model from scratch.")
            
            # set random seed
            seed = 666
            np.random.seed(seed)
            random.seed(seed)
            seed_torch(seed)

            # parameters
            memory_size = 60000
            batch_size = 400
            target_update = 400

            self.num_steps = 60000

            self.model = DQNAgent(578, 6, memory_size, batch_size, target_update, seed)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    self.logger.debug("Querying model for action.")

    action = self.model.select_action(state_to_features(game_state))

    return ACTIONS[action]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    # For example, you could construct several channels of equal shape, ...
    channels = []

    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    others = [xy for (n, s, b, xy) in game_state['others']]
    if game_state["self"][2]:
        arena[(x, y)] = 2
    else:
        arena[(x, y)] = 1.5
    if others:
        for xy in others:
            arena[xy[0], xy[1]] = -2
    coins = game_state['coins']
    for cx, cy in coins:
        arena[cx, cy] = 5
    bombs = game_state['bombs']
    if bombs:
        for (bx, by), t in bombs:
            arena[bx, by] = -1
    channels.append(arena)

    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb - h, yb) for h in range(1, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                if arena[(i, j)] == -1:
                    break
                else:
                    bomb_map[i, j] = min(bomb_map[i, j], t)
        bomb_map[xb, yb] = min(bomb_map[xb, yb], t)
        for (i, j) in [(xb + h, yb) for h in range(1, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                if arena[(i, j)] == -1:
                    break
                else:
                    bomb_map[i, j] = min(bomb_map[i, j], t)
        for (i, j) in [(xb, yb - h) for h in range(1, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                if arena[(i, j)] == -1:
                    break
                else:
                    bomb_map[i, j] = min(bomb_map[i, j], t)
        for (i, j) in [(xb, yb + h) for h in range(1, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                if arena[(i, j)] == -1:
                    break
                else:
                    bomb_map[i, j] = min(bomb_map[i, j], t)
    explosion_map = game_state['explosion_map']
    bomb_map[np.where(explosion_map == 1)] = 0
    channels.append(bomb_map)
    
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
