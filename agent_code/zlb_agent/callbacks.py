import os
import pickle
import random
import json

import numpy as np


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

    #if self.train or not os.path.isfile("my-saved-model.pt"):
    if self.train or not os.path.isfile("q_table.json"):
        if self.saved_flag:
            self.logger.info("Loading model from saved state.")
            # with open("my-saved-model.pt", "rb") as file:
            #     self.model = pickle.load(file)
            with open("q_table.json", "r") as file:
                self.model = json.load(file)
        else:
            self.logger.info("Setting up model from scratch.")
            self.model = dict()
    else:
        self.logger.info("Loading model from saved state.")
        # with open("my-saved-model.pt", "rb") as file:
        #     self.model = pickle.load(file)
        with open("q_table.json", "r") as file:
                self.model = json.load(file)


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

    features = state_to_features(self, game_state)
    self.logger.debug("Querying model for action.")

    action_index = np.argmax(self.model[", ".join(features)])
    self.logger.debug(", ".join(features))

    return ACTIONS[action_index]


def state_to_features(self, game_state: dict):
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

    features = []

    _, score, bombs_left, (x, y) = game_state['self']
    arena = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    others = [xy for (n, s, b, xy) in game_state['others']]
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    if bombs:
        for (bx, by), t in bombs:
            arena[bx, by] = -1
    if others:
        for xy in others:
            arena[xy[0], xy[1]] = -1

    # BFS Coin Feature
    if len(coins) == 0:
        features.append("none")
    else:
        cfield = arena.copy()
        for cx, cy in coins:
            cfield[cx, cy] = 2
        features.append(BFS((x, y), cfield, 2))

    # BFS Crate Feature
    features.append("none")
    for d in directions[-4:]:
        if arena[d] == 1:
            features[1] = 'met'
            break
    if features[1] != 'met':
        features[1] = BFS((x, y), arena, 1)
 
    # whether Bomb is Possible and to bomb something Feature
    features.append('false')
    for d in directions[-4:]:
        if (game_state["self"][2]) and (arena[d] == 1 or d in others):
            features[2] = 'true'
            break
        
    
    # BFS Safe Zone Feature
    features.append('none')
    bfield = arena.copy()
    for (xb, yb), t in bombs:
        bfield[xb, yb] = -2
        for (i, j) in [(xb - h, yb) for h in range(1, 4)]:
            if (0 < i < bfield.shape[0]) and (0 < j < bfield.shape[1]):
                if bfield[(i, j)] == -1:
                    break
                else:
                    bfield[i, j] = -2
        for (i, j) in [(xb + h, yb) for h in range(1, 4)]:
            if (0 < i < bfield.shape[0]) and (0 < j < bfield.shape[1]):
                if bfield[(i, j)] == -1:
                    break
                else:
                    bfield[i, j] = -2
        
        for (i, j) in [(xb, yb - h) for h in range(1, 4)]:
            if (0 < i < bfield.shape[0]) and (0 < j < bfield.shape[1]):
                if bfield[(i, j)] == -1:
                    break
                else:
                    bfield[i, j] = -2
        for (i, j) in [(xb, yb + h) for h in range(1, 4)]:
            if (0 < i < bfield.shape[0]) and (0 < j < bfield.shape[1]):
                if bfield[(i, j)] == -1:
                    break
                else:
                    bfield[i, j] = -2
    for d in directions[-4:]:
        if bfield[d] == -2:
            features[3] = 'stop'
            break
    if bfield[x, y] == -2:
        features[3] = BFS((x, y), bfield, 0, -2, arena)
    for d in directions[-4:]:
        if game_state['explosion_map'][d] >= 1:
            self.logger.debug("aaaaa")
            features[3] = 'stop'
            break

    
    try:
        self.model[", ".join(features)]
    except Exception as ex:
        self.model[", ".join(features)] = list(np.zeros(6))
    return features


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
                if not hfield[n_x, n_y] == 0:
                    continue
            if not (field[n_x, n_y] == 0
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
        return 'left'

    # X coordinate to the right
    if diff[0] > 0:
        return 'right'

    # Y coordinate Up
    if diff[1] < 0:
        return 'up'

    # Y coordinate Down
    if diff[1] > 0:
        return 'down'
