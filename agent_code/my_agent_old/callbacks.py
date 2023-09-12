import os
import json
from random import shuffle
import numpy as np


#ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = ["LEFT","UP","RIGHT","DOWN","BOMB","WAIT"]


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
    
    self.state = None

    if self.train or not os.path.isfile("q_table.json"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading saved q_table.")
        with open("q_table.json", "r") as file:
            self.q_table = json.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    # random_prob = .1
    # if self.train and random.random() < random_prob:
    #     self.logger.debug("Choosing action purely at random.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.
    #     return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=self.model)

    self.state = get_state(self, game_state)
    try:
        knowledge_list = np.array(self.q_table[", ".join(self.state)])
        best_indices = np.where(knowledge_list == max(knowledge_list))[0]
        action = np.random.choice(best_indices)
    except KeyError as ke:
            self.logger.debug(2)
            action = np.random.randint(0,6)
    return ACTIONS[action]
    


def get_state(self, game_state):
    state = list()

    _, _, _, (x, y) = game_state['self']
    arena = game_state['field']
    others = [xy for (n, s, b, xy) in game_state['others']]
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
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
    coins = game_state['coins']


    # self, right, left, down, up
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] 
    for d in directions:
        if arena[d] == 1:
            state.append("crate")
            continue
        elif arena[d] == -1:
            state.append("wall")
            continue
        
        # tile an explosion will be present
        # to be determine
        if game_state['explosion_map'][d] >= 1:
            state.append("wall")
            continue
        
        if d in bomb_xys:
            state.append("bomb")
            continue

        # 检测d与炸弹之间有没有wall或者crate
        if bomb_map[d] == 0:
            state.append("danger")
            continue
        
        if d in others:
            state.append("enemy")
            continue
        
        if d in coins:
            state.append("coin")
            continue

        state.append("empty")

    # 避免放置炸弹后走进死胡同
    if state[0] == "bomb":
        if state[1] == "empty":
            if (arena[(x + 1, y + 1)] != 0 and arena[(x + 1, y - 1)] != 0 and
                arena[(x + 2, y)] != 0):
                state[1] = "wall"
        if state[2] == "empty":
            if (arena[(x - 1, y + 1)] != 0 and arena[(x - 1, y - 1)] != 0 and
                arena[(x - 2, y)] != 0):
                state[2] = "wall"
        if state[3] == "empty":
            if (arena[(x - 1, y + 1)] != 0 and arena[(x + 1, y + 1)] != 0 and
                arena[(x, y + 2)] != 0):
                state[3] = "wall"
        if state[4] == "empty":
            if (arena[(x - 1, y - 1)] != 0 and arena[(x + 1, y - 1)] != 0 and
                arena[(x, y - 2)] != 0):
                state[4] = "wall"
    
    # 周围没有威胁和硬币的时候
    if all(t in ["wall", "empty"] for t in state[1:]):
        free_space = arena == 0
        cols = range(1, arena.shape[0] - 1)
        rows = range(1, arena.shape[0] - 1)
        dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                    and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
        targets = coins + dead_ends + crates + others
        targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]
        frontier = [(x, y)]
        parent_dict = {(x, y): (x, y)}
        dist_so_far = {(x, y): 0}
        best = (x, y)
        best_dist = np.sum(np.abs(np.subtract(targets, (x, y))), axis=1).min()

        while len(frontier) > 0:
            current = frontier.pop(0)
            # Find distance from current position to all targets, track closest
            d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
            if d + dist_so_far[current] <= best_dist:
                best = current
                best_dist = d + dist_so_far[current]
            if d == 0:
                # Found path to a target's exact position, mission accomplished!
                best = current
                break
            # Add unexplored free neighboring tiles to the queue in a random order
            x, y = current
            neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
            shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in parent_dict:
                    frontier.append(neighbor)
                    parent_dict[neighbor] = current
                    dist_so_far[neighbor] = dist_so_far[current] + 1
        # Determine the first step towards the best found target tile
        current = best
        while True:
            if parent_dict[current] == (x, y):
                target = current
                break
            current = parent_dict[current]
        
        if target == (x + 1, y): state[1] = "priority"
        if target == (x - 1, y): state[2] = "priority"
        if target == (x, y + 1): state[3] = "priority"
        if target == (x, y - 1): state[4] = "priority"
    
    state.append(str(game_state["self"][2] == 0))

    str_state = ", ".join(state)
    try:
        self.q_table[str_state]
    except Exception as ex:
        self.q_table[str_state] = list(np.zeros(6))
    
    return state

