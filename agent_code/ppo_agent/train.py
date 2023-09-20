from collections import namedtuple, deque

import pickle
from typing import List
import argparse

import events as e
from .callbacks import state_to_features,BFS,generate_field_from_state

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import sys
sys.path.append("..")

from .TrickyPPO import *

from ..rule_based_agent.callbacks import act as chosen_by_rule

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 16  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.feature_dim = 81
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.total_step = 0
    self.model.mini_batch_size = self.mini_batch_size
    self.model.batch_size = self.batch_size
    print("setting up ReplayBuffer from scartch")
    self.replaybuffer = ReplayBuffer(batch_size=self.batch_size, state_dim=self.feature_dim)

    # setup parameters for rule based agent
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    self.customed_event_list = []
    self.reward_scaling = RewardScaling(shape=1, gamma=0.995)
    self.previous_action = 'NONE'
    self.previous_rule_based_action = 'NONE'
    self.current_step = 0

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def self_in_danger_zone(field,x,y):
    if field[x][y] == -5 or field[x][y] == -3:
        return False
    else:
        return True

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """


    _, score, bombs_old, (x_old, y_old) = old_game_state['self']
    _, score, bombs_new, (x_new, y_new) = new_game_state['self']
    # arena = old_game_state['field']
    # coins = old_game_state['coins']
    # for cx, cy in coins:
    #     arena[cx, cy] = 5
        
    # coin_direction = self.BFS((x,y), self.arena, 5)
    # print("coin_direction ", coin_direction)

    self.current_step += 1
    self.customed_event_list = []
    if self.action_by_rule == True:
        self.rule_based_action = self_action
    else:
        self.rule_based_action = chosen_by_rule(self, old_game_state)

    if self_action == self.rule_based_action:
        self.customed_event_list.append('SAME_AS_RULE')
    # elif not 'BOMB_WHEN_SHOULD' in self.customed_event:
    #     self.customed_event_list.append('NOT_SAME_AS_RULE')
    else:
        self.customed_event_list.append('NOT_SAME_AS_RULE')
        # print("same as rule based action", self.rule_based_action, self_action)
    

    # field[x][y] == -5 when bomb available and in safe region == -3 when bomb inavailable and in safe region
    field_old = generate_field_from_state(self,old_game_state)
    field_new = generate_field_from_state(self,new_game_state)


    # 是否在放炸弹后wait或bomb
    if self.previous_action == 'BOMB':
        if self_action == 'WAIT' or self_action == 'BOMB':
            self.customed_event_list.append('WAIT_AFTER_BOMB')
    self.previous_action = self_action


    # 是否朝着最近的安全区行动，当处于危险区域时, 有问题，在刚扔下炸弹的时候无法立刻触发，总是在能够逃离危险区的时候触发
    direction_toward_nearest_safe_region = BFS((x_old,y_old),field_old,0)
    if self_action == direction_toward_nearest_safe_region and self_in_danger_zone(field_old,x_old,y_old):
        self.customed_event_list.append('HEADIND_TOWARD_NEAREST_SAFE_REGION')
    # 是否朝着bomb行动，也可以不要这个

    # 是否尝试炸一个爆炸范围内的敌人或箱子
    flag_for_this = False
    if self_action == 'BOMB' and old_game_state['self'][2] == True:
        reward_range_for_bomb_crates = 1
        for i in range(reward_range_for_bomb_crates):
            i += 1
            if y_old + i >= field_old.shape[1]:
                break
            # -3 and -2: enemy. 1: crate
            if field_old[x_old][y_old + i] == -3 or field_old[x_old][y_old + i] == -2 or field_old[x_old][y_old + i] == 1:
                flag_for_this = True
                break
        for i in range(reward_range_for_bomb_crates):
            i += 1
            if y_old - i < 0:
                break
            if field_old[x_old][y_old - i] == -3 or field_old[x_old][y_old - i] == -2 or field_old[x_old][y_old - i] == 1:
                flag_for_this = True
                break
        for i in range(reward_range_for_bomb_crates):
            i += 1
            if x_old + i >= field_old.shape[0]:
                break
            if field_old[x_old + i][y_old] == -3 or field_old[x_old + i][y_old] == -2 or field_old[x_old + i][y_old] == 1:
                flag_for_this = True
                break
        for i in range(reward_range_for_bomb_crates):
            i += 1
            if x_old - i < 0:
                break
            if field_old[x_old - i][y_old] == -3 or field_old[x_old - i][y_old] == -2 or field_old[x_old - i][y_old] == 1:
                flag_for_this = True
                break
        if flag_for_this == True:
            self.customed_event_list.append('BOMB_WHEN_SHOULD')
            

    # 是否进入了即将爆炸的危险区，并且不是由于自己扔炸弹导致的
    if not (self_in_danger_zone(field_old,x_old,y_old)) and self_in_danger_zone(field_new,x_new,y_new) and self_action != 'BOMB':  
        self.customed_event_list.append('ENTER_DANGER_ZONE')

    # 是否从自己的爆炸中生存下来
    if bombs_old == False and bombs_new == True:
        self.customed_event_list.append('SURVIVED_WHEN_SELF_BOMB_EXPLODED')
    # 是否逃出了危险区
    if self_in_danger_zone(field_old,x_old,y_old) and not (self_in_danger_zone(field_new,x_new,y_new)):
        self.customed_event_list.append('RUN_OUT_FROM_DANGER_ZONE')

    

    # self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)
    # events.append(PLACEHOLDER_EVENT)
    self.total_step += 1

    a = ACTIONS.index(self_action)
    s = state_to_features(self,old_game_state)
    s_ = state_to_features(self,new_game_state)
    r = reward_from_events(self, events)
    a_logprob = self.log_probs

    dw = False
    done = False
    for e in events:
        if e == "GOT_KILLED" or e == "KILLED_SELF":
            dw = True 
            done = True

    # tell model rule-based-action
    # true_rule_based_action = chosen_by_rule(self, new_game_state)
    # self.previous_rule_based_action = true_rule_based_action

    # state_to_features is defined in callbacks.py
    cur_transition = Transition(s, a, s_, r)._asdict()
    self.replaybuffer.store(s, a, a_logprob, r, s_, dw, done)

    self.transitions.append(cur_transition)
    if self.replaybuffer.count == self.batch_size:
        # print("buffer full, updating when 1 step further")
        self.logger.info(f'buffer full, updating when 1 step further and change self.action_by_rule from {self.action_by_rule} to {not self.action_by_rule}')
        # self.action_by_rule = not self.action_by_rule
        self.model.update(self.replaybuffer, self.total_step)
        self.replaybuffer.count = 0

    


def end_of_round(self, last_game_state: dict, last_action: str, state_after_last_game_state: dict, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.customed_event_list = []
    self.total_step += 1
    self.current_step = 0

    if self.action_by_rule == True:
        self.rule_based_action = last_action
    else:
        self.rule_based_action = chosen_by_rule(self, last_game_state)
    if last_action == self.rule_based_action: 
        self.customed_event_list.append('SAME_AS_RULE')
    else:
        self.customed_event_list.append('NOT_SAME_AS_RULE')
        
        # print("same as rule based action", self.rule_based_action, last_action)

    if state_after_last_game_state == None:
        state_after_last_game_state = last_game_state


    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    agent_name,agent_score,agent_bomb_left,(agent_x,agent_y) = last_game_state['self']
    if last_action == 'UP':
        agent_y -= 1
    elif last_action == 'DOWN':
        agent_y += 1
    elif last_action == 'LEFT':
        agent_x -= 1
    elif last_action == 'RIGHT':
        agent_x += 1
    state_after_last_game_state['self'] = agent_name,agent_score,agent_bomb_left,(agent_x,agent_y)



    a = ACTIONS.index(last_action)
    s = state_to_features(self,last_game_state)
    s_ = state_to_features(self,state_after_last_game_state)
    r = reward_from_events(self, events)
    a_logprob = self.log_probs

    dw = False
    done = True
    for e in events:
        if e == "GOT_KILLED" or e == "KILLED_SELF":
            dw = True 

    self.reward_scaling.reset()

    self.previous_action = 'NONE'
    self.previous_rule_based_action = 'NONE'
    # state_to_features is defined in callbacks.py
    self.replaybuffer.store(s, a, a_logprob, r, s_, dw, done)

    if self.replaybuffer.count == self.batch_size:
        # print("buffer full, updating when end of round")
        # self.action_by_rule = not self.action_by_rule
        self.logger.info(f'buffer full, updating when end of round and change self.action_by_rule to {self.action_by_rule}')
        self.model.update(self.replaybuffer, self.total_step)
        self.replaybuffer.count = 0


    field_old = generate_field_from_state(self,last_game_state)
    _, score, bombs_old, (x_old, y_old) = last_game_state['self']
    
    # 是否进入了即将爆炸的危险区
    if not (self_in_danger_zone(field_old,x_old,y_old)) and last_action != 'BOMB' and last_action != 'WAIT':  
        self.customed_event_list.append('ENTER_DANGER_ZONE')
    
    reset_self(self)

    if(last_game_state["round"]%10 == 0):
        self.logger.info(f' round :{last_game_state["round"]}, score in this round:{last_game_state["self"][1]}')
        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 10,
        e.KILLED_SELF: -5,
        e.GOT_KILLED: -5,
        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 1,
        # e.SURVIVED_ROUND: 20,
        e.MOVED_LEFT: -.5,
        e.MOVED_RIGHT: -.5,
        e.MOVED_UP: -.5,
        e.MOVED_DOWN: -.5,
        e.WAITED: -.5,
        e.BOMB_DROPPED: -.5,
        e.INVALID_ACTION: -1,
        e.ClOSING_COIN: 0.1,
        e.SAME_AS_RULE: 1.1,
        e.NOT_SAME_AS_RULE: -2,
        e.BOMB_WHEN_SHOULD: 0.6,
        e.HEADIND_TOWARD_NEAREST_SAFE_REGION: 0.6,
        e.ENTER_DANGER_ZONE: -2,
        e.SURVIVED_WHEN_SELF_BOMB_EXPLODED: 0.6,
        e.RUN_OUT_FROM_DANGER_ZONE: 0.6,
        e.WAIT_AFTER_BOMB: -0.5
    }
    reward_sum = 0
    for my_e in self.customed_event_list:
        events.append(my_e)
        
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    reward_sum *= 2
    self.logger.info(f'Awarded {reward_sum} for events {", ".join(events)} in step {self.current_step} Rule {self.rule_based_action}')
    scaled_reward = self.reward_scaling(reward_sum)

    return scaled_reward

def state_value(self, feature_vector: np.array) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    state_value_approximation = np.dot(feature_vector.T,self.model)
    return state_value_approximation
