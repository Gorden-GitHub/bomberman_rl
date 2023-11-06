from collections import namedtuple, deque

import pickle
import json
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np
import pandas as pd

ACTION_TO_INDEX = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
alpha = 0.25
gamma = 0.65

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
CRATE_TOWARDS = "moved towards the closest reachable crate"
COIN_TOWARDS = "moved towards the closest reachable coin"
DANGER_AWAY = "away from danger"
MAB = "meet something and bomb possible"
MBN = "meet something but no bomb"
SUICIDE_AVOID = "Suicide avoid"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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


    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    

    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    if self_action.lower() == old_features[0]:
        events.append(COIN_TOWARDS)
    if self_action.lower() == old_features[1]:
        events.append(CRATE_TOWARDS)
    if 'true' == old_features[2]:
        events.append(MAB)
    if self_action.lower() == old_features[3]:
        events.append(DANGER_AWAY)
    if old_features[1] == 'met' and old_features[2] == 'false':
        if self_action == 'WAIT':
            events.append(MBN)
    if old_features[3] == 'stop':
        if self_action == 'WAIT':
            events.append(SUICIDE_AVOID)


    rewards = reward_from_events(self, events)
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Update Q-table
    Qs0a0 = self.model[", ".join(old_features)][ACTION_TO_INDEX[self_action]]
    Qs1a1 = max(self.model[", ".join(new_features)])
    new = Qs0a0 + alpha * (rewards + gamma * Qs1a1 - Qs0a0)
    self.model[", ".join(old_features)][ACTION_TO_INDEX[self_action]] = new



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    last_features = state_to_features(self, last_game_state)
    

    if last_action.lower() == last_features[0]:
        events.append(COIN_TOWARDS)
    if last_action.lower() == last_features[1]:
        events.append(CRATE_TOWARDS)
    if 'true' == last_features[2]:
        events.append(MAB)
    if last_action.lower() == last_features[3]:
        events.append(DANGER_AWAY)
    if last_features[1] == 'met' and last_features[2] == 'false':
        if last_action == 'WAIT':
            events.append(MBN)

    rewards = reward_from_events(self, events)
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    
    Qs0a0 = self.model[", ".join(last_features)][ACTION_TO_INDEX[last_action]]
    new = Qs0a0 + alpha * (rewards - Qs0a0)
    self.model[", ".join(last_features)][ACTION_TO_INDEX[last_action]] = new




    # Store the model
    # with open("my-saved-model.pt", "wb") as file:
    #     pickle.dump(self.model, file)
    with open("q_table.json", "w") as file:
        file.write(json.dumps(self.model))
    
    df = pd.DataFrame(self.model).transpose()
    df.to_csv('data.csv')


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 22,
        e.KILLED_OPPONENT: 5,
        COIN_TOWARDS: 20,
        CRATE_TOWARDS: 10,  
        e.KILLED_SELF: -30,
        e.INVALID_ACTION: -20,
        e.WAITED: 1,
        e.CRATE_DESTROYED: 10,
        e.COIN_FOUND: 3,
        DANGER_AWAY: 20,
        MAB: 10,
        MBN: 10,
        SUICIDE_AVOID: 20,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
