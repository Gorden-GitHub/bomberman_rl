from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS
import sys

import numpy as np

from agent_code.rule_based_agent.callbacks import act as ra
from agent_code.rule_based_agent.callbacks import setup as ru


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
SAME = "rule"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.update_cnt = 0
    #self.losses = []
    #self.scores = []
    self.score = 0
    self.step = 1

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
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    ru(self)
    if ra(self, old_game_state) == self_action:
        events.append(SAME)
    

    rewards = reward_from_events(self, events)
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    self.model.transition = [state_to_features(old_game_state), ACTIONS.index(self_action), rewards, state_to_features(new_game_state), False]
    # N-step transition
    if self.model.use_n_step:
        one_step_transition = self.model.memory_n.store(*self.model.transition)
    # 1-step transition
    else:
        one_step_transition = self.model.transition
    # add a single step transition
    if one_step_transition:
        self.model.memory.store(*one_step_transition)
    
    #self.score += rewards

    fraction = min(self.step / self.num_steps, 1.0)
    self.model.beta = self.model.beta + fraction * (1.0 - self.model.beta)
    self.step += 1
    

    # if training is ready
    if len(self.model.memory) >= self.model.batch_size:
        loss = self.model.update_model()
        #self.losses.append(loss)
        self.update_cnt += 1

        # if hard update is needed
        if self.update_cnt % self.model.target_update == 0:
            self.model._target_hard_update()
            with open("my-saved-model.pt", "wb") as file:
                pickle.dump(self.model, file)
    
    if self.step > self.num_steps:
        sys.exit(0)

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
    
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    # with open("my-saved-model.pt", "wb") as file:
    #     pickle.dump(self.model, file)

    ru(self)
    if ra(self, last_game_state) == last_action:
        events.append(SAME)

    rewards = reward_from_events(self, events)
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.model.transition = [state_to_features(last_game_state), ACTIONS.index(last_action), rewards, np.zeros(17*17*2), True]
    # N-step transition
    if self.model.use_n_step:
        one_step_transition = self.model.memory_n.store(*self.model.transition)
    # 1-step transition
    else:
        one_step_transition = self.model.transition
    # add a single step transition
    if one_step_transition:
        self.model.memory.store(*one_step_transition)
    
    #self.score += rewards

    fraction = min(self.step / self.num_steps, 1.0)
    self.model.beta = self.model.beta + fraction * (1.0 - self.model.beta)
    self.step += 1

    # if training is ready
    if len(self.model.memory) >= self.model.batch_size:
        loss = self.model.update_model()
        #self.losses.append(loss)
        self.update_cnt += 1

        # if hard update is needed
        if self.update_cnt % self.model.target_update == 0:
            self.model._target_hard_update()
    
    if self.step > self.num_steps:
        sys.exit(0)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 600,
        e.KILLED_OPPONENT: 10,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        e.KILLED_SELF: -200,
        e.GOT_KILLED: -200,
        e.INVALID_ACTION: -600,
        e.CRATE_DESTROYED: 10,
        e.WAITED: -800,
        SAME: 800,

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
