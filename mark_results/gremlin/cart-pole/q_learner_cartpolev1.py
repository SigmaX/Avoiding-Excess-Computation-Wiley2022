#!/usr/bin/self._env python3
"""
    Solution to the CartPole model

    Q-Learning solution for OpenAI Gym Cart Pole V1.

    Inspired by Ali Fakhry's approach
    https:#medium.com/swlh/using-q-learning-for-openais-cartpole-v1-4a216ef237df

"""
import time
import math
import pickle
from os import makedirs
from os.path import dirname, exists
from typing import Dict, List
from random import gauss

from tqdm import tqdm
import numpy as np
import gym
import pandas as pd


class CartPoleV1_Q_Learner:
    """Q learner for OpenAI Gym Cart Pole V1.
    Inspired by: Ali Fakhry's approach
    https:#medium.com/swlh/using-q-learning-for-openais-cartpole-v1-4a216ef237df

    """

    def __init__(self) -> None:
        self._env_name = "CartPole-v1"
        self._env = gym.make(self._env_name)
        # set up randomized q-table
        Observation = [30, 30, 50, 50]
        self._q_table = np.random.uniform(
            low=0, high=1, size=(Observation + [self._env.action_space.n])
        )
        self._env.close()

    def _get_discrete_state(self, state: np.array):
        np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])
        discrete_state = state / np_array_win_size + np.array([15, 10, 1, 10])
        return tuple(discrete_state.astype(np.int))

    def save(self, fpath: str) -> None:
        """Save model q-table to .pkl file

        :param fpath: file path to save q-table .pkl file to
        """
        if not exists(dirname(fpath)):
            makedirs(dirname(fpath))
        with open(fpath, "wb") as f:
            pickle.dump(self._q_table, f)

    def load(self, fpath: str) -> np.array:
        """Load model (q-table) from .pkl file

        :param fpath: file path to saved q-table .pkl file
        """
        with open(fpath, "rb") as f:
            self._q_table = pickle.load(f)

    def train(
        self,
        is_verbose: bool,
        is_render: bool,
        initial_state_dist: Dict[str, Dict[str, float]] = None,
        learning_rate=0.1,
        discount=0.95,
        episodes=60000,
        epsilon=0.05,
        epsilon_decay_value=0.99995,
    ) -> pd.DataFrame:
        """Do the actual training

        :param is_verbose: chatty output
        :param is_render: render animation of cart pole
        :param initial_state_dist: if None, random init state from Gym used. Else
            if type Dict, keys should be in ['angle', 'position',
            'rotational_velocity', 'velocity'] and values should also be Dicts
            with keys ['mean', 'median', 'std'] mapped to the respective values.
        :param learning_rate: Q learner learning rate
        :param discount:= Q learner discount,
        :param episodes:= Q learner training episode count,
        :param epsilon:= Q learner epsilon ,
        :param epsilon_decay_value: Q learner epsilon decay,
        :return: training progress DataFrame
        """
        # If stats is NOT None, then we know we're resuming training of a previous
        # model.  Therefore we need to load the previous model, and then while
        # training, only randomly sample from within the distributions found in
        # stats.
        self._env = gym.make(self._env_name)
        total = 0
        total_reward = 0
        prior_reward = 0

        progress = []
        for episode in tqdm(
            range(episodes), desc="Training Q Learner"
        ):  # go through the self._episodes
            t0 = time.time()  # set the initial time
            initial_state = self._env.reset()
            self._env.seed(int(time.time()))
            # Use specified initial state if provided.
            if initial_state_dist != None:
                params = [
                    "angle",
                    "position",
                    "rotational_velocity",
                    "velocity",
                ]
                self._env.state = [
                    gauss(
                        initial_state_dist.get(param).get("mean", 0),
                        initial_state_dist.get(param).get("std", 0),
                    )
                    for param in params
                ]
            discrete_state = self._get_discrete_state(
                initial_state
            )  # get the discrete start for the restarted self._environment
            done = False
            episode_reward = 0  # reward starts as 0 for each episode
            if is_verbose and (episode % 2000 == 0):
                print("Episode: " + str(episode))
            while not done:
                if np.random.random() > epsilon:
                    action = np.argmax(
                        self._q_table[discrete_state]
                    )  # take cordinated action
                else:
                    action = np.random.randint(
                        0, self._env.action_space.n
                    )  # do a random ation
                new_state, reward, done, _ = self._env.step(
                    action
                )  # step action to get new states, reward, and the 'done' status.
                episode_reward += reward  # add the reward
                new_discrete_state = self._get_discrete_state(new_state)
                if is_render and (episode % 2000 == 0):  # render
                    self._env.render()
                if not done:  # update q-table
                    max_future_q = np.max(self._q_table[new_discrete_state])
                    current_q = self._q_table[discrete_state + (action,)]
                    new_q = (1 - learning_rate) * current_q + learning_rate * (
                        reward + discount * max_future_q
                    )
                    self._q_table[discrete_state + (action,)] = new_q
                discrete_state = new_discrete_state
                """if epsilon > 0.05:  # epsilon modification
                    if episode_reward > prior_reward and episode > 10000:
                        epsilon = math.pow(epsilon_decay_value, episode)
                        if is_verbose and (episode % 500 == 0):
                            print("epsilon: " + str(epsilon))"""
            t1 = time.time()  # episode has finished
            episode_total = t1 - t0  # episode total time
            total = total + episode_total
            total_reward += episode_reward  # episode total reward
            prior_reward = episode_reward
            if is_verbose and (episode % 1000 == 0):
                mean = total / 1000
                print("Time Average: " + str(mean))
                total = 0
                mean_reward = total_reward / 1000
                print("Mean Reward: " + str(mean_reward))
                total_reward = 0
            episode_progress = [episode, episode_reward, episode_total]
            episode_progress.extend(initial_state)
            progress.append(episode_progress)

        self._env.close()
        progress = pd.DataFrame(
            progress,
            columns=[
                "Episode",
                "Reward",
                "TrainTime",
                "Position",
                "Velocity",
                "Angle",
                "RotationalVelocity",
            ],
        )
        return progress

    def predict(self, initial_state: List[float]) -> int:
        """Runs the model on a provided initial_state

        :param initial_state: initial state of cart pole:
            position, velocity, angle, rotational velocity
            in order.
        """
        # Run model for this state
        self._env = gym.make(self._env_name)
        self._env.reset()
        self._env.seed(int(time.time()))
        self._env.state = initial_state
        discrete_state = self._get_discrete_state(initial_state)

        done = False
        total_reward = 0
        while not done:
            # take cordinated action
            action = np.argmax(self._q_table[discrete_state])
            # step action to get new states, reward, and the 'done' status.
            new_state, reward, done, _ = self._env.step(action)
            # add the reward
            total_reward += reward
            discrete_state = self._get_discrete_state(new_state)
            # self._env.render()
        # Cleaning up after yourself is just nice.
        self._env.close()
        return total_reward
