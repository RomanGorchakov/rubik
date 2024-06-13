#!usr/env/bin python3
# -*- coding: utf-8 -*-


import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv

class RubikEnv(gym.Env):
    def __init__(self):
        super(RubikEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(12)  # 6 faces * 2 directions
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6, 3, 3), dtype=np.float32)
        self.state = self.reset()

    def step(self, action):
        self._take_action(action)
        reward = self._get_reward()
        done = self._is_done()
        obs = self.state
        return obs, reward, done, {}

    def reset(self):
        self.state = self._scramble_cube()
        return self.state

    def render(self, mode='human'):
        pass

    def _take_action(self, action):
        # Действие должно корректно обновлять состояние кубика
        print(f"Action taken: {action}")  # Добавляем отладочное сообщение
        # Обновление состояния кубика на основе действия (упрощенное для примера)
        self.state = np.random.rand(6, 3, 3)  # Упрощенное действие для примера

    def _get_reward(self):
        return 1 if self._is_done() else 0

    def _is_done(self):
        # Упрощенная проверка завершения (для примера)
        return False

    def _scramble_cube(self):
        # Случайное состояние кубика для отладки
        return np.random.rand(6, 3, 3)

# Создаем векторизированную среду
env = DummyVecEnv([lambda: RubikEnv()])

# Обучаем модель DQN
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save('rubik_solver')

def solve_rubik_cube(env):
    model = DQN.load('rubik_solver')
    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        print(f"Action: {action}, Rewards: {rewards}, Done: {done}")  # Отладочное сообщение
        if done:
            break
    return env.state