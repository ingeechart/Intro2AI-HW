#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os

import torch
import numpy as np

from dqn import DQN
from utils.env import Environment

CHECKPOINT_PATH = 'checkpoint.pth.tar'
BEST_CHECKPOINT_PATH = 'best_checkpoint.pth.tar'
SAMPLE_SIZE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(env, model, n=SAMPLE_SIZE):
    scores = []
    model.to(device)
    for _ in range(n):
        env.reset()
        state = env.start()
        total_reward = 0
        while not env.game_over():
            state = torch.from_numpy(state).float() / 255.0
            state = state.to(device)
            action = model(state).max(1)[1].cpu().data[0]
            state, reward, done = env.step(action)
            total_reward += reward
        scores.append(total_reward)
    return np.mean(scores)


if __name__ == "__main__":
    model = DQN(84, 84, 2)
    checkpoint_path = list(filter(lambda x: x.endswith('.pth.tar'), os.listdir(os.path.dirname(__file__))))
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), checkpoint_path[0]))
    model.load_state_dict(checkpoint['state_dict'])

    print(evaluate(Environment(device), model))
