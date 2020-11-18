#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
submissions
├┬ 0657
│├- 홍길동_180000_assignsubmission_file_
│└- 홍길동_180000_assignsubmission_onlinetext_
├- 0658
└- 0659
---
>> python evaluator.py [--path ./submissions]
"""
import os
import re
import sys
import argparse

import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.env import Environment   # noqa: E402

CHECKPOINT_PATH = 'checkpoint.pth.tar'
BEST_CHECKPOINT_PATH = 'best_checkpoint.pth.tar'
SAMPLE_SIZE = 5
CUTLINE = 200

regex = re.compile(r'(?P<name>[\w\s]+)_[\d]+_assignsubmission_file_')


def init(path):
    classes = os.listdir(path)
    for klass in classes:
        submissions = os.path.join(path, str(klass))
        for student in os.listdir(submissions):
            student_path = os.path.join(submissions, student)
            files = os.listdir(student_path)
            python_files = list(filter(lambda x: x.endswith('.py'), files))
            assert len(python_files) == 1
            os.rename(os.path.join(student_path, python_files[0]), os.path.join(student_path, 'dqn.py'))


def evaluate(env, model, n=SAMPLE_SIZE):
    scores = []
    model.to(device)
    for _ in range(n):
        env.reset()
        state = env.start()
        total_reward = 0
        while not env.game_over() and total_reward < CUTLINE:
            state = torch.from_numpy(state).float() / 255.0
            state = state.to(device)
            action = model(state).max(1)[1].cpu().data[0]
            state, reward, done = env.step(action)
            total_reward += reward
        scores.append(total_reward)
    return np.mean(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'submissions'))
    args = parser.parse_args()

    init(args.path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Environment(device, display=False)

    classes = os.listdir(args.path)
    for clazz in classes:
        students = os.listdir(os.path.join(args.path, str(clazz)))
        for student in students:
            student_name = regex.search(student).group('name')
            student_path = os.path.join(args.path, str(clazz), student)

            if os.path.exists(os.path.join(student_path, BEST_CHECKPOINT_PATH)):
                checkpoint = torch.load(os.path.join(student_path, BEST_CHECKPOINT_PATH))
            elif os.path.exists(os.path.join(student_path, CHECKPOINT_PATH)):
                checkpoint = torch.load(os.path.join(student_path, CHECKPOINT_PATH))
            else:
                sys.stderr.write('[ERROR] No checkpoint found - %s(%s)\n' % (student_name, clazz))
                with open(os.path.join(os.path.dirname(__file__), str(clazz) + '.csv'), 'at', encoding='utf-8') as f:
                    f.write('%s,%f\n' % (student_name, 0.0))
                continue

            sys.path.append(student_path)

            try:
                del sys.modules['dqn']  # Remove cache
            except KeyError:
                pass

            from dqn import DQN

            model = DQN(84, 84, len(env.action_set))
            model.load_state_dict(checkpoint['state_dict'])
            mean_score = evaluate(env, model)
            print('[%s] %s: %f' % (clazz, student_name, mean_score))

            with open(os.path.join(os.path.dirname(__file__), str(clazz) + '.csv'), 'at', encoding='utf-8') as f:
                f.write('%s,%f\n' % (student_name, mean_score))

            sys.path.pop()
