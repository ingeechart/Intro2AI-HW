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
import zipfile

import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.env import Environment   # noqa: E402

CHECKPOINT_PATH = 'checkpoint.pth.tar'
BEST_CHECKPOINT_PATH = 'best_checkpoint.pth.tar'
SAMPLE_SIZE = 5
CUTLINE = 200

regex = re.compile(r'(?P<name>[\w\s]+)_[\d]+_assignsubmission_file_')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def __remove_directory(path):
    try:
        if os.path.isdir(path):
            os.rmdir(path)
        elif os.path.isfile(path):
            os.remove(path)
    except:
        for f in os.listdir(path):
            __remove_directory(os.path.join(path, f))
        os.rmdir(path)


def init(path):
    classes = os.listdir(path)
    for klass in classes:
        submissions = os.path.join(path, str(klass))
        for student in os.listdir(submissions):
            if student.endswith('_assignsubmission_onlinetext_'):
                __remove_directory(os.path.join(submissions, student))  # continue
                continue
            student_path = os.path.join(submissions, student)
            files = os.listdir(student_path)
            # 1. zip
            if not [f for f in files if f.endswith('.py') or f.endswith('.tar')] and any(list(filter(lambda x: x.endswith('.zip'), files))):
                zip_file = list(filter(lambda x: x.endswith('.zip'), files))[0]
                print('[zip] %s - %s' % (student, zip_file))
                with zipfile.ZipFile(os.path.join(student_path, zip_file)) as zf:
                    zf.extractall(student_path)
                    if any(list(filter(lambda x: os.path.isdir(os.path.join(student_path, x)), os.listdir(student_path)))):
                        dirname = list(filter(lambda x: os.path.isdir(os.path.join(student_path, x)), os.listdir(student_path)))[0]
                        for extracted in os.listdir(os.path.join(student_path, dirname)):
                            print('extracted:', extracted)
                            if not os.path.exists(os.path.join(student_path, extracted)):
                                os.rename(os.path.join(student_path, dirname, extracted), os.path.join(student_path, extracted))
                files = os.listdir(student_path)
            python_files = list(filter(lambda x: x.endswith('.py'), files))
            assert len(python_files) == 1
            os.rename(os.path.join(student_path, python_files[0]), os.path.join(student_path, 'dqn.py'))
            checkpoint_files = list(filter(lambda x: x.endswith('.tar'), files))
            assert len(checkpoint_files) >= 1
            if not os.path.exists(os.path.join(student_path, CHECKPOINT_PATH)):
                os.rename(os.path.join(student_path, checkpoint_files[0]), os.path.join(student_path, CHECKPOINT_PATH))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'submissions'))
    args = parser.parse_args()

    init(args.path)

    env = Environment(device, display=False)

    classes = os.listdir(args.path)
    for clazz in classes:
        students = os.listdir(os.path.join(args.path, str(clazz)))
        for student in students:
            if student.endswith('_assignsubmission_onlinetext_'):
                continue
            student_name = regex.search(student).group('name')
            student_path = os.path.join(args.path, str(clazz), student)

            print('<< %s (%s) >>' % (student_name, clazz))

            if os.path.exists(os.path.join(student_path, BEST_CHECKPOINT_PATH)):
                checkpoint = torch.load(os.path.join(student_path, BEST_CHECKPOINT_PATH))
                print('- checkpoint:', os.path.join(student_path, BEST_CHECKPOINT_PATH))
            elif os.path.exists(os.path.join(student_path, CHECKPOINT_PATH)):
                checkpoint = torch.load(os.path.join(student_path, CHECKPOINT_PATH))
                print('- checkpoint:', os.path.join(student_path, CHECKPOINT_PATH))
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

            print('- model:', sys.modules['dqn'])
            model = DQN(84, 84, len(env.action_set))
            try:
                model.load_state_dict(checkpoint['state_dict'])
                mean_score = evaluate(env, model)
                print('[%s] %s: %f' % (clazz, student_name, mean_score))
            except RuntimeError as e:
                sys.stderr.write('[ERROR] %s - RuntimeError\n' % (student,))
                sys.stderr.write('%s\n' % (e,))
                mean_score = 0.0

            with open(os.path.join(os.path.dirname(__file__), str(clazz) + '.csv'), 'at', encoding='utf-8') as f:
                f.write('%s,%f\n' % (student_name, mean_score))

            sys.path.pop()

        with open(os.path.join(os.path.dirname(__file__), str(clazz) + '.csv'), 'at', encoding='utf-8') as f:
            f.write('----------\n')


if __name__ == "__main__":
    main()
