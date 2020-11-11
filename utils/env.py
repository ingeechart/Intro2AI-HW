import os
import numpy as np

from ple import PLE
from ple.games.flappybird import FlappyBird

from utils.utils import convert, make_video


class Environment():
    def __init__(self, device, display=True):
        # Design reward
        reward_values = {
            "positive": 1,
            "tick": 0.1,
            "loss": -1,
        }
        self.env = PLE(FlappyBird(),
                       display_screen=display,
                       reward_values=reward_values)
        self.device = device
        self.action_set = self.env.getActionSet()

        self.frames = []

    def reset(self):
        self.env.reset_game()

    def start(self):
        self.env.act(0)
        obs = convert(self.env.getScreenGrayscale())
        self.state = np.stack([[obs for _ in range(4)]], axis=0)
        self.t_alive = 0
        self.total_reward = 0

        return self.state

    def game_over(self):
        return self.env.game_over()

    def getScore(self):
        return self.env.score()

    def step(self, action):

        reward = self.env.act(self.action_set[action])

        # make next state
        obs = convert(self.env.getScreenGrayscale())
        obs = np.reshape(obs, [1, 1, obs.shape[0], obs.shape[1]])
        next_state = np.append(self.state[:, 1:, ...], obs, axis=1)

        self.t_alive += 1
        self.total_reward += reward
        self.state = next_state

        return self.state, reward, self.env.game_over()

    def get_screen(self):
        return self.env.getScreenRGB()

    def record(self):
        self.frames.append(self.env.getScreenRGB())

    def saveVideo(self, episode, video_path):
        os.makedirs(video_path, exist_ok=True)
        clip = make_video(self.frames, fps=60).rotate(-90)
        clip.write_videofile(os.path.join(
            video_path, 'env_{}.mp4'.format(episode)), fps=60)
        print('Episode: {} t: {} Reward: {:.3f}'.format(
            episode, self.t_alive, self.total_reward))


if __name__ == '__main__':
    env = Environment(device='cpu')
    print(env.action_set)
