import time
import os
import numpy as np
import tensorflow as tf

from gym.utils import colorize

from rlarm.algorithms.tools.utils import REPO_ROOT
    
    
####################################################################################################
class Policy_Base():
    def __init__(self, name, env, deterministic, save_tensorboard, save_matplotlib):
        self.name = name
        self.env = env
        self.save_tensorboard = save_tensorboard
        self.save_matplotlib = save_matplotlib
        
        if deterministic:
            np.random.seed(1)
            tf.random.set_seed(1)
            
        self.act_dim = 2
        self.state_dim = (9,)

        print(colorize("State dim: {}".format(self.state_dim), "yellow"))
        print(colorize("Act dim: {}".format(self.act_dim), "yellow"))

        if self.save_tensorboard:
            os.makedirs(os.path.join(REPO_ROOT, 'tensorboard'), exist_ok = True)
            self.writer = tf.summary.create_file_writer(os.path.join(REPO_ROOT, 'tensorboard', self.name))

    # ----------------------------------------------------------------------------------------------------
    class TrainConfig():
        pass

    # ----------------------------------------------------------------------------------------------------
    def reset(self):
        return self.env.reset()

    # ----------------------------------------------------------------------------------------------------
    def step(self, action):
        ob_next, r, done = self.env.step(action)
        self.env.render()
        return ob_next, r, done

    # ----------------------------------------------------------------------------------------------------
    def get_sample(self):
        return self.env.sample_action()
    
    # ----------------------------------------------------------------------------------------------------
    def act(self, state):
        pass

    # ----------------------------------------------------------------------------------------------------
    def fit_net(self, lr, returns, obs, actions, batch_size):
        pass

    # ----------------------------------------------------------------------------------------------------
    def train(self, config: TrainConfig):
        pass