import tensorflow as tf
import numpy as np

import gym
from gym.utils import colorize

from rlarm.algorithms.tools.huber_loss import huber_loss


####################################################################################################
class ACTOR_DDPG(tf.keras.Model):
    def __init__(self, state_shape, max_action, layer_sizes = 3, units = [400, 300, 1]):
        super(ACTOR_DDPG, self).__init__()
        assert layer_sizes > 1, "layer_sizes must be greater than one."
        assert layer_sizes == len(units), "layer sizes must match with length of units."
        
        self.max_action = max_action
        
        print(colorize("ACTOR NET | layer_sizes: {}".format(layer_sizes), "green"))

        # Last unit is action dim
        self.layers_actor = []
        for i in range(layer_sizes):
            print(colorize("ACTOR NET Layer: {} | units: {}".format(i, units[i]), "green"))
            self.layers_actor.append(tf.keras.layers.Dense(units=units[i]))

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32)))
                                     
    def call(self, s):
        x = s
        for i in range(0, len(self.layers_actor)-1):
            x = tf.nn.relu(self.layers_actor[i](x))
        
        x = self.layers_actor[len(self.layers_actor)-1](x)
        action = self.max_action * tf.nn.tanh(x)
        return action
    
####################################################################################################
class CRITIC_DDPG(tf.keras.Model):
    def __init__(self, state_shape, action_dim, layer_sizes = 3, units = [400, 300, 1]):
        super(CRITIC_DDPG, self).__init__()
        assert layer_sizes > 1, "layer_sizes must be greater than one."
        assert layer_sizes == len(units), "layer sizes must match with length of units."
        
        print(colorize("CRITIC NET | layer_sizes: {}".format(layer_sizes), "green"))

        # Last unit is 1
        self.layers_critic = []
        for i in range(layer_sizes):
            print(colorize("CRITIC NET Layer: {} | units: {}".format(i, units[i]), "green"))
            self.layers_critic.append(tf.keras.layers.Dense(units=units[i]))
        
        dummy_state = tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        
        with tf.device("/cpu:0"):
            self(dummy_state, dummy_action)
                                     
    def call(self, s, a):
        x = tf.concat([s, a], axis=1)
        
        for i in range(0, len(self.layers_critic)-1):
            x = tf.nn.relu(self.layers_critic[i](x))                   
        
        x = self.layers_critic[len(self.layers_critic)-1](x)
            
        return x        
