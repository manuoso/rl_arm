import tensorflow as tf
import numpy as np

import gym
from gym.utils import colorize

    
####################################################################################################
class CRITIC_V(tf.keras.Model):
    def __init__(self, state_shape, units=[256, 256]):
        super().__init__()
        assert len(units) > 1, "layer_sizes must be greater than one."
        print(colorize("CRITIC NET | layer_sizes: {}".format(len(units)), "green"))
        
        print(colorize("CRITIC NET Layer: {} | units: {}".format(0, units[0]), "green"))
        self.l1 = tf.keras.layers.Dense(units[0], name="L1", activation='relu')
        
        print(colorize("CRITIC NET Layer: {} | units: {}".format(1, units[1]), "green"))
        self.l2 = tf.keras.layers.Dense(units[1], name="L2", activation='relu')
        
        print(colorize("CRITIC NET Layer: {} | units: {}".format(2, 1), "green"))
        self.l3 = tf.keras.layers.Dense(1, name="L3", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        with tf.device("/cpu:0"):
            self(dummy_state)

    def call(self, states):
        features = self.l1(states)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1, name="values")

####################################################################################################
class CRITIC_Q(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[256, 256]):
        super().__init__()
        assert len(units) > 1, "layer_sizes must be greater than one."
        print(colorize("CRITIC NET | layer_sizes: {}".format(len(units)), "green"))
        
        print(colorize("CRITIC NET Layer: {} | units: {}".format(0, units[0]), "green"))
        self.l1 = tf.keras.layers.Dense(units[0], name="L1", activation='relu')
        
        print(colorize("CRITIC NET Layer: {} | units: {}".format(1, units[1]), "green"))
        self.l2 = tf.keras.layers.Dense(units[1], name="L2", activation='relu')
        
        print(colorize("CRITIC NET Layer: {} | units: {}".format(2, 1), "green"))
        self.l3 = tf.keras.layers.Dense(1, name="L3", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        with tf.device("/cpu:0"):
            self([dummy_state, dummy_action])

    def call(self, inputs):
        [states, actions] = inputs
        features = tf.concat([states, actions], axis=1)
        features = self.l1(features)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1)
