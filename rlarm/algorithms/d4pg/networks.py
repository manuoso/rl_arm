import tensorflow as tf
import numpy as np

import gym
from gym.utils import colorize

from rlarm.algorithms.d4pg.l2_projection import _l2_project


####################################################################################################
class ACTOR_NET(tf.keras.Model):
    def __init__(self, action_bound_high, action_bound_low, state_dims, action_dims, layers_units = [400, 300]):
        super(ACTOR_NET, self).__init__()
        
        print(colorize("ACTOR NET | dense1 size & dense2 size: {}".format(layers_units), "green"))

        self.action_bound_high = action_bound_high
        self.action_bound_low = action_bound_low

        self.layers_d1 = []
        self.layers_d2 = []
        self.output_layers = []

        # TODO: check if state dims and action_dims are float

        print(colorize("DENSE 1 Layer: {} | units: {}".format(0, layers_units[0]), "green"))
        self.layers_d1.append(tf.keras.layers.Dense(units=layers_units[0], 
                                                    activation=tf.nn.relu, 
                                                    kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(state_dims, tf.float32))), 1/tf.sqrt(tf.cast(state_dims, tf.float32))),
                                                    bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(state_dims, tf.float32))), 1/tf.sqrt(tf.cast(state_dims, tf.float32)))))
        
        # print(colorize("DENSE 1 Layer: {} | RELU".format(1), "green"))
        # self.layers_d1.append(tf.keras.layers.ReLU())

        print(colorize("DENSE 2 Layer: {} | units: {}".format(0, layers_units[1]), "green"))
        self.layers_d2.append(tf.keras.layers.Dense(units=layers_units[1], 
                                                    activation=tf.nn.relu, 
                                                    kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(layers_units[0], tf.float32))), 1/tf.sqrt(tf.cast(layers_units[0], tf.float32))),
                                                    bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(layers_units[0], tf.float32))), 1/tf.sqrt(tf.cast(layers_units[0], tf.float32)))))

        # print(colorize("DENSE 2 Layer: {} | RELU".format(1), "green"))
        # self.layers_d2.append(tf.keras.layers.ReLU())

        # TODO 666: change this hack
        final_layer_init = 0.003 # Initialise networks' final layer weights in range +/-final_layer_init
        print(colorize("OUTPUT Layer: {} | units: {}".format(0, action_dims), "green"))
        self.output_layers.append(tf.keras.layers.Dense(units=action_dims, 
                                                    # activation=tf.nn.relu, 
                                                    kernel_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                                    bias_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init)))                        

    def call(self, s):
        x = s
        x = self.layers_d1[0](x)
        x = self.layers_d2[0](x)

        output = self.output_layers[0](x)
        output = tf.math.tanh(output)

        # Scale tanh output to lower and upper action bounds
        output = tf.multiply(0.5, tf.multiply(output, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low))

        return output

####################################################################################################
class CRITIC_NET(tf.keras.Model):
    def __init__(self, state_dims, action_dims, layers_units = [400, 300, 51]):
        super(CRITIC_NET, self).__init__()
        
        print(colorize("CRITIC NET | dense1 size & dense2 size: {}".format(layers_units), "green"))

        self.layers_d1 = []
        self.layers_d2 = []
        self.relu_layers = []
        self.output_layers = []

        # TODO: check if state dims and action_dims are float

        print(colorize("DENSE 1 Layer: {} | units: {}".format(0, layers_units[0]), "green"))
        self.layers_d1.append(tf.keras.layers.Dense(units=layers_units[0], 
                                                    activation=tf.nn.relu, 
                                                    kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(state_dims, tf.float32))), 1/tf.sqrt(tf.cast(state_dims, tf.float32))),
                                                    bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(state_dims, tf.float32))), 1/tf.sqrt(tf.cast(state_dims, tf.float32)))))
        
        # print(colorize("DENSE 1 Layer: {} | RELU".format(1), "green"))
        # self.layers_d1.append(tf.keras.layers.ReLU())

        print(colorize("DENSE 2 Layer: {} | units: {}".format(0, layers_units[1]), "green"))
        self.layers_d2.append(tf.keras.layers.Dense(units=layers_units[1], 
                                                    # activation=tf.nn.relu, 
                                                    kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(layers_units[0]+action_dims, tf.float32))), 1/tf.sqrt(tf.cast(layers_units[0]+action_dims, tf.float32))),
                                                    bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(layers_units[0]+action_dims, tf.float32))), 1/tf.sqrt(tf.cast(layers_units[0]+action_dims, tf.float32)))))
        
        print(colorize("DENSE 2 Layer: {} | units: {}".format(1, layers_units[1]), "green"))
        self.layers_d2.append(tf.keras.layers.Dense(units=layers_units[1], 
                                                    # activation=tf.nn.relu, 
                                                    kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(layers_units[0]+action_dims, tf.float32))), 1/tf.sqrt(tf.cast(layers_units[0]+action_dims, tf.float32))),
                                                    bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(layers_units[0]+action_dims, tf.float32))), 1/tf.sqrt(tf.cast(layers_units[0]+action_dims, tf.float32)))))

        self.relu_layers.append(tf.keras.layers.ReLU())

        # TODO: change this hack
        final_layer_init = 0.003 # Initialise networks' final layer weights in range +/-final_layer_init
        print(colorize("OUTPUT Layer: {} | units: {}".format(0, layers_units[2]), "green"))
        self.output_layers.append(tf.keras.layers.Dense(units=layers_units[2], 
                                                    # activation=tf.nn.relu, 
                                                    kernel_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                                    bias_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init)))             
                   
        # TODO: change this
        v_min = -20.0 # Lower bound of critic value output distribution
        v_max = 0.0 # Upper bound of critic value output distribution (V_min and V_max should be chosen based on the range of normalised reward values in the chosen env)
        num_atoms = 51 # Number of atoms in output layer of distributional critic
        self.z_atoms = tf.linspace(v_min, v_max, num_atoms)

    def call(self, s, a, give_grads):
        x = s
        x = self.layers_d1[0](x)

        d2_1 = self.layers_d2[0](x)
        d2_2 = self.layers_d2[1](a)

        d2 = self.relu_layers[0](d2_1 + d2_2)

        output_logits = self.output_layers[0](d2)
        output_probs = tf.nn.softmax(output_logits)
        
        if give_grads:
            # TODO: NOT WORKING!!!
            with tf.GradientTape() as tape:        
                tape.watch(output_probs)
                # tape.watch(a)
            action_grads = tape.gradient(output_probs, a, self.z_atoms)  
            # action_grads = tf.gradients(output_probs, a, self.z_atoms) # gradient of mean of output Z-distribution wrt action input - used to train actor network, weighing the grads by z_values gives the mean across the output distribution
                
            return output_logits, output_probs, action_grads
        else:
            return output_logits, output_probs, None

####################################################################################################
class ACTOR_TRAINER():
    def __init__(self, initLr, batch_size):
        self.lr = tf.Variable(initLr)
        self.batch_size = batch_size

        self.optim_a = tf.keras.optimizers.Adam(learning_rate = self.lr)

    def __loss(self, output):
        loss_a = output
        return loss_a

    # @tf.function
    def __grad(self, model, state, action_grads):
        with tf.GradientTape() as tape:
            output = model(state)

            loss_value = self.__loss(output)
        return loss_value, tape.gradient(loss_value, model.trainable_variables, -action_grads)        

    def updateLr(self, lr):
        self.lr.assign(lr)

    def train_step(self, model, state, action_grads):
        loss, grads = self.__grad(model, state, action_grads)
        grads_scaled = list(map(lambda x: tf.divide(x, self.batch_size), grads)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients

        self.optim_a.apply_gradients(zip(grads_scaled, model.trainable_variables))
        return loss

####################################################################################################
class CRITIC_TRAINER():
    def __init__(self, initLr, l2_lambda):
        self.lr = tf.Variable(initLr)
        
        self.optim_c = tf.keras.optimizers.Adam(learning_rate = self.lr)

    def __loss(self, output_logits, target_Z_projected):
        loss_c = tf.nn.softmax_cross_entropy_with_logits(logits=output_logits, labels=tf.stop_gradient(target_Z_projected))
        return loss_c

    # @tf.function
    def __grad(self, model, state, action, target_Z_dist, target_Z_atoms, IS_weights, l2_lambda):
        with tf.GradientTape() as tape:
            output_logits, _, _ = model(state, action, False)            
            target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, model.z_atoms)  
            
            loss_value = self.__loss(output_logits, target_Z_projected)
            weighted_loss = loss_value * IS_weights
            mean_loss = tf.reduce_mean(weighted_loss)
            l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if 'kernel' in v.name]) * l2_lambda
            total_loss = mean_loss + l2_reg_loss

        return loss_value, total_loss, tape.gradient(total_loss, model.trainable_variables)        

    def updateLr(self, lr):
        self.lr.assign(lr)

    def train_step(self, model, state, action, target_Z_dist, target_Z_atoms, IS_weights, l2_lambda): 
        loss_value, total_loss, grads = self.__grad(model, state, action, target_Z_dist, target_Z_atoms, IS_weights, l2_lambda)

        self.optim_c.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value, total_loss

