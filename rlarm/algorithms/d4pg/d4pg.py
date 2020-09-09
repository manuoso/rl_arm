import os   
import numpy as np
import tensorflow as tf

from gym.spaces import Box
from gym.utils import colorize

from rlarm.algorithms.policy_base import Policy_Base, TypeExcept

from rlarm.algorithms.d4pg.networks import ACTOR_NET, CRITIC_NET

from collections import deque
from rlarm.algorithms.d4pg.prioritised_experience_replay import PrioritizedReplayBuffer
from rlarm.algorithms.d4pg.gaussian_noise import GaussianNoiseGenerator
from rlarm.algorithms.d4pg.l2_projection import _l2_project

from rlarm.algorithms.tools.utils import plot_learning_curve, REPO_ROOT


####################################################################################################
class D4PG(Policy_Base):
    def __init__(self, name, env, dir_checkpoints, save_tensorboard = True, save_matplotlib = True, deterministic = False, lr_actor = 0.0001, lr_critic = 0.001, layers_units = [400, 300]):
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
            
        Policy_Base.__init__(self, name, env, deterministic, save_tensorboard, save_matplotlib)     

        self.action_space_high = 1.0
        self.action_space_low = -1.0
        
        self.actor_network = ACTOR_NET(self.action_space_high, self.action_space_low, self.state_dim, self.act_dim, layers_units = layers_units)      
        self.target_actor_network = ACTOR_NET(self.action_space_high, self.action_space_low, self.state_dim, self.act_dim, layers_units = layers_units)      
        
        self.critic_network = CRITIC_NET(self.state_dim, self.act_dim, layers_units = layers_units + [51])    
        self.target_critic_network = CRITIC_NET(self.state_dim, self.act_dim, layers_units = layers_units + [51])            
        
        self.optim_a = tf.keras.optimizers.Adam(learning_rate = lr_actor)
        self.optim_c = tf.keras.optimizers.Adam(learning_rate = lr_critic)
        
        self.init_target_net()
        
        self.dir_checkpoints = dir_checkpoints
        if self.dir_checkpoints is None:
            policy_dir = os.path.join(REPO_ROOT, 'checkpoints', self.name)
            
            self._dir_actor = os.path.join(policy_dir, 'actor')
            self._dir_actor_target = os.path.join(policy_dir, 'actor_target')
            self._dir_critic = os.path.join(policy_dir, 'critic')
            self._dir_critic_target = os.path.join(policy_dir, 'critic_target')
            
            os.makedirs(os.path.join(REPO_ROOT, 'checkpoints'), exist_ok = True)
            os.makedirs(policy_dir, exist_ok = True)
            
            os.makedirs(self._dir_actor, exist_ok = True)
            os.makedirs(self._dir_actor_target, exist_ok = True)
            os.makedirs(self._dir_critic, exist_ok = True)
            os.makedirs(self._dir_critic_target, exist_ok = True)
            
            self.set_check_point()
        else:
            self._dir_actor = os.path.join(self.dir_checkpoints, 'actor')
            self._dir_actor_target = os.path.join(self.dir_checkpoints, 'actor_target')
            self._dir_critic = os.path.join(self.dir_checkpoints, 'critic')
            self._dir_critic_target = os.path.join(self.dir_checkpoints, 'critic_target')
            
            self.set_check_point()
            self.load()

    # ----------------------------------------------------------------------------------------------------
    class TrainConfig():
        def __init__(self):
            self.critic_l2_lambda = 0.0     # Coefficient for L2 weight regularisation in critic - if 0, no regularisation is performed
            
            self.n_episodes = 1000
            self.batch_size = 64
            self.log_every_episode = 10
            self.save_model_interval = 10000 # save checkpoints every X steps
            
            self.priority_alpha = 0.6       # Controls the randomness vs prioritisation of the prioritised sampling (0.0 = Uniform sampling, 1.0 = Greedy prioritisation)
            self.priority_beta_start = 0.4  # Starting value of beta - controls to what degree IS weights influence the gradient updates to correct for the bias introduced by priority sampling (0 - no correction, 1 - full correction)
            self.priority_beta_end = 1.0    # Beta will be linearly annealed from its start value to this value throughout training
            self.priority_eps = 0.00001     # Small value to be added to updated priorities to ensure no sample has a probability of 0 of being chosen
            self.noise_scale = 0.3          # Scaling to apply to Gaussian noise
            self.noise_decay = 0.9999       # Decay noise throughout training by scaling by noise_decay**training_step
            self.discount_rate = 0.99       # Discount rate (gamma) for future rewards
            self.n_step_returns = 5         # Number of future steps to collect experiences for N-step returns
            
            # For target network polyak averaging
            self.tau = 0.001
    
        def __str__(self):
            return str(self.__class__) + " : " + str(self.__dict__)

    # ----------------------------------------------------------------------------------------------------
    ####################################################################################################
    # ----------------------------------------------------------------------------------------------------      
      
    # @tf.function
    def train_step(self, batch_size, states, actions, next_states, rewards, terminals, weights, gamma, l2_lambda):
        with tf.GradientTape() as g:
            # Critic training step   
            # Predict actions for next states by passing next states through policy target network
            next_states = tf.reshape(next_states, (batch_size, self.state_dim[0]))
            future_action = self.actor_network(next_states)

            # Predict future Z distribution by passing next states and actions through value target network, also get target network's Z-atom values
            _, target_Z_dist, _ = self.critic_network(next_states, future_action, False)
            target_Z_atoms = self.critic_network.z_atoms

            # Create batch of target network's Z-atoms
            target_Z_atoms = np.repeat(np.expand_dims(target_Z_atoms, axis=0), batch_size, axis=0)
            # Value of terminal states is 0 by definition
            target_Z_atoms[terminals, :] = 0.0
            # Apply Bellman update to each atom
            target_Z_atoms = np.expand_dims(rewards, axis=1) + (target_Z_atoms*np.expand_dims(gamma, axis=1))

            output_logits, _, _ = self.critic_network(states, actions, False)            
            target_Z_projected = _l2_project(np.float32(target_Z_atoms), target_Z_dist, self.critic_network.z_atoms)  
            
            loss_c = tf.nn.softmax_cross_entropy_with_logits(logits=output_logits, labels=tf.stop_gradient(target_Z_projected))

            weighted_loss = loss_c * np.float32(weights)
            mean_loss = tf.reduce_mean(weighted_loss)
            l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.critic_network.trainable_variables if 'kernel' in v.name]) * l2_lambda
            total_loss = mean_loss + l2_reg_loss

        critic_grads = g.gradient(total_loss, self.critic_network.trainable_variables)
        self.optim_c.apply_gradients(zip(critic_grads, self.critic_network.trainable_variables))

        with tf.GradientTape() as gg:
            # Actor training step
            # Get policy network's action outputs for selected states
            states = tf.reshape(states, (batch_size, self.state_dim[0]))
            actions_a = self.actor_network(states)
            
            # Compute gradients of critic's value output distribution wrt actions
            _, _, action_grads =  self.critic_network(states, actions_a, True)

            loss_a = self.actor_network(states)

        actor_grads = gg.gradient(loss_a, self.actor_network.trainable_variables, -action_grads)  
        grads_scaled = list(map(lambda x: tf.divide(x, batch_size), actor_grads)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradientss
        self.optim_a.apply_gradients(zip(grads_scaled, self.actor_network.trainable_variables))

        return loss_c

    # ----------------------------------------------------------------------------------------------------
    ####################################################################################################
    # ---------------------------------------------------------------------------------------------------- 

    # ----------------------------------------------------------------------------------------------------
    def act(self, state, noise_generator, n_eps, decay = 0.9999):
        state = tf.reshape(state, (1, self.state_dim[0]))
        # state = np.expand_dims(state, 0) # Add batch dimension to single state input, and remove batch dimension from single action output    
        action = self.actor_network(state)
        
        act = action.numpy()[0]
        act += (noise_generator() * decay**n_eps)
        
        return act

    # ----------------------------------------------------------------------------------------------------
    def init_target_net(self):
        network_params = self.actor_network.trainable_variables + self.critic_network.trainable_variables
        target_network_params = self.target_actor_network.trainable_variables + self.target_critic_network.trainable_variables
        
        for q_t_v, q_v in zip(target_network_params, network_params):
            q_t_v.assign(q_v)

    # ----------------------------------------------------------------------------------------------------
    def update_target_net(self, tau = 0.01):
        network_params = self.actor_network.trainable_variables + self.critic_network.trainable_variables
        target_network_params = self.target_actor_network.trainable_variables + self.target_critic_network.trainable_variables
        
        for q_t_v, q_v in zip(target_network_params, network_params):
            q_t_v.assign(((1.0 - tau) * q_t_v) + (tau * q_v))

    # ----------------------------------------------------------------------------------------------------
    def fit_net(self, tau, memory, batch_size, priority_beta, priority_eps, l2_lambda):
        batch = memory.sample(batch_size, priority_beta)  
        states_batch = batch[0]
        actions_batch = batch[1]
        rewards_batch = batch[2]
        next_states_batch = batch[3]
        terminals_batch = batch[4]
        gammas_batch = batch[5]
        weights_batch = batch[6]
        idx_batch = batch[7]         

        TD_error = self.train_step(batch_size, 
                                    states_batch, 
                                    actions_batch, 
                                    next_states_batch, 
                                    rewards_batch, 
                                    terminals_batch, 
                                    weights_batch, 
                                    gammas_batch, 
                                    l2_lambda)

        # Use critic TD errors to update sample priorities
        memory.update_priorities(idx_batch, (np.abs(TD_error)+priority_eps))

        # Update target networks
        self.update_target_net(tau)  
              
    # ----------------------------------------------------------------------------------------------------
    def train(self, config: TrainConfig):        
        replay_mem_size = 1000000 # Soft maximum capacity of replay memory
        prior_rep_buffer = PrioritizedReplayBuffer(replay_mem_size, config.priority_alpha)
        
        noise_generator = GaussianNoiseGenerator(self.act_dim, self.action_space_low, self.action_space_high, config.noise_scale)
        
        # Initialise deque buffer to store experiences for N-step returns
        buffer = deque()
        
        reward_history = []
        reward_averaged = []

        n_all_steps = 0
        
        self.save_model_interval = config.save_model_interval
        
        # Initialise beta
        priority_beta = config.priority_beta_start
        beta_increment = (config.priority_beta_end - config.priority_beta_start) / config.n_episodes

        for n_episode in range(config.n_episodes):
            ob = self.reset()
            
            done = False
            episode_reward = 0.
            buffer.clear()

            while not done:
                a = self.act(ob, noise_generator, n_all_steps, config.noise_decay)

                ob_next, r, done = self.step(a)
                
                # Convert from numpy array to float
                # r = r[0]

                n_all_steps += 1
                episode_reward += r

                buffer.append((ob, a, r))
                
                # We need at least N steps in the experience buffer before we can compute Bellman rewards and add an N-step experience to replay memory
                if len(buffer) >= config.n_step_returns:
                    state_0, action_0, reward_0 = buffer.popleft()
                    discounted_reward = reward_0
                    gamma = config.discount_rate
                    for (_, _, r_i) in buffer:
                        discounted_reward += r_i * gamma
                        gamma *= config.discount_rate
                
                    prior_rep_buffer.add(state_0, action_0, discounted_reward, ob_next, done, gamma)
                
                ob = ob_next
                
                if len(prior_rep_buffer) >= config.batch_size:
                    self.fit_net(config.tau, prior_rep_buffer, config.batch_size, priority_beta, config.priority_eps, config.critic_l2_lambda)
                
                # Increment beta value at end of every step   
                priority_beta += beta_increment

            # One trajectory/Episode is complete!
            reward_history.append(episode_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))

            if self.save_tensorboard:
                with self.writer.as_default():
                    tf.summary.scalar("reward_history", episode_reward, step=n_episode)
                    tf.summary.scalar("reward_averaged", reward_averaged[n_episode], step=n_episode)
                    if n_epochs > config.n_warmup and self.dir_checkpoints is None:
                        tf.summary.scalar("training_reward", episode_reward, step=total_steps-config.n_warmup)

                    self.writer.flush()

            if (reward_history and config.log_every_episode and n_episode % config.log_every_episode == 0):
                print(colorize("[episodes:{}/steps:{}], best:{}, avg:{:.2f}:{}".format(
                        n_episode, n_all_steps, np.max(reward_history),
                        np.mean(reward_history[-10:]), reward_history[-5:]),
                        "blue"))
                
            if n_all_steps % self.save_model_interval == 0:
                self.checkpoint_manager_actor.save()
                self.checkpoint_manager_actor_target.save()
                
                self.checkpoint_manager_critic.save()
                self.checkpoint_manager_critic_target.save()

        print(colorize("[FINAL] Num episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)), "magenta"))

        if self.save_tensorboard:
            self.writer.close()

        if self.save_matplotlib:
            data_dict = {
                'reward': reward_history,
                'reward_smooth10': reward_averaged,
            }
            plot_learning_curve(self.name, data_dict, xlabel='episode')
            
    # ----------------------------------------------------------------------------------------------------
    def set_check_point(self):        
        self._checkpoint_actor = tf.train.Checkpoint(net=self.actor_network)
        self._checkpoint_actor_target = tf.train.Checkpoint(net=self.target_actor_network)
        
        self._checkpoint_critic = tf.train.Checkpoint(net=self.critic_network)
        self._checkpoint_critic_target = tf.train.Checkpoint(net=self.target_critic_network)

        self.checkpoint_manager_actor = tf.train.CheckpointManager(
            self._checkpoint_actor, directory=self._dir_actor, max_to_keep=5)
        self.checkpoint_manager_actor_target = tf.train.CheckpointManager(
            self._checkpoint_actor_target, directory=self._dir_actor_target, max_to_keep=5)
        
        self.checkpoint_manager_critic = tf.train.CheckpointManager(
            self._checkpoint_critic, directory=self._dir_critic, max_to_keep=5)
        self.checkpoint_manager_critic_target = tf.train.CheckpointManager(
            self._checkpoint_critic_target, directory=self._dir_critic_target, max_to_keep=5)
    
    # ----------------------------------------------------------------------------------------------------
    def load(self):
        last_checkpoint_actor = tf.train.latest_checkpoint(self._dir_actor)        
        last_checkpoint_actor_target = tf.train.latest_checkpoint(self._dir_actor_target)
        
        last_checkpoint_critic = tf.train.latest_checkpoint(self._dir_critic)
        last_checkpoint_critic_target = tf.train.latest_checkpoint(self._dir_critic_target)

        if (last_checkpoint_actor is None) or (last_checkpoint_actor_target is None) or (last_checkpoint_critic is None) or (last_checkpoint_critic_target is None):
            raise TypeExcept("No checkpoint found")   
        else:
            self._checkpoint_actor.restore(last_checkpoint_actor)
            self._checkpoint_actor_target.restore(last_checkpoint_actor_target)
            self._checkpoint_critic.restore(last_checkpoint_critic)
            self._checkpoint_critic_target.restore(last_checkpoint_critic_target)
