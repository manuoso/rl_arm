import os   
import time
import numpy as np
import tensorflow as tf

from gym.utils import colorize

from rlarm.algorithms.policy_base import Policy_Base

from rlarm.algorithms.ddpg.networks import ACTOR_DDPG, CRITIC_DDPG
from rlarm.algorithms.tools.huber_loss import huber_loss

from rlarm.algorithms.tools.utils import plot_learning_curve, REPO_ROOT

from rlarm.algorithms.tools.get_replay_buffer import get_replay_buffer
from rlarm.algorithms.tools.target_update_ops import update_target_variables


tf.executing_eagerly()

####################################################################################################
class DDPG(Policy_Base):
    def __init__(self, name, env, deterministic = False, save_tensorboard = True, save_matplotlib = True, lr_actor = 0.0001, lr_critic = 0.001, max_action = 1., gamma = 0.99, actor_layers = [400, 300], critic_layers = [400, 300]):        
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
            
        Policy_Base.__init__(self, name, env, deterministic, save_tensorboard, save_matplotlib)
        
        # self.test_env = self.env                

        self.gamma = gamma
        self.max_action = max_action
        
        actor_layers += [self.act_dim]
        critic_layers += [1]
        layer_sizes = len(actor_layers)
        
        # Actor networks
        self.actor_network = ACTOR_DDPG(state_shape = tuple(self.state_dim), max_action = self.max_action, layer_sizes = layer_sizes, units = actor_layers)
        self.actor_target_network = ACTOR_DDPG(state_shape = tuple(self.state_dim), max_action = self.max_action, layer_sizes = layer_sizes, units = actor_layers)
        
        update_target_variables(self.actor_target_network.weights, self.actor_network.weights, tau=1.)

        # Critic networks
        self.critic_network = CRITIC_DDPG(state_shape = tuple(self.state_dim), action_dim = self.act_dim, layer_sizes = layer_sizes, units = critic_layers)
        self.critic_target_network = CRITIC_DDPG(state_shape = tuple(self.state_dim), action_dim = self.act_dim, layer_sizes = layer_sizes, units = critic_layers)
        
        update_target_variables(self.critic_target_network.weights, self.critic_network.weights, tau=1.)
                
        self.max_grad = 10.
        self.actor_optim = tf.keras.optimizers.Adam(learning_rate = lr_actor)
        self.critic_optim = tf.keras.optimizers.Adam(learning_rate = lr_critic)                

    # ----------------------------------------------------------------------------------------------------
    class TrainConfig():
        def __init__(self):
            self.use_prioritized_rb = True
            self.max_steps = 100000
            self.episode_max_steps = 1000
            self.n_warmup = 10000
            self.update_interval = 1
            self.test_interval = 10000
            self.test_episodes = 5
            
            self.memory_capacity = int(1e6) # 1000000
            self.batch_size = 100
            self.sigma = 0.1
            self.tau = 0.005
            
        def __str__(self):
            return str(self.__class__) + " : " + str(self.__dict__)
    
    # ----------------------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------------------
    ####################################################################################################
    # ----------------------------------------------------------------------------------------------------      
      
    # @tf.function
    def train_step(self, states, actions, next_states, rewards, dones, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)

        with tf.GradientTape() as g:
            if isinstance(actions, tf.Tensor):
                rewards = tf.expand_dims(rewards, axis=1)
                dones = tf.expand_dims(dones, 1)
                
            not_dones = 1. - dones
            target_Q = self.critic_target_network(next_states, self.actor_target_network(next_states))
            target_Q = rewards + (not_dones * self.gamma * target_Q)
            target_Q = tf.stop_gradient(target_Q)
            
            current_Q = self.critic_network(states, actions)
            
            td_errors = target_Q - current_Q   
        
            h_loss = huber_loss(td_errors, delta=self.max_grad)        
            weights_tensor = tf.convert_to_tensor(weights[None, :], dtype=tf.float32)
            
            critic_loss = tf.reduce_mean(h_loss * weights_tensor)            
            
        critic_grad = g.gradient(critic_loss, self.critic_network.trainable_variables)  
        self.critic_optim.apply_gradients(zip(critic_grad, self.critic_network.trainable_variables))
                     
        with tf.GradientTape() as gg:
            next_action = self.actor_network(states)
            out_c = self.critic_network(states, next_action)
            
            actor_loss = -tf.reduce_mean(out_c)            
            
        actor_grad = gg.gradient(actor_loss, self.actor_network.trainable_variables)
        self.actor_optim.apply_gradients(zip(actor_grad, self.actor_network.trainable_variables))
        
        return np.abs(np.ravel(td_errors.numpy())), actor_loss, critic_loss

    # ----------------------------------------------------------------------------------------------------
    ####################################################################################################
    # ----------------------------------------------------------------------------------------------------     
    
    # ----------------------------------------------------------------------------------------------------
    def act(self, state, test=False):
        is_single_state = len(state.shape) == 1
        assert isinstance(state, np.ndarray)
        
        state = np.expand_dims(state, axis=0).astype(np.float32) if is_single_state else state
        
        action = self.actor_network(state)
        action += tf.random.normal(shape=action.shape, mean=0., stddev=self.sigma*test, dtype=tf.float32)
        
        acti = tf.clip_by_value(action, -self.actor_network.max_action, self.actor_network.max_action)
        
        return acti.numpy()[0] if is_single_state else acti.numpy()

    # ----------------------------------------------------------------------------------------------------
    def fit_net(self, samples):    
        td_errors, actor_loss, critic_loss = self.train_step(samples["obs"], 
                                                             samples["act"], 
                                                             samples["next_obs"], 
                                                             samples["rew"], 
                                                             np.array(samples["done"]), 
                                                             None if not self.use_prioritized_rb else samples["weights"])
        
        # Update target networks
        update_target_variables(self.critic_target_network.weights, self.critic_network.weights, self.tau)
        update_target_variables(self.actor_target_network.weights, self.actor_network.weights, self.tau)
        
        return td_errors, actor_loss, critic_loss
    
    # ----------------------------------------------------------------------------------------------------
    def train(self, config: TrainConfig):
        self.sigma = config.sigma
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.use_prioritized_rb = config.use_prioritized_rb
        
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        
        reward_history = []
        reward_averaged = []
        
        episode_start_time = time.time()
        n_episode = 0

        replay_buffer = get_replay_buffer(obs_shape = self.state_dim, 
                                          act_shape = self.act_dim, 
                                          mem_capacity = config.memory_capacity, 
                                          use_prioritized_rb = self.use_prioritized_rb, 
                                          on_policy = False, discrete = False)

        obs = self.reset()

        while total_steps < config.max_steps:
            
            if total_steps < config.n_warmup:
                action = self.get_sample()                 
            else:
                action = self.act(obs)

            # print("Action: ", action)
            next_obs, reward, done = self.step(action)
            
            episode_steps += 1
            total_steps += 1
            
            episode_return += reward
            done_flag = done
                
            replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag)
            
            obs = next_obs

            if done or episode_steps == config.episode_max_steps:
                obs = self.reset()        

                n_episode += 1
                fps = episode_steps / (time.time() - episode_start_time)
                
                print(colorize("[episodes: {} / steps: {} / episode_steps: {}], return: {:.4f}, FPS: {:5.2f}".format(
                        n_episode, total_steps, episode_steps,
                        episode_return, fps),
                        "blue"))
                
                if self.save_tensorboard:
                    with self.writer.as_default():
                        tf.summary.scalar("training_return", episode_return, step=total_steps)

                    self.writer.flush()
                    
                reward_history.append(episode_return)
                reward_averaged.append(np.mean(reward_history[-10:]))

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.time()

            if total_steps >= config.n_warmup and total_steps % config.update_interval == 0:
                samples = replay_buffer.sample(self.batch_size)       
                         
                td_error, actor_loss, critic_loss = self.fit_net(samples)
                if self.save_tensorboard:
                    with self.writer.as_default():
                        # tf.summary.scalar("td_error", td_error, step=total_steps)
                        tf.summary.scalar("actor_loss", actor_loss, step=total_steps)
                        tf.summary.scalar("critic_loss", critic_loss, step=total_steps)                        
                        
                    self.writer.flush()
                        
                if self.use_prioritized_rb:
                    replay_buffer.update_priorities(samples["indexes"], np.abs(td_error) + 1e-6)
                
                # if total_steps % config.test_interval == 0:                
                #     avg_test_return = self._evaluate_policy(total_steps, config.test_episodes, config.episode_max_steps)
                    
                #     print(colorize("Evaluation total steps: {}, Average Reward: {:5.4f} over {} episodes".format(
                #         total_steps, avg_test_return, config.test_episodes),
                #         "cyan"))
                        
                #     if self.save_tensorboard:
                #         with self.writer.as_default():
                #             tf.summary.scalar("average_test_return", avg_test_return, step=total_steps)

                #         self.writer.flush()
        
        print(colorize("[FINAL] Num steps: {}, Max reward: {}, Average reward: {}".format(
            total_steps, np.max(reward_history), np.mean(reward_history)), "magenta"))

        if self.save_tensorboard:
            self.writer.close()

        if self.save_matplotlib:
            data_dict = {
                'reward': reward_history,
                'reward_smooth10': reward_averaged,
            }
            plot_learning_curve(self.name, data_dict, xlabel='episode')

    # ----------------------------------------------------------------------------------------------------
    def _evaluate_policy(self, total_steps, test_episodes, episode_max_steps):        
        avg_test_return = 0.
        
        for i in range(test_episodes):
            episode_return = 0.
            done = False

            obs = self.test_env.reset()
            for _ in range(episode_max_steps):
                action = self.act(obs, test=True)
                
                next_obs, reward, done, _ = self.test_env.step(action)
                # self.test_env.render()

                episode_return += reward
                obs = next_obs
                
                if done:
                    break
                
            avg_test_return += episode_return

        return avg_test_return / test_episodes
        
