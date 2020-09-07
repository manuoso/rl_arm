import os   
import time
import numpy as np
import tensorflow as tf

from gym.utils import colorize

from rlarm.algorithms.policy_base import Policy_Base, TypeExcept

from rlarm.algorithms.sac.networks import CRITIC_V, CRITIC_Q
from rlarm.algorithms.tools.huber_loss import huber_loss

from rlarm.algorithms.sac.gaussian_actor import GaussianActor

from rlarm.algorithms.tools.utils import plot_learning_curve, REPO_ROOT

from rlarm.algorithms.tools.get_replay_buffer import get_replay_buffer
from rlarm.algorithms.tools.target_update_ops import update_target_variables


tf.executing_eagerly()

####################################################################################################
class SAC(Policy_Base):
    def __init__(self, name, env, dir_checkpoints, deterministic = False, save_tensorboard = True, save_matplotlib = True, lr = 3e-4, max_action = 1., discount = 0.99, alpha = .2, auto_alpha = False, actor_layers = [400, 300], critic_layers = [400, 300]):        
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
            
        Policy_Base.__init__(self, name, env, deterministic, save_tensorboard, save_matplotlib)
        
        self.max_grad = 10.
        self.discount = discount
        
        self.actor = GaussianActor(self.state_dim, self.act_dim, max_action, squash=True, units=actor_layers)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        self.vf = CRITIC_V(self.state_dim, critic_layers)
        self.vf_target = CRITIC_V(self.state_dim, critic_layers)
        
        update_target_variables(self.vf_target.weights, self.vf.weights, tau=1.)
        
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        self.qf1 = CRITIC_Q(self.state_dim, self.act_dim, critic_layers)
        self.qf2 = CRITIC_Q(self.state_dim, self.act_dim, critic_layers)
        
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # Set hyper-parameters
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.log_alpha = tf.Variable(0., dtype=tf.float32)
            self.alpha = tf.Variable(0., dtype=tf.float32)
            self.alpha.assign(tf.exp(self.log_alpha))
            self.target_alpha = -self.act_dim
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            self.alpha = alpha
        
        self.dir_checkpoints = dir_checkpoints
        if self.dir_checkpoints is None:
            policy_dir = os.path.join(REPO_ROOT, 'checkpoints', self.name)
            
            self._dir_actor = os.path.join(policy_dir, 'actor')
            
            self._dir_critic_vf = os.path.join(policy_dir, 'critic_vf')
            self._dir_critic_vf_target = os.path.join(policy_dir, 'critic_vf_target')
            
            self._dir_critic_qf1 = os.path.join(policy_dir, 'critic_qf1')
            self._dir_critic_qf2 = os.path.join(policy_dir, 'critic_qf2')
                 
            os.makedirs(os.path.join(REPO_ROOT, 'checkpoints'), exist_ok = True)
            os.makedirs(policy_dir, exist_ok = True)
            
            os.makedirs(self._dir_actor, exist_ok = True)
            os.makedirs(self._dir_critic_vf, exist_ok = True)
            os.makedirs(self._dir_critic_vf_target, exist_ok = True)
            os.makedirs(self._dir_critic_qf1, exist_ok = True)
            os.makedirs(self._dir_critic_qf2, exist_ok = True)
            
            self.set_check_point()
        else:
            self._dir_actor = os.path.join(self.dir_checkpoints, 'actor')
            self._dir_critic_vf = os.path.join(self.dir_checkpoints, 'critic_vf')
            self._dir_critic_vf_target = os.path.join(self.dir_checkpoints, 'critic_vf_target')
            self._dir_critic_qf1 = os.path.join(self.dir_checkpoints, 'critic_qf1')
            self._dir_critic_qf2 = os.path.join(self.dir_checkpoints, 'critic_qf2')
            
            self.set_check_point()
            self.load()
                
    # ----------------------------------------------------------------------------------------------------
    class TrainConfig():
        def __init__(self):
            self.use_prioritized_rb = True
            self.max_epochs = 1000
            self.episode_max_steps = 1000
            self.n_warmup = 10000
            self.update_interval = 1
            self.test_interval = 10000
            self.test_episodes = 5
            self.save_model_interval = 10000 # save checkpoints every X steps
            
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
            
        if tf.rank(rewards) == 2:
            rewards = tf.squeeze(rewards, axis=1)
        not_dones = 1. - tf.cast(dones, dtype=tf.float32)

        with tf.GradientTape(persistent = True) as tape:
            # Compute loss of critic Q
            current_q1 = self.qf1([states, actions])
            current_q2 = self.qf2([states, actions])
            vf_next_target = self.vf_target(next_states)

            target_q = tf.stop_gradient(
                rewards + not_dones * self.discount * vf_next_target)

            td_loss_q1 = tf.reduce_mean(huber_loss(
                target_q - current_q1, delta=self.max_grad) * weights)
            td_loss_q2 = tf.reduce_mean(huber_loss(
                target_q - current_q2, delta=self.max_grad) * weights)  # Eq.(7)

            # Compute loss of critic V
            current_v = self.vf(states)

            sample_actions, logp, _ = self.actor(states)  # Resample actions to update V
            current_q1 = self.qf1([states, sample_actions])
            current_q2 = self.qf2([states, sample_actions])
            current_min_q = tf.minimum(current_q1, current_q2)

            target_v = tf.stop_gradient(
                current_min_q - self.alpha * logp)
            td_errors = target_v - current_v
            td_loss_v = tf.reduce_mean(
                huber_loss(td_errors, delta=self.max_grad) * weights)  # Eq.(5)

            td_errors_q1 = target_q - current_q1

            # Compute loss of policy
            policy_loss = tf.reduce_mean(
                (self.alpha * logp - current_min_q) * weights)  # Eq.(12)

            # Compute loss of temperature parameter for entropy
            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(
                    (self.log_alpha * tf.stop_gradient(logp + self.target_alpha)))

        q1_grad = tape.gradient(td_loss_q1, self.qf1.trainable_variables)
        self.qf1_optimizer.apply_gradients(
            zip(q1_grad, self.qf1.trainable_variables))
        q2_grad = tape.gradient(td_loss_q2, self.qf2.trainable_variables)
        self.qf2_optimizer.apply_gradients(
            zip(q2_grad, self.qf2.trainable_variables))

        vf_grad = tape.gradient(td_loss_v, self.vf.trainable_variables)
        self.vf_optimizer.apply_gradients(
            zip(vf_grad, self.vf.trainable_variables))
        update_target_variables(
            self.vf_target.weights, self.vf.weights, self.tau)

        actor_grad = tape.gradient(
            policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables))

        if self.auto_alpha:
            alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(
                zip(alpha_grad, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))

        del tape
    
        return td_errors_q1, policy_loss, td_loss_v, td_loss_q1, tf.reduce_min(logp), tf.reduce_max(logp), tf.reduce_mean(logp)

    # ----------------------------------------------------------------------------------------------------
    ####################################################################################################
    # ----------------------------------------------------------------------------------------------------     
    
    # ----------------------------------------------------------------------------------------------------
    def act(self, state, test = False):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == 1

        state = np.expand_dims(state, axis=0).astype(np.float32) if is_single_state else state
        action = self.actor(state, test)[0]

        return action.numpy()[0] if is_single_state else action.numpy()

    # ----------------------------------------------------------------------------------------------------
    def fit_net(self, samples):    
        td_errors, actor_loss, vf_loss, qf_loss, logp_min, logp_max, logp_mean = self.train_step(samples["obs"], 
                                                             samples["act"], 
                                                             samples["next_obs"], 
                                                             samples["rew"], 
                                                             np.array(samples["done"]), 
                                                             None if not self.use_prioritized_rb else samples["weights"])
        
        return td_errors, actor_loss, vf_loss, qf_loss, logp_min, logp_max, logp_mean
    
    # ----------------------------------------------------------------------------------------------------
    def train(self, config: TrainConfig):
        self.sigma = config.sigma
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.use_prioritized_rb = config.use_prioritized_rb
        
        self.save_model_interval = config.save_model_interval
        
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        n_epochs = 0
        
        reward_history = []
        reward_averaged = []
        
        episode_start_time = time.time()

        replay_buffer = get_replay_buffer(obs_shape = self.state_dim, 
                                          act_shape = self.act_dim, 
                                          mem_capacity = config.memory_capacity, 
                                          use_prioritized_rb = self.use_prioritized_rb, 
                                          on_policy = False, discrete = False)

        obs = self.reset()

        while n_epochs < config.max_epochs:
            
            if self.dir_checkpoints is None:
                if n_epochs < config.n_warmup:
                    action = self.get_sample()                 
                else:
                    action = self.act(obs)
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

                n_epochs += 1
                fps = episode_steps / (time.time() - episode_start_time)
                
                print(colorize("[episodes: {} / steps: {} / episode_steps: {}], return: {:.4f}, FPS: {:5.2f}".format(
                        n_epochs, total_steps, episode_steps,
                        episode_return, fps),
                        "blue"))
                
                if self.save_tensorboard:
                    with self.writer.as_default():
                        tf.summary.scalar("return", episode_return, step=total_steps)
                        if n_epochs > config.n_warmup and self.dir_checkpoints is None:
                            tf.summary.scalar("training_return", episode_return, step=total_steps-config.n_warmup)

                    self.writer.flush()
                    
                reward_history.append(episode_return)
                reward_averaged.append(np.mean(reward_history[-10:]))

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.time()

            if n_epochs >= config.n_warmup and total_steps % config.update_interval == 0:
                samples = replay_buffer.sample(self.batch_size)       
                         
                td_errors, actor_loss, vf_loss, qf_loss, logp_min, logp_max, logp_mean = self.fit_net(samples)
                if self.save_tensorboard:
                    with self.writer.as_default():        
                        # tf.summary.scalar("td_error", td_error, step=total_steps)
                        tf.summary.scalar("actor_loss", actor_loss, step=total_steps)
                        tf.summary.scalar("critic_V_loss", vf_loss, step=total_steps)
                        tf.summary.scalar("critic_Q_loss", qf_loss, step=total_steps)
                        tf.summary.scalar("logp_min", logp_min, step=total_steps)
                        tf.summary.scalar("logp_max", logp_max, step=total_steps)
                        tf.summary.scalar("logp_mean", logp_mean, step=total_steps) 
                        
                        if self.auto_alpha:
                            tf.summary.scalar("log_ent", self.log_alpha, step=total_steps)
                            tf.summary.scalar("logp_mean_target", logp_mean+self.target_alpha, step=total_steps)   
                            
                        tf.summary.scalar("ent", self.alpha, step=total_steps)                                        
                    self.writer.flush()
                        
                if self.use_prioritized_rb:
                    replay_buffer.update_priorities(samples["indexes"], np.abs(td_errors) + 1e-6)
                
                # if total_steps % config.test_interval == 0:                
                #     avg_test_return = self._evaluate_policy(total_steps, config.test_episodes, config.episode_max_steps)
                    
                #     print(colorize("Evaluation total steps: {}, Average Reward: {:5.4f} over {} episodes".format(
                #         total_steps, avg_test_return, config.test_episodes),
                #         "cyan"))
                        
                #     if self.save_tensorboard:
                #         with self.writer.as_default():
                #             tf.summary.scalar("average_test_return", avg_test_return, step=total_steps)

                #         self.writer.flush()
                
            if total_steps % self.save_model_interval == 0:
                self.checkpoint_manager_actor.save()
                
                self.checkpoint_manager_critic_vf.save()
                self.checkpoint_manager_critic_vf_target.save()
                
                self.checkpoint_manager_critic_qf1.save()
                self.checkpoint_manager_critic_qf2.save()
        
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
    def set_check_point(self):
        self._checkpoint_actor = tf.train.Checkpoint(net=self.actor)
            
        self._checkpoint_critic_vf = tf.train.Checkpoint(net=self.vf)
        self._checkpoint_critic_vf_target = tf.train.Checkpoint(net=self.vf_target)
        
        self._checkpoint_critic_qf1 = tf.train.Checkpoint(net=self.qf1)
        self._checkpoint_critic_qf2 = tf.train.Checkpoint(net=self.qf2)

        self.checkpoint_manager_actor = tf.train.CheckpointManager(
            self._checkpoint_actor, directory=self._dir_actor, max_to_keep=5)
        
        self.checkpoint_manager_critic_vf = tf.train.CheckpointManager(
            self._checkpoint_critic_vf, directory=self._dir_critic_vf, max_to_keep=5)
        self.checkpoint_manager_critic_vf_target = tf.train.CheckpointManager(
            self._checkpoint_critic_vf_target, directory=self._dir_critic_vf_target, max_to_keep=5)
        
        self.checkpoint_manager_critic_qf1 = tf.train.CheckpointManager(
            self._checkpoint_critic_qf1, directory=self._dir_critic_qf1, max_to_keep=5)
        self.checkpoint_manager_critic_qf2 = tf.train.CheckpointManager(
            self._checkpoint_critic_qf2, directory=self._dir_critic_qf2, max_to_keep=5)
    
    # ----------------------------------------------------------------------------------------------------
    def load(self):
        last_checkpoint_actor = tf.train.latest_checkpoint(self._dir_actor)   
             
        last_checkpoint_critic_vf = tf.train.latest_checkpoint(self._dir_critic_vf)
        last_checkpoint_critic_vf_target = tf.train.latest_checkpoint(self._dir_critic_vf_target)
        
        last_checkpoint_critic_qf1 = tf.train.latest_checkpoint(self._dir_critic_qf1)
        last_checkpoint_critic_qf2 = tf.train.latest_checkpoint(self._dir_critic_qf2)

        if (last_checkpoint_actor is None) or (last_checkpoint_critic_vf is None) or (last_checkpoint_critic_vf_target is None) or (last_checkpoint_critic_qf1 is None) or (last_checkpoint_critic_qf2 is None):
            raise TypeExcept("No checkpoint found")   
        else:
            self._checkpoint_actor.restore(last_checkpoint_actor)
            self._checkpoint_critic_vf.restore(last_checkpoint_critic_vf)
            self._checkpoint_critic_vf_target.restore(last_checkpoint_critic_vf_target)
            self._checkpoint_critic_qf1.restore(last_checkpoint_critic_qf1)
            self._checkpoint_critic_qf2.restore(last_checkpoint_critic_qf2)
    
    # ----------------------------------------------------------------------------------------------------
    def evaluate(self, episode_max_steps, sigma):
        self.sigma = sigma  # 666 change this
        
        step = 0
        episode_return = 0.0
        
        # Initial state
        obs = self.initialPose()
        # obs = self.reset()
        
        while True:
            action = self.act(obs, True)
            # print("Action: ", action)
            # print("Step: ", step)
            
            next_obs, reward, done = self.step(action)
            
            step += 1
            episode_return += reward
            
            obs = next_obs
                
            # Episode can finish either by reaching terminal state or max episode steps
            if done or step == episode_max_steps:
                break
    
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
