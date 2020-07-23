import time
import os

from rlarm.algorithms.tools.utils import REPO_ROOT

from rlarm.envs.envPyglet import ArmEnv
import calvopy

from rlarm.algorithms.ddpg.ddpg import DDPG
from rlarm.algorithms.bi_res_ddpg.bi_res_ddpg import BI_RES_DDPG
from rlarm.algorithms.d4pg.d4pg import D4PG
from rlarm.algorithms.sac.sac import SAC


####################################################################################################
if __name__ == '__main__':
    # Variables for Debug
    save_file_train_params = True
    deterministic = True
    save_tensorboard = True
    save_matplotlib = True
    
    ########## NAME ##########
    # Alg name
    alg_name = "BIDDPG"
    
    # Env name
    # env_name = "PygletArm2D"
    env_name = "ArmiPy"
    
    # Time name
    time_name = str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min)+'_'+str(time.localtime().tm_sec)
    
    name = alg_name + "-" + env_name + "-" + time_name
    
    ########## ENV CREATION ##########      
    # env = ArmEnv()
    # env.render()    # Need two render before begin
    # env.render()
    
    env = calvopy.ArmiPy("Default")
    env.init(0.1, True)
 
    ########## POLICY ##########
    # Create Policy
    # policy = DDPG(name = name, env = env, dir_checkpoints = None,
    #               deterministic = deterministic, save_tensorboard = save_tensorboard, save_matplotlib = save_matplotlib, 
    #               lr_actor = 0.001, lr_critic = 0.001, max_action = 1., gamma = 0.9, actor_layers = [400, 300], critic_layers = [400, 300])
    
    # train_params = policy.TrainConfig()    
    # train_params.use_prioritized_rb = False
    # train_params.max_epochs = 2000
    # train_params.episode_max_steps = 500
    # train_params.n_warmup = 500
    # train_params.update_interval = 1
    # train_params.test_interval = 1000
    # train_params.test_episodes = 5
    
    # train_params.memory_capacity = 1000000 # 1000000
    # train_params.batch_size = 64
    # train_params.sigma = 0.1
    # train_params.tau = 0.01
    
    # train_params.save_model_interval = 10000 # save checkpoints every X steps
    
    policy = BI_RES_DDPG(name = name, env = env, dir_checkpoints = None,
                         deterministic = deterministic, save_tensorboard = save_tensorboard, save_matplotlib = save_matplotlib, 
                         lr_actor = 0.001, lr_critic = 0.001, gamma = 0.99, eta = 0.05, actor_layers = [400, 300], critic_layers = [400, 300])
    
    train_params = policy.TrainConfig()    
    train_params.use_prioritized_rb = False
    train_params.max_epochs = 1000
    train_params.episode_max_steps = 500
    train_params.n_warmup = 300
    train_params.update_interval = 1
    train_params.test_interval = 1000
    train_params.test_episodes = 5
    
    train_params.memory_capacity = 30000 # 1000000
    train_params.batch_size = 64
    train_params.sigma = 0.1
    train_params.tau = 0.01
    
    train_params.save_model_interval = 10000 # save checkpoints every X steps
    
    # -------------------- NOT WORKING D4PG --------------------
    # policy = D4PG(name = name, env = env, dir_checkpoints = None,
    #               deterministic = deterministic, save_tensorboard = save_tensorboard, save_matplotlib = save_matplotlib, 
    #               layers_units = [400, 300])
    
    # train_params = policy.TrainConfig()
    # train_params.lr_a = 0.0001
    # train_params.lr_c = 0.0001
    # train_params.critic_l2_lambda = 0.0    
    
    # train_params.n_episodes = 1000
    # train_params.batch_size = 64
    # train_params.log_every_episode = 10
    # train_params.save_model_interval = 10000 # save checkpoints every X steps
    
    # train_params.priority_alpha = 0.6       
    # train_params.priority_beta_start = 0.4   
    # train_params.priority_beta_end = 1.0     
    # train_params.priority_eps = 0.00001     
    # train_params.noise_scale = 0.3          
    # train_params.discount_rate = 0.99       
    # train_params.n_step_returns = 5         
    # train_params.tau = 0.001
    
    # policy = SAC(name = name, env = env, dir_checkpoints = None, 
    #              deterministic = deterministic, save_tensorboard = save_tensorboard, save_matplotlib = save_matplotlib, 
    #              lr = 3e-4, max_action = 1., discount = 0.99, alpha = .2, auto_alpha = False, actor_layers = [400, 300], critic_layers = [400, 300])
    
    # train_params = policy.TrainConfig()    
    # train_params.use_prioritized_rb = False
    # train_params.max_epochs = 2000
    # train_params.episode_max_steps = 500
    # train_params.n_warmup = 500
    # train_params.update_interval = 1
    # train_params.test_interval = 1000
    # train_params.test_episodes = 5
    
    # train_params.memory_capacity = 1000000 # 1000000
    # train_params.batch_size = 64
    # train_params.sigma = 0.1
    # train_params.tau = 0.01
    
    # train_params.save_model_interval = 10000 # save checkpoints every X steps
    
    print('\n--------------------------------------------------')
    print('Loaded env:', name)
    print('Loaded policy:', policy.__class__)
    print('Train params:', train_params)
    print('--------------------------------------------------\n')
    
    # Save train params
    if save_file_train_params:
        os.makedirs(os.path.join(REPO_ROOT, 'train_params'), exist_ok = True)
        file_train_params = open('rlarm/train_params/' + name + ".txt", 'w')
        file_train_params.write(str(train_params))
        file_train_params.close()

    print('Start learning')
    policy.train(train_params)

    # env.close()
    print('Training completed')
