import time
import os

from rlarm.algorithms.tools.utils import REPO_ROOT

from rlarm.envs.envPyglet import ArmEnv
import calvopy

from rlarm.algorithms.ddpg.ddpg import DDPG
from rlarm.algorithms.bi_res_ddpg.bi_res_ddpg import BI_RES_DDPG


####################################################################################################
if __name__ == '__main__':
    # Variables for Debug
    save_file_train_params = True
    deterministic = True
    save_tensorboard = True
    save_matplotlib = True
    
    ########## NAME ##########
    # Alg name
    alg_name = "DDPG"
    
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
    policy = DDPG(name = name, env = env, 
                  deterministic = deterministic, save_tensorboard = save_tensorboard, save_matplotlib = save_matplotlib, 
                  lr_actor = 0.001, lr_critic = 0.001, max_action = 1., gamma = 0.9, actor_layers = [400, 300], critic_layers = [400, 300])
    
    train_params = policy.TrainConfig()    
    train_params.use_prioritized_rb = False
    train_params.max_epochs = 1500
    train_params.episode_max_steps = 200
    train_params.n_warmup = 300
    train_params.update_interval = 1
    train_params.test_interval = 1000
    train_params.test_episodes = 5
    
    train_params.memory_capacity = 30000 # 1000000
    train_params.batch_size = 32
    train_params.sigma = 0.1
    train_params.tau = 0.01
    
    train_params.save_model_interval = 10000 # save checkpoints every X steps
    
    # policy = BI_RES_DDPG(name = name, env = env,
    #                      deterministic = deterministic, save_tensorboard = save_tensorboard, save_matplotlib = save_matplotlib, 
    #                      lr_actor = 0.001, lr_critic = 0.001, gamma = 0.99, eta = 0.05, actor_layers = [400, 300], critic_layers = [400, 300])
    
    # train_params = policy.TrainConfig()    
    # train_params.use_prioritized_rb = True
    # train_params.max_epochs = 1500
    # train_params.episode_max_steps = 200
    # train_params.n_warmup = 300
    # train_params.update_interval = 1
    # train_params.test_interval = 1000
    # train_params.test_episodes = 5
    
    # train_params.memory_capacity = 30000 # 1000000
    # train_params.batch_size = 32
    # train_params.sigma = 0.1
    # train_params.tau = 0.01
    
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
