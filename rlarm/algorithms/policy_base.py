import time
import os
import numpy as np
import tensorflow as tf

from gym.utils import colorize

from rlarm.algorithms.tools.utils import REPO_ROOT
    
class TypeExcept(Exception):
    pass

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
        
        if self.env.name() == "PygletArm2D":
            self.act_dim = 2
            self.state_dim = (9,)

            print(colorize("State dim: {}".format(self.state_dim), "yellow"))
            print(colorize("Act dim: {}".format(self.act_dim), "yellow"))
            
        elif self.env.name() == 'ArmiPy':
            self.act_dim = 3
            self.state_dim = (13,)

            print(colorize("State dim: {}".format(self.state_dim), "yellow"))
            print(colorize("Act dim: {}".format(self.act_dim), "yellow"))
            
        else:
            raise TypeExcept("Unrecognized Env!")   
            
        if self.save_tensorboard:
            os.makedirs(os.path.join(REPO_ROOT, 'tensorboard'), exist_ok = True)
            self.writer = tf.summary.create_file_writer(os.path.join(REPO_ROOT, 'tensorboard', self.name))

    # ----------------------------------------------------------------------------------------------------
    class TrainConfig():
        pass

    # ----------------------------------------------------------------------------------------------------
    def reset(self):
        if self.env.name() == "PygletArm2D":
            ob = self.env.reset()
            
        elif self.env.name() == 'ArmiPy':
            rec = self.env.reset()
            
            ob_rec = []
            ob_rec.append(rec["a1X"])
            ob_rec.append(rec["a1Y"])
            ob_rec.append(rec["a1Z"])
            
            ob_rec.append(rec["a2X"])
            ob_rec.append(rec["a2Y"])
            ob_rec.append(rec["a2Z"])
            
            ob_rec.append(rec["dist1X"])
            ob_rec.append(rec["dist1Y"])
            ob_rec.append(rec["dist1Z"])
            
            ob_rec.append(rec["dist2X"])
            ob_rec.append(rec["dist2Y"])
            ob_rec.append(rec["dist2Z"])      
            
            goal = 0.0
            ob_rec.append(goal) 
            
            ob = np.array(ob_rec)  
            
        else:
            raise TypeExcept("Unrecognized Env!")   
        
        return ob
    
        # ----------------------------------------------------------------------------------------------------
    def initialPose(self):
        if self.env.name() == "PygletArm2D":
            ob = self.env.reset()
            
        elif self.env.name() == 'ArmiPy':
            rec = self.env.initialState()
            
            ob_rec = []
            ob_rec.append(rec["a1X"])
            ob_rec.append(rec["a1Y"])
            ob_rec.append(rec["a1Z"])
            
            ob_rec.append(rec["a2X"])
            ob_rec.append(rec["a2Y"])
            ob_rec.append(rec["a2Z"])
            
            ob_rec.append(rec["dist1X"])
            ob_rec.append(rec["dist1Y"])
            ob_rec.append(rec["dist1Z"])
            
            ob_rec.append(rec["dist2X"])
            ob_rec.append(rec["dist2Y"])
            ob_rec.append(rec["dist2Z"])      
            
            goal = 0.0
            ob_rec.append(goal) 
            
            ob = np.array(ob_rec)  
            
        else:
            raise TypeExcept("Unrecognized Env!")   
        
        return ob

    # ----------------------------------------------------------------------------------------------------
    def step(self, action):
        if self.env.name() == "PygletArm2D":
            ob_next, r, done = self.env.step(action)
            self.env.render()
            
        elif self.env.name() == 'ArmiPy':
            converted_act = {
                             "armJoint0": float(action[0]),
                             "armJoint1": float(action[1]),
                             "armJoint2": float(action[2]),
                             }
   
            rec = self.env.step(converted_act)
            
            ob_rec = []
            ob_rec.append(rec["a1X"])
            ob_rec.append(rec["a1Y"])
            ob_rec.append(rec["a1Z"])
            
            ob_rec.append(rec["a2X"])
            ob_rec.append(rec["a2Y"])
            ob_rec.append(rec["a2Z"])
            
            ob_rec.append(rec["dist1X"])
            ob_rec.append(rec["dist1Y"])
            ob_rec.append(rec["dist1Z"])
            
            ob_rec.append(rec["dist2X"])
            ob_rec.append(rec["dist2Y"])
            ob_rec.append(rec["dist2Z"])   
            
            goal = rec["on_goal"]
            ob_rec.append(1. if goal else 0.)
            
            ob_next = np.array(ob_rec)
            r = rec["reward"]
            done = rec["done"]   
             
        else:
            raise TypeExcept("Unrecognized Env!")   
        
        return ob_next, r, done

    # ----------------------------------------------------------------------------------------------------
    def get_sample(self):
        if self.env.name() == "PygletArm2D":
            acts = self.env.sample_action()
            
        elif self.env.name() == 'ArmiPy':
            random_acts = []        
            rec = self.env.sample()
            
            random_acts.append(rec["armJoint0"])
            random_acts.append(rec["armJoint1"])
            random_acts.append(rec["armJoint2"])  
            acts = np.array(random_acts)
            
        else:
            raise TypeExcept("Unrecognized Env!")   
        
        return acts
    
    # ----------------------------------------------------------------------------------------------------
    def act(self, state):
        pass

    # ----------------------------------------------------------------------------------------------------
    def fit_net(self, lr, returns, obs, actions, batch_size):
        pass

    # ----------------------------------------------------------------------------------------------------
    def train(self, config: TrainConfig):
        pass
    
    # ----------------------------------------------------------------------------------------------------
    def set_check_point(self, policy_dir):
        pass
    
    # ----------------------------------------------------------------------------------------------------
    def load(self):
        pass
    
    # ----------------------------------------------------------------------------------------------------
    def evaluate(self, episode_max_steps, sigma):
        pass