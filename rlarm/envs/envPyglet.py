import numpy as np
import pyglet


####################################################################################################
class ArmEnv(object):
    def __init__(self, dt = 0.1, action_bound = [-1, 1], goal = {'x': 150., 'y': 150., 'l': 20} ):
        self.viewer = None
        self.dt = dt
        self.action_bound = action_bound
        self.goal = goal

        self.on_goal = 0
        
        self.arm_info = np.zeros(2, dtype=[('l', np.float32), ('r', np.float32)])
        
        self.arm_info['l'] = [200, 200]
        self.arm_info['r'] = [-1*np.pi/2, -1*np.pi/2]
        
        self.center_coord = np.array([400, 400])
        
    def _pose(self):
        (a1l, a2l) = self.arm_info['l']  # mm, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        
        a1xy = self.center_coord    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 900, (self.goal['y'] - a1xy_[1]) / 900]
        dist2 = [(self.goal['x'] - a2xy_[0]) / 900, (self.goal['y'] - a2xy_[1]) / 900]
        
        return dist1, dist2, a1xy_, a2xy_

    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize
        
        # limit angles
        # if self.arm_info['r'][0] < 0:
        #     self.arm_info['r'][0] = 0
        # elif self.arm_info['r'][0] > np.pi:
        #     self.arm_info['r'][0] = np.pi
            
        # if self.arm_info['r'][1] < -np.pi/6:
        #     self.arm_info['r'][1] = -np.pi/6
        # elif self.arm_info['r'][1] > (4*np.pi)/3:
        #     self.arm_info['r'][1] = (4*np.pi)/3
            
        dist1, dist2, a1xy_, a2xy_ = self._pose()
        
        r = -np.sqrt(dist2[0]**2 + dist2[1]**2)

        # done and reward
        if (self.goal['x'] - self.goal['l']/2 < a2xy_[0] < self.goal['x'] + self.goal['l']/2) and (self.goal['y'] - self.goal['l']/2 < a2xy_[1] < self.goal['y'] + self.goal['l']/2):
            r += 1.
            self.on_goal += 1
            if self.on_goal > 50:
                done = True
        else:
            self.on_goal = 0

        # state
        s = np.concatenate((a1xy_/400, a2xy_/400, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s, r, done

    def reset(self):
        self.goal['x'] = np.random.rand()*900.
        self.goal['y'] = np.random.rand()*900.
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
        self.on_goal = 0
        
        dist1, dist2, a1xy_, a2xy_ = self._pose()
        
        # state
        s = np.concatenate((a1xy_/400, a2xy_/400, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians


class Viewer(pyglet.window.Window):
    def __init__(self, arm_info, goal, bar_thc = 5):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=900, height=900, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        
        self.bar_thc = bar_thc
        
        self.arm_info = arm_info
        self.goal_info = goal
        
        self.center_coord = np.array([400, 400])
        
        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,               
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (249, 86, 86) * 4))    # color
        
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [0, 0,                
                     0, 0,
                     0, 0,
                     0, 0]),
            ('c3B', (86, 109, 249) * 4,))   
        
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [0, 0,             
                     0, 0,
                     0, 0,
                     0, 0]), 
            ('c3B', (86, 109, 249) * 4,))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # update goal
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2,
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2)

        # update arm
        (a1l, a2l) = self.arm_info['l']     
        (a1r, a2r) = self.arm_info['r']    
        a1xy = self.center_coord           
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  
        
        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))

    # convert the mouse coordinate to goal's coordinate
    # def on_mouse_motion(self, x, y, dx, dy):
    #     self.goal_info['x'] = x
    #     self.goal_info['y'] = y


if __name__ == '__main__':
    env = ArmEnv()
    env.render()    # Need two render before begin
    env.render()
    while True:
        env.render()
        # env.reset()
        env.step(env.sample_action())
