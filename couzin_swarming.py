import numpy as np
from numpy.linalg import *
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

data = pd.read_csv("C:/Users/Betsy/Desktop/\ds project/analytics/distance and direction2.csv")
dimension = '2d'    # 2d/3d
n = 30             # number of agents
dt = 0.1
r_r = 18 #12
r_o = 60
r_a = 452 #245.2
field_of_view = 3*pi/2
theta_dot_max = 1
speed_list = []
distance_change_sorted = sorted(data["distance_change"])
speed_chosen = distance_change_sorted[int(len(distance_change_sorted) * 0.25): int(len(distance_change_sorted) * 0.55)]
for i in range(30):
    temp = np.random.choice(speed_chosen)
    speed_list.append(temp)

np.seterr(divide='ignore', invalid='ignore')


class Field:
    def __init__(self):
        self.width = 1280    # x_max[m]1920
        self.height = 580   # y_max[m]1080
        self.depth = 100    # z_max[m]


class Agent:
    def __init__(self, agent_id, speed):
        self.id = agent_id
        self.pos = np.array([0, 0, 0])
        self.pos[0] = np.random.uniform(0, field.width)
        self.pos[1] = np.random.uniform(0, field.height)
        self.pos[2] = np.random.uniform(0, field.depth)
        self.vel = np.random.uniform(-1, 1, 3)
        if dimension == '2d':
            self.pos[2] = 0
            self.vel[2] = 0
        self.vel = self.vel / norm(self.vel) * speed

    def update_position(self, delta_t):
        new_pos = self.pos + self.vel * delta_t
        if new_pos[0] < 0 or new_pos[0] > field.width:
            # Turn right if the agent hits the x-boundaries
            self.vel[0] = -self.vel[0]
        if new_pos[1] < 0 or new_pos[1] > field.height:
            # Turn right if the agent hits the y-boundaries
            self.vel[1] = -self.vel[1]
        if new_pos[2] < 0 or new_pos[2] > field.depth:
            # Turn right if the agent hits the z-boundaries
            self.vel[2] = -self.vel[2]

        # Update position to the new clipped position
        self.pos = new_pos
'''
class Leader:
    def __init__(self, agent_id, speed):
        self.id = agent_id
        self.pos = np.array([0, 0, 0])
        self.pos[0] = np.random.uniform(0, field.width)
        self.pos[1] = np.random.uniform(0, field.height)
        self.pos[2] = np.random.uniform(0, field.depth)
        self.vel = np.random.uniform(-1, 1, 3)
        if dimension == '2d':
            self.pos[2] = 0
            self.vel[2] = 0
        self.vel = self.vel / norm(self.vel) * (speed)  # Increase the speed slightly for the leader
        self.r_o = r_o

    def update_position(self, delta_t):
        new_pos = self.pos + self.vel * delta_t
        if new_pos[0] < 0 or new_pos[0] > field.width:
            # Turn right if the agent hits the x-boundaries
            self.vel[0] = -self.vel[0]
        if new_pos[1] < 0 or new_pos[1] > field.height:
            # Turn right if the agent hits the y-boundaries
            self.vel[1] = -self.vel[1]
        if new_pos[2] < 0 or new_pos[2] > field.depth:
            # Turn right if the agent hits the z-boundaries
            self.vel[2] = -self.vel[2]

        # Update position to the new clipped position
        self.pos = new_pos
'''
def create_swarm_with_leader(n_agents): #, leader_speed
    swarm = []
    for i in range(n_agents):
        agent_speed = speed_list[i]
        agent = Agent(i, agent_speed)
        swarm.append(agent)
    #leader = Leader(n_agents, leader_speed)
    #swarm.append(leader)
    return swarm

def rotation_matrix_about(axis, theta):
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


if __name__ == '__main__':
    
    field = Field()
    swarm = create_swarm_with_leader(n)

    fig = plt.figure()
    if dimension == '3d':
        ax = fig.gca(projection='3d')
    else:
        ax = fig.gca()

    
    #new_columns = {f'agent{i}': f'agent{i + 1}' for i in range(n)}
    #positions_history.rename(columns=new_columns, inplace=True)
    times_list ={"times" : [0] * 500}
    positions_history = pd.DataFrame(times_list) #record the position
    for i in range(31):
        agentNamex = "agent" + str(i) + "x"
        positions_history[agentNamex] = [None] * 500
        agentNamey = "agent" + str(i) + "y"
        positions_history[agentNamey] = [None] * 500
    positions = []
    times = 0
    columns = ["frame_index", "fish_index", "x", "y"]
    result_df = pd.DataFrame(columns=columns)
    while times < 3000:
        #temp_position = []
        positions_history.loc[times, 'times'] = times
        x = np.array([])
        y = np.array([])
        z = np.array([])
        x_dot = np.array([])
        y_dot = np.array([])
        z_dot = np.array([])
        frame_data = {"frame_index": times}
        for agent in swarm:
            x = np.append(x, agent.pos[0])
            y = np.append(y, agent.pos[1])
            z = np.append(z, agent.pos[2])
            x_dot = np.append(x_dot, agent.vel[0])
            y_dot = np.append(y_dot, agent.vel[1])
            z_dot = np.append(z_dot, agent.vel[2])
            tempx = agent.pos[0]
            tempy = agent.pos[1]
            #print(tempx)
            #print(tempy)
            #print(temp)
            positions_history.loc[times, "agent" + str(agent.id) + "x"] = tempx
            positions_history.loc[times, "agent" + str(agent.id) + "y"] = tempy
            #positions_history[times]["agent" + str(agent.id)] = temp

            #result_df = result_df.append(pd.Series({**frame_data, **agent_data}), ignore_index=True)
        times = times + 1
        ax.clear()
        #positions_history.append([agent.pos.copy() for agent in swarm])

        if dimension == '2d':
            ax.quiver(x, y, x_dot / 30 * field.width, y_dot / 30 * field.width, color='#377eb8')
            ax.set_aspect('equal', 'box')
            ax.set_xlim(0, field.width)
            ax.set_ylim(0, field.height)
        else:
            ax.quiver(x, y, z, x_dot / 80 * field.width, y_dot / 80 * field.width,  z_dot / 80 * field.width, color='#377eb8')
            ax.quiver(swarm[-1].pos[0], swarm[-1].pos[1], swarm[-1].pos[2], 
                      swarm[-1].vel[0] / 80 * field.width, swarm[-1].vel[1] / 80 * field.width, 
                      swarm[-1].vel[2] / 80 * field.width, color='orange')
            ax.set_aspect('equal', 'box')
            ax.set_xlim(0, field.width)
            ax.set_ylim(0, field.height)
            ax.set_zlim(0, field.depth)
        plt.pause(0.0001)

        for agent in swarm:
            d = 0
            d_r = 0
            d_o = 0
            d_a = 0
            for neighbor in swarm:
                if agent.id != neighbor.id:
                    r = neighbor.pos - agent.pos
                    r_normalized = r/norm(r) #direction for agent and neighbor
                    norm_r = norm(r) #distance
                    agent_vel_normalized = agent.vel/norm(agent.vel) #direction of the agent
                    if acos(np.dot(r_normalized, agent_vel_normalized)) < field_of_view / 2: #visualized
                        if norm_r < r_r:
                            d_r = d_r - r_normalized
                        elif norm_r < r_o:
                            d_o = d_o + neighbor.vel/norm(neighbor.vel)
                        elif norm_r < r_a:
                            d_a = d_a + r_normalized 
            if norm(d_r) != 0:
                d = d_r
            
            elif norm(d_r) != 0 and norm(d_o) != 0:
                angle_between_roo = acos(np.dot(d_r, d_o))
                if angle_between_roo < field_of_view / 2:  
                    d = (d_o + d_r) / 2  
                else:
                    d = d_a
            elif norm(d_a) != 0:
                d = d_a
            elif norm(d_o) != 0:
                d = d_o
            
            if norm(d) != 0:
                z = np.cross(d/norm(d), agent.vel/norm(agent.vel))
                angle_between = asin(norm(z))
                if angle_between >= theta_dot_max*dt:
                    rot = rotation_matrix_about(z, theta_dot_max*dt)
                    agent.vel = np.asmatrix(agent.vel) * rot
                    agent.vel = np.asarray(agent.vel)[0]
                elif abs(angle_between)-pi > 0:
                    agent.vel = d/norm(d) * 2

        [agent.update_position(dt) for agent in swarm]
#print(result_df)
filtered = positions_history[positions_history['times'] % 3 == 0]
file_path = 'C:/Users/Betsy/ds_csv/try3.csv'
filtered.to_csv(file_path, index=False)