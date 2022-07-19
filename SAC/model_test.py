from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from gibson2.envs.igibson_env import iGibsonEnv
import pybullet as p
import yaml
import numpy as np
import matplotlib.pyplot as plt
import csv

def plot_lidar(scan):
    #step = 2*np.pi/360
    #angles = np.linspace(-29*step, 210*step, 228)
    angles = np.linspace(0, 2*np.pi, len(scan))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(angles, scan)
    plt.show()

def write_to_file(arr, name):
    np.savetxt(f'./data/{name}.csv', arr, delimiter=',')
    #with open(f"./data/{name}", "w") as csvfile:
    #    csvwriter = csv.writer(csvfile)

config_filename = "configs/custom_env.yaml"
config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

config_data['use_ped_vel'] = True
config_data['use_orca'] = False
config_data['scene_id'] = "straight_narrow"
config_data['num_pedestrians'] = 2
#config_data['record'] = True

env = iGibsonEnv(config_file=config_data, mode="gui")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-80, cameraTargetPosition=[0,0,5])

#model = SAC.load("./models/3_256net_2peds4nodes_autoent_meanpooling", env=env)
# larger_net_120scan_1ped_4wp_pot_003ent_ero6_3m
# X_256net_120scan_1ped_vel_4wp_pot_001ent_sde_2m

#model = SAC.load("./tmp/3_256net_2peds4nodes_autoent_maxpooling.zip", env=env)
model = SAC.load("./tmp/best_model/3_256net_2peds4nodes_autoent_maxpooling", env=env)
#print(model.get_parameters())
#check_env(env)
# best: sc_narrow_H_no_pot_long_low_reverse_no_peds_rb_no_conv_flatten

episodes = 100
collisions = 0
timeouts = 0
i = 0
rob_pos = []
ped_pos = []
for _ in range(episodes):
    obs = env.reset()
    #print(obs)
    rewards = 0
    # while True:
    # for ped in env.task.pedestrians:
    #     ped_pos.append(ped.get_position()[:2])
    #     rob_pos.append(env.robots[0].get_position()[:2])
    for j in range(500):
        #action, _states = model.predict(obs, deterministic=True)    #deterministic=True
        action = [-0.666, 0.]
        #action = [1, 0]
        obs, reward, done, info = env.step(action)
        rewards += reward
        # for ped in env.task.pedestrians:
        #     ped_pos.append(ped.get_position()[:2])
        #     rob_pos.append(env.robots[0].get_position()[:2])

        if i % 50 == 0:
            #print(obs["pedestrians"])
            #print(obs["scan"].view(-1, 1, 360))
            #plot_lidar(obs["scan"])
            pass
        i = i+1
        if done:
            #print(info)
            if info["success"] == False:
                collisions += 1
                print("COLLISION!")
            break
        if j == 499:
            timeouts += 1
            print("TIMEOUT!")
print("Number of collisions: ", collisions)
print("Number of timeouts: ", timeouts)
#write_to_file(ped_pos, "ped_positions")
#write_to_file(rob_pos, "rob_positions")

