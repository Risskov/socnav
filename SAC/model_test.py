from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from gibson2.envs.igibson_env import iGibsonEnv
import pybullet as p
import yaml
import numpy as np
import matplotlib.pyplot as plt

def plot_lidar(scan):
    #step = 2*np.pi/360
    #angles = np.linspace(-29*step, 210*step, 228)
    angles = np.linspace(0, 2*np.pi, len(scan))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(angles, scan)
    plt.show()

config_filename = "configs/custom_env.yaml"
config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

config_data['scene_id'] = "cross_narrow"
config_data['num_pedestrians'] = 2

env = iGibsonEnv(config_file=config_data, mode="gui")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-80, cameraTargetPosition=[0,0,5])

#model = SAC.load("./models/larger_net_120scan_1ped_4wp_pot_003ent_ero6_3m", env=env)
model = SAC.load("./tmp/best_model.zip", env=env)
#model = SAC.load("./tmp/best_model/512net_120scan_123ped_orca_4wp_pot_2te_sde_10m.zip", env=env)
#print(model.get_parameters())
#check_env(env)
# best: sc_narrow_H_no_pot_long_low_reverse_no_peds_rb_no_conv_flatten

episodes = 50
i = 0
for _ in range(episodes):
    obs = env.reset()
    #print(obs)
    rewards = 0
    # while True:
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)    #deterministic=True
        #action = [-0.666, 0]
        obs, reward, done, info = env.step(action)
        rewards += reward
        #print(obs["pedestrians"])

        if i % 50 == 0:
            #print(obs)
            #print(obs["scan"].view(-1, 1, 360))
            #plot_lidar(obs["scan"])
            pass
        i = i+1
        if done:
            #print(info)
            if info["success"] == False:
                print("COLLISION!")
            break
