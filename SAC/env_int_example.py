import logging
import os

import yaml
import numpy as np
import gibson2
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.render.profiler import Profiler

def set_camera_pos(sim):
    sim.viewer.py = 1
    initial_view_direction = [0, -1, -0.3]
    sim.viewer.theta = np.arctan2(
        initial_view_direction[1], initial_view_direction[0])
    sim.viewer.phi = np.arctan2(initial_view_direction[2], np.sqrt(initial_view_direction[0] ** 2 +
                                                                   initial_view_direction[1] ** 2))
    sim.viewer.view_direction = np.array(
        [np.cos(sim.viewer.theta) * np.cos(sim.viewer.phi), np.sin(sim.viewer.theta) * np.cos(
            sim.viewer.phi), np.sin(sim.viewer.phi)])

def main(selection="user", headless=False, short_exec=False):
    """
    Social Navigation
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_filename = "configs/my_env.yaml"
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Shadows and PBR do not make much sense for a Gibson static mesh
    config_data["enable_shadow"] = False
    config_data["enable_pbr"] = False

    env = iGibsonEnv(config_file=config_data, mode="gui" if not headless else "headless")
    sim = env.simulator
    if not headless:
        set_camera_pos(sim)

    max_iterations = 1 if not short_exec else 1
    for j in range(max_iterations):
        print("Resetting environment")
        env.reset()
        for i in range(300):
            with Profiler("Environment action step"):
                #action = env.action_space.sample()
                action = [1., 0.]
                state, reward, done, info = env.step(action)
                #print(action)
                print(state['scan'].shape)
                lidar = sim.renderer.get_lidar_all()
                print("Dimensions of the lidar observation: {}".format(lidar.shape))
                if done:
                    print("Episode finished after {} time steps".format(i + 1))
                    break
    env.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()