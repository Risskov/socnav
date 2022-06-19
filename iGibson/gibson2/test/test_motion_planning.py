import gibson2
from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import os
from gibson2.utils.assets_utils import download_assets, download_demo_data
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import numpy as np
import matplotlib.pyplot as plt

def test_occupancy_grid():
    print("Test env")
    download_assets()
    download_demo_data()
    config_filename = os.path.join(gibson2.root_path, 'test', 'test_house_occupancy_grid.yaml')
    
    nav_env = iGibsonEnv(config_file=config_filename, mode='headless')
    nav_env.reset()
    nav_env.robots[0].set_position_orientation([0,0,0],[0,0,0,1])
    nav_env.simulator.step()

    action = nav_env.action_space.sample()
    ts = nav_env.step(action)
    assert np.sum(ts[0]['occupancy_grid'] == 0) > 0
    assert np.sum(ts[0]['occupancy_grid'] == 1) > 0
    plt.imshow(ts[0]['occupancy_grid'][:,:,0])
    plt.colorbar()
    plt.savefig('occupancy_grid.png')
    nav_env.clean()


def test_base_planning():
    print("Test env")
    download_assets()
    download_demo_data()
    config_filename = os.path.join(gibson2.root_path, 'test', 'test_house_occupancy_grid.yaml')

    nav_env = iGibsonEnv(config_file=config_filename, mode='headless')
    motion_planner = MotionPlanningWrapper(nav_env)
    state = nav_env.reset()
    nav_env.robots[0].set_position_orientation([0,0,0],[0,0,0,1])
    nav_env.simulator.step()
    plan = None
    itr = 0
    while plan is None and itr < 10:
        plan = motion_planner.plan_base_motion([0.5,0,0])
        print(plan)
        itr += 1
    motion_planner.dry_run_base_plan(plan)

    assert len(plan) > 0 
    nav_env.clean()

