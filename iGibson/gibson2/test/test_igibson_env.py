import gibson2
from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import os
from gibson2.utils.assets_utils import download_assets, download_demo_data


def test_env():
    download_assets()
    download_demo_data()
    config_filename = os.path.join(
        gibson2.root_path, 'test', 'test_house.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='headless')
    try:
        for j in range(2):
            env.reset()
            for i in range(300):    # 300 steps, 30s world time
                s = time()
                action = env.action_space.sample()
                ts = env.step(action)
                print('ts', 1 / (time() - s))
                if ts[2]:
                    print("Episode finished after {} timesteps".format(i + 1))
                    break
    finally:
        env.close()


def test_env_reload():
    download_assets()
    download_demo_data()
    config_filename = os.path.join(
        gibson2.root_path, 'test', 'test_house.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='headless')
    try:
        for i in range(3):
            env.reload(config_filename)
            env.reset()
            for i in range(300):    # 300 steps, 30s world time
                s = time()
                action = env.action_space.sample()
                ts = env.step(action)
                print('ts', 1 / (time() - s))
                if ts[2]:
                    print("Episode finished after {} timesteps".format(i + 1))
                    break
    finally:
        env.close()


def test_env_reset():
    download_assets()
    download_demo_data()
    config_filename = os.path.join(
        gibson2.root_path, 'test', 'test_house.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='headless')

    class DummyTask(object):
        def __init__(self):
            self.reset_scene_called = False
            self.reset_agent_called = False
            self.get_task_obs_called = False

        def get_task_obs(self, env):
            self.get_task_obs_called = True

        def reset_scene(self, env):
            self.reset_scene_called = True

        def reset_agent(self, env):
            self.reset_agent_called = True

    env.task = DummyTask()
    env.reset()
    assert env.task.reset_scene_called
    assert env.task.reset_agent_called
    assert env.task.get_task_obs_called
