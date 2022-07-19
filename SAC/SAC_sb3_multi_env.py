import gym
import yaml
import os
from gibson2.envs.igibson_env import iGibsonEnv
from feature_extractor_sym import CustomCombinedExtractor
#from feature_extractor import CustomCombinedExtractor
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from save_model_callback import SaveOnBestTrainingRewardCallback
from gym.wrappers.time_limit import TimeLimit


def single_env(scene_id, envs, num_peds):
    config_filename = f"configs/custom_env.yaml"
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config_data['scene_id'] = envs[scene_id]
    config_data['num_pedestrians'] = num_peds[scene_id]
    config_data['use_orca'] = False
    env = iGibsonEnv(config_file=config_data, mode="headless")
    return env

def make_env(rank, envs, num_peds):
    def _init():
        config_filename = f"configs/custom_env.yaml"
        config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        config_data['scene_id'] = envs[rank]
        config_data['num_pedestrians'] = num_peds[rank]
        config_data['use_ped_vel'] = True
        config_data['use_orca'] = False
        action_timestep = 1/10.
        env = iGibsonEnv(config_file=config_data, mode="headless", action_timestep=action_timestep)
        #env = TimeLimit(env, max_episode_steps=200)
        return env
    return _init

# if len(envs) < num_envs:
#     print("Environment list shorter than number of environments")
#     exit(666)

# Create single environment
#env = single_env(1)
# env_fn = lambda: single_env(1)
# env = DummyVecEnv([env_fn for _ in range(4)])
# env = DummyVecEnv([make_env(1) for _ in range(4)])
# env = VecFrameStack(env, n_stack=4, channels_order="last")
# env = VecMonitor(env, log_dir)
#env = Monitor(env, log_dir)

def start_training(env, name, timesteps, ent=0.005):
    # Create SAC model
    learning_rate = 3e-4
    buffer_size = 1_000_000
    batch_size = 256  # 256
    tau = 0.005
    gamma = 0.99  # 0.99

    ent_coef = ent
    target_ent = -2.  # te
    learning_starts = 50000
    use_sde = True
    use_sde_at_warmup = True
    layer_dim = 256  # 1024 512 256
    net_arch = [layer_dim, layer_dim, layer_dim]
    # Set up callback
    os.makedirs(log_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback(name, check_freq=1000, log_dir=log_dir)

    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
                         net_arch=net_arch,
                         #share_features_extractor=False
                         #log_std_init=-2,
                         )
    #model = SAC.load("./models/straight_cross_bend_multi_peds_random", env=env)
    model = SAC("MultiInputPolicy", env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                policy_kwargs=policy_kwargs,
                #replay_buffer_kwargs=dict(handle_timeout_termination=True),
                verbose=1,
                tensorboard_log="./sac_tensorboard/",
                gradient_steps=-1,
                ent_coef=ent_coef,
                #target_entropy=target_ent,
                use_sde=use_sde,
                use_sde_at_warmup=use_sde_at_warmup,
                #learning_starts=learning_starts,
                )
    model.learn(total_timesteps=timesteps, log_interval=4, tb_log_name=name, callback=callback)
    model.save(name)
    model.save_replay_buffer(f"./buffers/{name}")

def continue_training(env, name, suffix, timesteps, best=False, use_replay_buffer=True):
    # Set up callback
    os.makedirs(log_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback(name, check_freq=1000, log_dir=log_dir)
    if best:
        model = SAC.load(f"./tmp/best_model/{name}_continued", env=env)  # load the final model
    else:
        model = SAC.load(f"./models/{name}", env=env)  # load the final model
    if use_replay_buffer:
        model.load_replay_buffer(f"./buffers/{name}")
    model.learn(total_timesteps=timesteps,
                log_interval=4,
                tb_log_name=f"{name}_{suffix}",
                callback=callback,
                reset_num_timesteps=False)
    model.save(f"{name}_{suffix}")
    model.save_replay_buffer(f"./buffers/{name}_{suffix}")


log_dir = "tmp/"
#envs = ["straight", "cross", "bend"]
envs = ["H", "cross_narrow", "straight_narrow"]
num_peds = [2, 2, 2]
env_select = 2
num_envs = 3  # Number of processes to use
# Create the vectorized environment
env = SubprocVecEnv([make_env(env_select, envs, num_peds) for i in range(num_envs)], start_method='fork')
#env = VecFrameStack(env, n_stack=4, channels_order="last")
env = VecMonitor(env, log_dir)
timesteps = 1_500_000
name = "testing_cone" #"X_256net_2peds4nodes_0005ent_maxpooling"
suffix = "2peds"
new_training = True

if new_training:
    start_training(env, name, timesteps, ent=0.005)
else:
    continue_training(env, name, suffix, timesteps, best=False, use_replay_buffer=True)

