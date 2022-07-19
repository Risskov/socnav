import gym
import yaml
import os
from gibson2.envs.igibson_env import iGibsonEnv
from feature_extractor_sym import CustomCombinedExtractor, LargeMaxpoolExtractor, LargeMeanpoolExtractor
#from feature_extractor import CustomCombinedExtractor
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from save_model_callback import SaveOnBestTrainingRewardCallback

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

def start_training(env, name, timesteps, ent, feat, layer_dim=(3, 256)):
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
    net_arch = [layer_dim[1]]*layer_dim[0]

    # Set up callback
    os.makedirs(log_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback(name, check_freq=1000, log_dir=log_dir)

    policy_kwargs = dict(features_extractor_class=feat,
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
    del model

log_dir = "tmp/"
#envs = ["straight", "cross", "bend"]
envs = ["H", "cross_narrow", "straight_narrow"]
num_peds = [2, 2, 2]
env_select = 1
num_envs = 3  # Number of processes to use
# Create the vectorized environment
env = SubprocVecEnv([make_env(i, envs, num_peds) for i in range(num_envs)], start_method='fork')
#env = VecFrameStack(env, n_stack=4, channels_order="last")
env = VecMonitor(env, log_dir)
timesteps = 1_500_000
#name = "X_512net_2peds4nodes_auto_maxpooling"

start_training(env, "3_256net_2peds4nodes_autoent_meanpooling", timesteps,
                ent='auto', feat=LargeMeanpoolExtractor)

start_training(env, "3_256net_2peds4nodes_autoent_maxpooling", timesteps,
                ent='auto', feat=LargeMaxpoolExtractor)

# start_training(env, "3_512net_432peds4nodes_autoent_meanpooling", timesteps,
#                ent='auto', feat=LargeMeanpoolExtractor, layer_dim=(3, 512))
# start_training(env, "3_512net_432peds4nodes_autoent_maxpooling", timesteps,
#                ent='auto', feat=LargeMaxpoolExtractor, layer_dim=(3, 512))

