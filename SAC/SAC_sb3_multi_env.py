import yaml
import os
from gibson2.envs.igibson_env import iGibsonEnv
from feature_extractor import CustomCombinedExtractor
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from save_model_callback import SaveOnBestTrainingRewardCallback

#envs = ["straight", "cross", "bend"]
envs = ["H", "cross_narrow", "straight_narrow"]
num_peds = [3, 2, 1]

def single_env(scene_id):
    config_filename = f"configs/custom_env.yaml"
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config_data['scene_id'] = envs[scene_id]
    env = iGibsonEnv(config_file=config_data, mode="headless")
    return env

def make_env(rank, seed=0):
    def _init():
        config_filename = f"configs/custom_env.yaml"
        config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        config_data['scene_id'] = envs[rank]
        config_data['num_pedestrians'] = num_peds[rank]
        # action_timestep = 1/4.
        env = iGibsonEnv(config_file=config_data, mode="headless")
        #env = Monitor(env)
        return env
    return _init


num_envs = 3  # Number of processes to use
if len(envs) < num_envs:
    print("Environment list shorter than number of environments")
    exit(666)

# Set up callback
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# Create the vectorized environment
env = SubprocVecEnv([make_env(i) for i in range(num_envs)], start_method='fork')
#env = VecFrameStack(env, n_stack=3)
env = VecMonitor(env, log_dir)

# Create single environment
# env = single_env(1)
# env = Monitor(env, log_dir)

# Create SAC model
learning_rate = 3e-4
buffer_size = 1_000_000
batch_size = 256
tau = 0.005
gamma = 0.99

ent_coef = 0.05
target_ent = -2.  # te
learning_starts = 5000
use_sde = True
use_sde_at_warmup = True
timesteps = 15_000_000
net_arch = [512, 512, 512] #[256, 256, 256]


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
            verbose=1,
            tensorboard_log="./sac_tensorboard/",
            gradient_steps=-1,
            #ent_coef=ent_coef,
            target_entropy=target_ent,
            use_sde=use_sde,
            use_sde_at_warmup=use_sde_at_warmup,
            learning_starts=learning_starts,
            )
model.learn(total_timesteps=timesteps, log_interval=4, callback=callback)
model.save("512net_120scan_123ped_orca_4wp_pot_2et_sde_15m")
model.save_replay_buffer("./buffers/512net_120scan_123ped_orca_4wp_pot_2et_sde_15m")
