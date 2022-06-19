import yaml
from gibson2.envs.igibson_env import iGibsonEnv
import pybullet as p
from stable_baselines3 import SAC
from feature_extractor import CustomCombinedExtractor
from stable_baselines3.common.env_checker import check_env

headless = True
config_filename = "configs/straight_env.yaml"
config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

# Create iGibson environment
env = iGibsonEnv(config_file=config_data, mode="gui" if not headless else "headless")
if not headless:
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# Create SAC model
learning_rate = 3e-4
buffer_size = 150_000
batch_size = 256
tau = 0.005
gamma = 0.99
timesteps = 100_000

policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
                     #share_features_extractor=False
                     )
model = SAC("MultiInputPolicy", env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            verbose=1,
            #tensorboard_log="./sac_tensorboard/"
            )
model.learn(total_timesteps=timesteps, log_interval=4)
#model.save("straight_on")
