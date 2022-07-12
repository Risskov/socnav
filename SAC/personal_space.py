import rvo2
import yaml
import pybullet as p
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.objects.pedestrian import Pedestrian
"""
action_timestep = 1/10.
neighbor_dist = 5
max_neighbors = 2
time_horizon = 2.0
time_horizon_obst = 2.0
orca_radius = 0.3
orca_max_speed = 0.3

orca_sim = rvo2.PyRVOSimulator(
            action_timestep,
            neighbor_dist,
            max_neighbors,
            time_horizon,
            time_horizon_obst,
            orca_radius,
            orca_max_speed)

orca_ped = orca_sim.addAgent((0, 0))

config_filename = "configs/custom_env.yaml"
config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

config_data['use_ped_vel'] = True
config_data['scene_id'] = "H"
config_data['num_pedestrians'] = 3

env = iGibsonEnv(config_file=config_data, mode="headless")
ped = Pedestrian(style=0, visual_only=True)
env.simulator.import_object(ped)
initial_orn = p.getQuaternionFromEuler(ped.default_orn_euler)
ped.set_position_orientation((0,0,0), initial_orn)

for _ in range(1):
    obs = env.reset()
    for _ in range(500):
        print("Orientation: ", ped.get_orientation())
        print("Yaw: ", ped.get_yaw())
"""
import numpy as np
import matplotlib.pyplot as plt
def get_envL():
    envL = [[0, 0, 1, 1],
            [3, 0, 1, 1],
            [-3, 0, 1, 1]]
    return envL
def get_envI():
    envI = [[0, 0, 1, 0.5],
            [3, 0, 1, 0.5],
            [-3, 0, 1, 0.5]]
    return envI
def get_envX():
    envX = [[0, 0, 0.75, 0.75],
            [3, 0, 1, 0.5],
            [-3, 0, 1, 0.5],
            [0, 3, 0.5, 1],
            [0, -3, 0.5, 1]]
    return envX
def get_envH():
    envH = [[-3.25, 3, 0.75, 0.75],
            [0.25, 3, 0.75, 0.75],
            [3.25, 3, 0.75, 0.75],
            [-3.25, -3, 0.75, 0.75],
            [-0.25, -3, 0.75, 0.75],
            [3.25, -3, 0.75, 0.75]]
    return envH
def get_envE():
    envE = [[-6.75, 6.25, 1.75, 2.25],
            [-8.25, 1.25, 0.75, 0.75],
            [-8.25, -2.75, 0.75, 0.75],
            [-8.25, -8.25, 0.75, 0.75],
            [-0.75, -8.25, 0.75, 0.75],
            [-1.75, -0.25, 1.25, 1.25],
            [0.5, 4.5, 0.25, 0.25],
            [0.5, 9.25, 0.25, 0.25],
            [3.25, 4.5, 0.25, 0.25],
            [3.25, 9.25, 0.25, 0.25],
            [3.25, -2.25, 0.25, 0.25],
            [3.25, -8.25, 0.25, 0.25],
            [7.5, -0.75, 1.25, 1.25],
            [7.5, -8.5, 1.5, 0.5]]
    return envE


def sample_new_pos(self, env):
    scene_id = env.scene.scene_id
    if scene_id == "straight":
        areas = get_envL()
    elif scene_id == "straight_narrow":
        areas = get_envI()
    elif scene_id == "cross_narrow":
        areas = get_envX()
    elif scene_id == "H":
        areas = get_envH()
    rnd = np.random.randint(len(areas))
    return random_point_ellipse(*areas[rnd])


def random_point_ellipse(cx, cy, rx, ry):
    angle = np.random.random()*2*np.pi
    dist = np.random.random()

    x = cx + np.sqrt(dist) * np.cos(angle) * rx
    y = cy + np.sqrt(dist) * np.sin(angle) * ry
    return x, y


xs = []
ys = []
for area in get_envE():
    for _ in range(300):
        x1, y1 = random_point_ellipse(*area)
        xs.append(x1)
        ys.append(y1)

plt.plot(xs, ys, 'x')
plt.axis('equal')
plt.show()
