import numpy as np

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
def get_envI_large():
    envI = [[0, 0, 2.5, 1],
            [6.5, 0, 2.5, 1],
            [-6.5, 0, 2.5, 1]]
    return envI
def get_envX_large():
    envX = [[0, 0, 1, 1],
            [6.5, 0, 2.5, 1],
            [-6.5, 0, 2.5, 1],
            [0, 6.5, 1, 2.5],
            [0, -6.5, 1, 2.5]]
    return envX

def sample_new_pos(env):
    scene_id = env.scene.scene_id
    if scene_id == "straight":
        areas = get_envL()
    elif scene_id == "straight_narrow":
        areas = get_envI()
    elif scene_id == "cross_narrow":
        areas = get_envX()
    elif scene_id == "H":
        areas = get_envH()
    elif scene_id == "E":
        areas = get_envE()
    elif scene_id == "I_large":
        areas = get_envI_large()
    elif scene_id == "X_large":
        areas = get_envX_large()
    rnd_ellipse = np.random.randint(len(areas))
    return random_point_ellipse(*areas[rnd_ellipse])


def random_point_ellipse(cx, cy, rx, ry):
    angle = np.random.random()*2*np.pi
    dist = np.random.random()

    x = cx + np.sqrt(dist) * np.cos(angle) * rx
    y = cy + np.sqrt(dist) * np.sin(angle) * ry
    return np.array([x, y, 0])