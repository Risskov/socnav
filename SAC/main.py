import logging
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from models import ValueNetwork, SoftQNetwork, PolicyNetwork
from replay_memory import ReplayBuffer
from gibson2.envs.igibson_env import iGibsonEnv
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import pybullet as p

def plot(frame_idx, data, text):
    clear_output(True)
    plt.title(f'frame {frame_idx}. {text}: {data[-1]}')
    plt.plot(data)
    plt.show()

def plot_lidar(scan):
    step = 2*np.pi/360
    #angles = np.linspace(-29*step, 210*step, 228)
    angles = np.linspace(0, 2*np.pi, 360)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(angles, scan)
    plt.show()

def soft_q_update(batch_size,
                  gamma=0.99,
                  mean_lambda=1e-3,
                  std_lambda=1e-3,
                  z_lambda=0.0,
                  soft_tau=1e-2,
                  ):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).unsqueeze(1).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(1).to(device)
    action = torch.FloatTensor(action).unsqueeze(1).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).unsqueeze(2).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).unsqueeze(2).to(device)

    expected_q_value = soft_q_net1(state, action)
    expected_value = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

    target_value = target_value_net(next_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = soft_q_criterion1(expected_q_value, next_q_value.detach())

    expected_new_q_value = soft_q_net1(state, new_action)
    next_value = expected_new_q_value - log_prob
    value_loss = value_criterion(expected_value, next_value.detach())

    log_prob_target = expected_new_q_value - expected_value
    policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

    mean_loss = mean_lambda * mean.pow(2).mean()
    std_loss = std_lambda * log_std.pow(2).mean()
    z_loss = z_lambda * z.pow(2).sum(1).mean()

    policy_loss += mean_loss + std_loss + z_loss

    soft_q_optimizer1.zero_grad()
    q_value_loss.backward()
    soft_q_optimizer1.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    return policy_loss.item()

def update(batch_size, gamma=0.99, soft_tau=1e-2, ):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).unsqueeze(1).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(1).to(device)
    action = torch.FloatTensor(action).unsqueeze(1).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).unsqueeze(2).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).unsqueeze(2).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value = value_net(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

    # Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())
    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()
    #print("Log: ", log_prob.shape)

    # Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - log_prob
    value_loss = value_criterion(predicted_value, target_value_func.detach())
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # Training Policy Function
    policy_loss = (log_prob - predicted_new_q_value).mean()
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
    return policy_loss.item() #q_value_loss2.item() #policy_loss.item()

#def main(selection="user", headless=False, short_exec=False):
"""
Social Navigation
"""

headless = False
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
config_filename = "configs/my_env.yaml"
config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

env = iGibsonEnv(config_file=config_data, mode="gui" if not headless else "headless")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#sim = env.simulator


# SAC setup
action_dim = env.action_space.shape[0]
scan_dim = config_data['n_horizontal_rays']
goal_dim = 2
pedestrian_dim = 2
hidden_dim = 256

value_net = ValueNetwork(scan_dim, goal_dim, pedestrian_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(scan_dim, goal_dim, pedestrian_dim, hidden_dim).to(device)

soft_q_net1 = SoftQNetwork(scan_dim, goal_dim, pedestrian_dim, action_dim, hidden_dim).to(device)
soft_q_net2 = SoftQNetwork(scan_dim, goal_dim, pedestrian_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(scan_dim, goal_dim, pedestrian_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

value_criterion = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

value_lr = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

replay_buffer_size = 100000 #0
replay_buffer = ReplayBuffer(replay_buffer_size)

max_frames = 500001
max_steps = 500
frame_index = 0
rewards = []
losses = []
batch_size = 128
print("Loading complete")
while frame_index < max_frames:
    obs = env.reset()
    scan = obs['scan']
    goal = obs['task_obs'][0:2]
    ped = obs['task_obs'][2:4]
    print(obs)
    state = np.concatenate((scan, goal, ped), axis=None)
    state = np.around(state, decimals=3)
    episode_reward = 0
    #print("New episode")
    #for step in range(max_steps):
    while True:
        if frame_index > 1000:
            action = policy_net.get_action(state)
            action = np.around(action, decimals=3)
            next_state, reward, done, _ = env.step(action)
            #print("Action: ", action)
        else:
            action = [0., 0.]
            #action = env.action_space.sample()
            action = np.around(action, decimals=3)
            next_state, reward, done, _ = env.step(action)
            #print("Reward: ", reward)

        scan = next_state['scan']
        goal = next_state['task_obs'][0:2]
        ped = next_state['task_obs'][2:4]
        print(next_state)
        next_state = np.concatenate((scan, goal, ped), axis=None)
        next_state = np.around(next_state, decimals=3)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        frame_index += 1
        if frame_index % 10 == 100:
            print("Action: ", action)
            plot_lidar(scan)
        if len(replay_buffer) > batch_size:
            #loss = soft_q_update(batch_size)
            loss = update(batch_size)
            losses.append(loss)
        if frame_index % 10000 == 0 and len(rewards):
            print("Reward length: ", len(rewards))
            plot(frame_index, rewards, "reward")
            plot(frame_index, losses, "loss")
        if done:
            #print("Info: ", info)
            break

    rewards.append(episode_reward)
"""
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
"""