import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

class ValueNetwork(nn.Module):
    def __init__(self, scan_size, goal_size, pedestrian_size, hidden_size=256, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        channels = 1
        self.scan_size = scan_size
        self.goal_size = goal_size
        self.pedestrian_size = pedestrian_size

        self.conv1 = nn.Conv1d(channels, channels, 3)
        self.conv2 = nn.Conv1d(channels, channels, 3)
        self.conv3 = nn.Conv1d(channels, channels, 3)
        self.linear_scan = nn.Linear(scan_size - 2*3, hidden_size)

        self.linear_goal = nn.Linear(goal_size, hidden_size)
        self.linear_pedestrian = nn.Linear(pedestrian_size, hidden_size)

        self.linear_cat = nn.Linear(hidden_size*3, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 1)

        self.linear_out.weight.data.uniform_(-init_w, init_w)
        self.linear_out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        scan, goal, ped = torch.split(state, [self.scan_size, self.goal_size, self.pedestrian_size], dim=2)
        x = F.relu(self.conv1(scan))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear_scan(x))
        y = F.relu(self.linear_goal(goal))
        z = F.relu(self.linear_pedestrian(ped))
        out = torch.cat((x, y, z), dim=2)
        out = F.relu(self.linear_cat(out))
        out = self.linear_out(out)

        return out

class SoftQNetwork(nn.Module):
    def __init__(self, scan_size, goal_size, pedestrian_size, action_size, hidden_size=256, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        channels = 1
        self.scan_size = scan_size
        self.goal_size = goal_size
        self.pedestrian_size = pedestrian_size
        self.action_size = action_size

        self.conv1 = nn.Conv1d(channels, channels, 3)
        self.conv2 = nn.Conv1d(channels, channels, 3)
        self.conv3 = nn.Conv1d(channels, channels, 3)
        self.linear_scan = nn.Linear(scan_size - 2 * 3, hidden_size)

        self.linear_goal = nn.Linear(goal_size, hidden_size)
        self.linear_pedestrian = nn.Linear(pedestrian_size, hidden_size)
        self.linear_action = nn.Linear(action_size, hidden_size)

        self.linear_cat = nn.Linear(hidden_size * 4, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 1)

        self.linear_out.weight.data.uniform_(-init_w, init_w)
        self.linear_out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        scan, goal, ped = torch.split(state, [self.scan_size, self.goal_size, self.pedestrian_size], dim=2)
        x = F.relu(self.conv1(scan))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear_scan(x))
        y = F.relu(self.linear_goal(goal))
        z = F.relu(self.linear_pedestrian(ped))
        a = F.relu(self.linear_action(action))
        out = torch.cat((x, y, z, a), dim=2)
        out = F.relu(self.linear_cat(out))
        out = self.linear_out(out)

        return out

class PolicyNetwork(nn.Module):
    def __init__(self, scan_size, goal_size, pedestrian_size, action_size, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        channels = 1
        self.scan_size = scan_size
        self.goal_size = goal_size
        self.pedestrian_size = pedestrian_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.conv1 = nn.Conv1d(channels, channels, 3)
        self.conv2 = nn.Conv1d(channels, channels, 3)
        self.conv3 = nn.Conv1d(channels, channels, 3)
        self.linear_scan = nn.Linear(scan_size - 2 * 3, hidden_size)

        self.linear_goal = nn.Linear(goal_size, hidden_size)
        self.linear_pedestrian = nn.Linear(pedestrian_size, hidden_size)

        self.linear_cat = nn.Linear(hidden_size * 3, hidden_size)

        self.linear_mean = nn.Linear(hidden_size, action_size)
        self.linear_mean.weight.data.uniform_(-init_w, init_w)
        self.linear_mean.bias.data.uniform_(-init_w, init_w)

        self.linear_log_std = nn.Linear(hidden_size, action_size)
        self.linear_log_std.weight.data.uniform_(-init_w, init_w)
        self.linear_log_std.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        scan, goal, ped = torch.split(state, [self.scan_size, self.goal_size, self.pedestrian_size], dim=-1)
        x = F.relu(self.conv1(scan))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear_scan(x))
        y = F.relu(self.linear_goal(goal))
        z = F.relu(self.linear_pedestrian(ped))
        out = torch.cat((x, y, z), dim=-1)
        out = F.relu(self.linear_cat(out))

        mean = self.linear_mean(out)
        log_std = self.linear_log_std(out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std


    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]

"""
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std*z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        #print(f"mu: {mean}, sigma: {std}")
        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std*z)
        action = action.cpu()
        return action[0]
"""