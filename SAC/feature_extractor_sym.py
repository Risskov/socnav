import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "scan":
                self.scan_dim = subspace.shape[0]
                extractors[key] = nn.Sequential(nn.Linear(self.scan_dim, 256), nn.ReLU())
                total_concat_size += 256
                print("Scan dim: ", self.scan_dim)
            elif key == "goal":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 256), nn.ReLU())
                total_concat_size += 256
            elif key == "waypoints":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 256), nn.ReLU())
                total_concat_size += 256
                print("Waypoint layer size: ", subspace.shape[0])
            elif key == "pedestrians":
                extractors[key] = nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                                nn.Linear(64, 128), nn.ReLU(),
                                                nn.Linear(128, 256), nn.ReLU(),
                                                )
                total_concat_size += 256
                print("Ped layer size: ", subspace.shape[0])
            elif key == "task_obs":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 256), nn.ReLU())
                total_concat_size += 256
            print("Added layer: ", key)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size
        #self.lstm = nn.LSTM(input_size=total_concat_size, hidden_size=total_concat_size, num_layers=3)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():

            if key == "pedestrians":
                x = extractor(observations[key])
                #x = torch.max(x, -2, keepdim=False)[0]
                x = torch.mean(x, -2, keepdim=False)
                #x = x.view(-1, 256)
                encoded_tensor_list.append(x)
            else:
                encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)

class LargeMeanpoolExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(LargeMeanpoolExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "scan":
                self.scan_dim = subspace.shape[0]
                extractors[key] = nn.Sequential(nn.Linear(self.scan_dim, 256), nn.ReLU(),
                                                #nn.Linear(256, 256), nn.ReLU(),
                                                #nn.Linear(256, 256), nn.ReLU()
                                                )
                total_concat_size += 256
                print("Scan dim: ", self.scan_dim)
            elif key == "goal":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 256), nn.ReLU())
                total_concat_size += 256
            elif key == "waypoints":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 256), nn.ReLU())
                total_concat_size += 256
                print("Waypoint layer size: ", subspace.shape[0])
            elif key == "pedestrians":
                extractors[key] = nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                                nn.Linear(64, 128), nn.ReLU(),
                                                nn.Linear(128, 256), nn.ReLU(),
                                                )
                total_concat_size += 256
                print("Ped layer size: ", subspace.shape[0])
            elif key == "task_obs":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 256), nn.ReLU())
                total_concat_size += 256
            print("Added layer: ", key)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():

            if key == "pedestrians":
                x = extractor(observations[key])
                #x = torch.max(x, -2, keepdim=False)[0]
                x = torch.mean(x, -2, keepdim=False)
                encoded_tensor_list.append(x)
            else:
                encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)

class LargeMaxpoolExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(LargeMaxpoolExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}
        dim = 256
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "scan":
                self.scan_dim = subspace.shape[0]
                extractors[key] = nn.Sequential(nn.Linear(self.scan_dim, dim), nn.ReLU(),
                                                #nn.Linear(dim, dim), nn.ReLU(),
                                                #nn.Linear(dim, dim), nn.ReLU()
                                                )
                total_concat_size += dim
                print("Scan dim: ", self.scan_dim)
            elif key == "goal":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], dim), nn.ReLU())
                total_concat_size += dim
            elif key == "waypoints":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], dim), nn.ReLU())
                total_concat_size += dim
                print("Waypoint layer size: ", subspace.shape[0])
            elif key == "pedestrians":
                extractors[key] = nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                                nn.Linear(64, 128), nn.ReLU(),
                                                nn.Linear(128, dim), nn.ReLU(),
                                                )
                total_concat_size += dim
                print("Ped layer size: ", subspace.shape[0])
            elif key == "task_obs":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], dim), nn.ReLU())
                total_concat_size += dim
            print("Added layer: ", key)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():

            if key == "pedestrians":
                x = extractor(observations[key])
                x = torch.max(x, -2, keepdim=False)[0]
                #x = torch.mean(x, -2, keepdim=False)
                encoded_tensor_list.append(x)
            else:
                encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
