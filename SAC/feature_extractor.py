import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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
                # extractors[key] = nn.Sequential(nn.Conv1d(1, 1, 3), nn.ReLU(),
                #                                 nn.Conv1d(1, 1, 3), nn.ReLU(),
                #                                 nn.Conv1d(1, 1, 3), nn.ReLU(),
                #                                 nn.Linear(self.scan_dim - 2*3, 256), nn.ReLU())
                extractors[key] = nn.Sequential(nn.Linear(self.scan_dim, 256), nn.ReLU(),
                                                nn.Linear(256, 256), nn.ReLU(),
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
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 256), nn.ReLU())
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
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
