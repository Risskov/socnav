from gibson2.reward_functions.reward_function_base import BaseRewardFunction


class TimestepReward(BaseRewardFunction):
    """
    Timestep reward
    Penalize being slow. Typically timestep_reward_weight is negative.
    """

    def __init__(self, config):
        super(TimestepReward, self).__init__(config)
        self.timestep_reward_weight = self.config.get(
            'timestep_reward_weight', -0.1
        )

    def get_reward(self, task, env):
        """
        Reward is self.timestep_reward_weight every timestep
        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        return self.timestep_reward_weight
