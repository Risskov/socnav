from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance


class PersonalSpaceReward(BaseRewardFunction):
    """
    Personal space reward
    Negative reward for violating personal space of pedestrians
    """

    def __init__(self, config):
        super(PersonalSpaceReward, self).__init__(config)
        self.personal_space_reward = self.config.get(
            'personal_space_reward', -1.0
        )
        self.orca_radius = self.config.get('orca_radius', 0.5)

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        personal_space_violation = False
        robot_pos = env.robots[0].get_position()[:2]
        for ped in task.pedestrians:
            ped_pos = ped.get_position()[:2]
            if l2_distance(robot_pos, ped_pos) < self.orca_radius:
                personal_space_violation = True
                #print("Personal space violation!")
                break
        reward = self.personal_space_reward if personal_space_violation else 0.0
        return reward
