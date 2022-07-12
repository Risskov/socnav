from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance


class WaypointReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(WaypointReward, self).__init__(config)
        self.waypoint_reward = self.config.get(
            'waypoint_reward', 0.5
        )
        self.dist_tol = self.config.get('dist_tol', 0.5)

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and any waypoint
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        waypoint_reached = False
        robot_pos = env.robots[0].get_position()[:2]
        for i, waypoint in enumerate(task.waypoints):
                if l2_distance(robot_pos, waypoint) < self.dist_tol:
                    waypoint_reached = True
                    task.update_waypoints(i+1) #test
                    break
        reward = self.waypoint_reward if waypoint_reached else 0.0
        return reward
