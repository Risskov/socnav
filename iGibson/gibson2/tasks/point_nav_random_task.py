from gibson2.tasks.point_nav_fixed_task import PointNavFixedTask
from gibson2.utils.utils import l2_distance
from gibson2.utils.start_stop_areas import sample_new_pos
import pybullet as p
import logging
import numpy as np


class PointNavRandomTask(PointNavFixedTask):
    """
    Point Nav Random Task
    The goal is to navigate to a random goal position
    """

    def __init__(self, env):
        super(PointNavRandomTask, self).__init__(env)
        self.target_dist_min = self.config.get('target_dist_min', 1.0)
        self.target_dist_max = self.config.get('target_dist_max', 10.0)

    def sample_initial_pose_and_target_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        #_, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        initial_pos = sample_new_pos(env)
        max_trials = 100
        dist = 0.0
        for _ in range(max_trials):
            #_, target_pos = env.scene.get_random_point(floor=self.floor_num)
            target_pos = sample_new_pos(env)
            if env.scene.build_graph:
                _, dist = env.scene.get_shortest_path(
                    self.floor_num,
                    initial_pos[:2],
                    target_pos[:2], entire_path=False)
            else:
                dist = l2_distance(initial_pos, target_pos)
            if self.target_dist_min < dist < self.target_dist_max:
                break
        if not (self.target_dist_min < dist < self.target_dist_max):
            print("WARNING: Failed to sample initial and target positions")
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        return initial_pos, initial_orn, target_pos

    def reset_scene(self, env):
        """
        Task-specific scene reset: get a random floor number first

        :param env: environment instance
        """
        self.floor_num = env.scene.get_random_floor()
        super(PointNavRandomTask, self).reset_scene(env)

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.

        :param env: environment instance
        """
        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            initial_pos, initial_orn, target_pos = \
                self.sample_initial_pose_and_target_pos(env)
            reset_success = env.test_valid_position(
                env.robots[0], initial_pos, initial_orn) and \
                env.test_valid_position(
                    env.robots[0], target_pos)
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        # removed cached state to prevent memory leak
        p.removeState(state_id)

        self.target_pos = target_pos
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        # self.initial_pos = np.array([-4, 0, 0])
        #self.initial_orn = np.array([0,0,np.pi /3.9 ])
        super(PointNavRandomTask, self).reset_agent(env)
