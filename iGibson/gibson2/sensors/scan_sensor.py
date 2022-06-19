from gibson2.sensors.sensor_base import BaseSensor
from gibson2.sensors.dropout_sensor_noise import DropoutSensorNoise
from gibson2.utils.constants import OccupancyGridState
import numpy as np
from transforms3d.quaternions import quat2mat
import pybullet as p
import cv2


class ScanSensor(BaseSensor):
    """
    1D LiDAR scanner sensor and occupancy grid sensor
    """

    def __init__(self, env, modalities):
        super(ScanSensor, self).__init__(env)
        self.modalities = modalities
        self.scan_noise_rate = self.config.get('scan_noise_rate', 0.0)
        self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
        self.n_vertical_beams = self.config.get('n_vertical_beams', 1)
        assert self.n_vertical_beams == 1, 'scan can only handle one vertical beam for now'
        self.laser_linear_range = self.config.get(
            'laser_linear_range', 10.0)
        self.laser_angular_range = self.config.get(
            'laser_angular_range', 180.0)
        self.min_laser_dist = self.config.get('min_laser_dist', 0.05)
        self.laser_link_name = self.config.get(
            'laser_link_name', 'scan_link')
        self.noise_model = DropoutSensorNoise(env)
        self.noise_model.set_noise_rate(self.scan_noise_rate)
        self.noise_model.set_noise_value(1.0)

        self.laser_pose = env.robots[0].parts[self.laser_link_name].get_pose()
        self.base_pose = env.robots[0].parts['base_link'].get_pose()

        if 'occupancy_grid' in self.modalities:
            self.grid_resolution = self.config.get('grid_resolution', 128)
            self.occupancy_range = self.config.get('occupancy_range', 5)  # m
            self.robot_footprint_radius = self.config.get(
                'robot_footprint_radius', 0.32)
            self.robot_footprint_radius_in_map = int(
                self.robot_footprint_radius / self.occupancy_range *
                self.grid_resolution)

    def get_local_occupancy_grid(self, scan):
        """
        Get local occupancy grid based on current 1D scan

        :param: 1D LiDAR scan
        :return: local occupancy grid
        """
        laser_linear_range = self.laser_linear_range
        laser_angular_range = self.laser_angular_range
        min_laser_dist = self.min_laser_dist

        laser_angular_half_range = laser_angular_range / 2.0

        angle = np.arange(
            -np.radians(laser_angular_half_range),
            np.radians(laser_angular_half_range),
            np.radians(laser_angular_range) / self.n_horizontal_rays
        )
        unit_vector_laser = np.array(
            [[np.cos(ang), np.sin(ang), 0.0] for ang in angle])

        scan_laser = unit_vector_laser * \
            (scan * (laser_linear_range - min_laser_dist) + min_laser_dist)

        laser_translation = self.laser_pose[:3]
        laser_rotation = quat2mat(
            [self.laser_pose[6], self.laser_pose[3], self.laser_pose[4], self.laser_pose[5]])
        scan_world = laser_rotation.dot(scan_laser.T).T + laser_translation

        base_translation = self.base_pose[:3]
        base_rotation = quat2mat(
            [self.base_pose[6], self.base_pose[3], self.base_pose[4], self.base_pose[5]])
        scan_local = base_rotation.T.dot((scan_world - base_translation).T).T
        scan_local = scan_local[:, :2]
        scan_local = np.concatenate(
            [np.array([[0, 0]]), scan_local, np.array([[0, 0]])], axis=0)

        # flip y axis
        scan_local[:, 1] *= -1

        occupancy_grid = np.zeros(
            (self.grid_resolution, self.grid_resolution)).astype(np.uint8)

        occupancy_grid.fill(int(OccupancyGridState.UNKNOWN * 2.0))
        scan_local_in_map = scan_local / self.occupancy_range * \
            self.grid_resolution + (self.grid_resolution / 2)
        scan_local_in_map = scan_local_in_map.reshape(
            (1, -1, 1, 2)).astype(np.int32)
        for i in range(scan_local_in_map.shape[1]):
            cv2.circle(img=occupancy_grid,
                       center=(scan_local_in_map[0, i, 0, 0],
                               scan_local_in_map[0, i, 0, 1]),
                       radius=2,
                       color=int(OccupancyGridState.OBSTACLES * 2.0),
                       thickness=-1)
        cv2.fillPoly(img=occupancy_grid,
                     pts=scan_local_in_map,
                     color=int(OccupancyGridState.FREESPACE * 2.0),
                     lineType=1)
        cv2.circle(img=occupancy_grid,
                   center=(self.grid_resolution // 2,
                           self.grid_resolution // 2),
                   radius=int(self.robot_footprint_radius_in_map),
                   color=int(OccupancyGridState.FREESPACE * 2.0),
                   thickness=-1)

        return occupancy_grid[:, :, None].astype(np.float32) / 2.0

    def get_obs(self, env):
        """
        Get current LiDAR sensor reading and occupancy grid (optional)

        :return: LiDAR sensor reading and local occupancy grid, normalized to [0.0, 1.0]
        """
        laser_angular_half_range = self.laser_angular_range / 2.0
        if self.laser_link_name not in env.robots[0].parts:
            raise Exception('Trying to simulate LiDAR sensor, but laser_link_name cannot be found in the robot URDF file. Please add a link named laser_link_name at the intended laser pose. Feel free to check out assets/models/turtlebot/turtlebot.urdf and examples/configs/turtlebot_p2p_nav.yaml for examples.')
        laser_pose = env.robots[0].parts[self.laser_link_name].get_pose()
        angle = np.arange(
            -laser_angular_half_range / 180 * np.pi,
            laser_angular_half_range / 180 * np.pi,
            self.laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays)
        unit_vector_local = np.array(
            [[np.cos(ang), np.sin(ang), 0.0] for ang in angle])
        transform_matrix = quat2mat(
            [laser_pose[6], laser_pose[3], laser_pose[4], laser_pose[5]])  # [x, y, z, w]
        unit_vector_world = transform_matrix.dot(unit_vector_local.T).T

        start_pose = np.tile(laser_pose[:3], (self.n_horizontal_rays, 1))
        start_pose += unit_vector_world * self.min_laser_dist
        end_pose = laser_pose[:3] + unit_vector_world * self.laser_linear_range
        results = p.rayTestBatch(start_pose, end_pose, 6)  # numThreads = 6

        # hit fraction = [0.0, 1.0] of self.laser_linear_range
        hit_fraction = np.array([item[2] for item in results])
        hit_fraction = self.noise_model.add_noise(hit_fraction)
        scan = np.expand_dims(hit_fraction, 1)

        state = {}
        state['scan'] = scan.flatten()
        if 'occupancy_grid' in self.modalities:
            state['occupancy_grid'] = self.get_local_occupancy_grid(scan)
        return state
