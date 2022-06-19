import logging
import pickle
import networkx as nx
import cv2
from PIL import Image
import numpy as np
from gibson2.objects.articulated_object import ArticulatedObject, URDFObject
from gibson2.utils.utils import l2_distance, get_transform_from_xyz_rpy, quatXYZWFromRotMat
from gibson2.utils.assets_utils import get_scene_path, get_texture_file, get_ig_scene_path
import pybullet_data
import pybullet as p
import os
from gibson2.scenes.wall_scene import WallScene

class WallLargeScene(WallScene):
    """
    Static wall scene class for iGibson.
    Contains the functionalities for navigation such as shortest path computation
    """

    def __init__(self,
                 scene_id,
                 trav_map_resolution=0.1,
                 trav_map_default_resolution=0.01,
                 trav_map_erosion=2,
                 trav_map_type='with_obj',
                 build_graph=True,
                 num_waypoints=10,
                 waypoint_resolution=0.2,
                 pybullet_load_texture=False,
                 ):
        """
        Load a building scene and compute traversability

        :param scene_id: Scene id
        :param trav_map_resolution: desired traversability map resolution
        :param trav_map_default_resolution: original traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param trav_map_type: type of traversability map, with_obj | no_obj
        :param build_graph: build connectivity graph
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        :param pybullet_load_texture: whether to load texture into pybullet. This is for debugging purpose only and does not affect robot's observations
        """
        super(WallLargeScene, self).__init__(
            scene_id,
            trav_map_resolution,
            trav_map_default_resolution,
            trav_map_erosion,
            trav_map_type,
            build_graph,
            num_waypoints,
            waypoint_resolution,
            pybullet_load_texture,
        )
        self.walls_id = []
        self.objects_by_name = {}
        logging.info("WallScene scene: {}".format(scene_id))

    def load_border_walls(self):
        """
        Load surrounding walls
        """
        wall_coll1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([10, 0.2, 1]))
        wall_coll2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([0.2, 10, 1]))
        wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                  basePosition=[0, 10.2, 0])
        wall2 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                  basePosition=[0, -10.2, 0])
        wall3 = p.createMultiBody(baseCollisionShapeIndex=wall_coll2,
                                  basePosition=[10.2, 0, 0])
        wall4 = p.createMultiBody(baseCollisionShapeIndex=wall_coll2,
                                  basePosition=[-10.2, 0, 0])
        self.walls_id = [wall1, wall2, wall3, wall4]

    def load_walls(self):
        """
        Load the walls
        """
        if self.scene_id == "straight":
            wall_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([5, 0.2, 1]))
            wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[0, 2.2, 0])
            wall2 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[0, -2.2, 0])
            self.walls_id += [wall1, wall2]
        elif self.scene_id == "straight_narrow_large":
            wall_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([10, 4.5, 1]))
            wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[0, 5.5, 0])
            wall2 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[0, -5.5, 0])
            self.walls_id += [wall1, wall2]
        elif self.scene_id == "H_large":
            wall_coll1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([10, 5, 1]))
            wall_coll2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([3.5, 1.5, 1]))
            wall_coll3 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([5.5, 1.5, 1]))
            wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                      basePosition=[0, 5, 0])
            wall2 = p.createMultiBody(baseCollisionShapeIndex=wall_coll2,
                                      basePosition=[-6.5, -5.5, 0])
            wall3 = p.createMultiBody(baseCollisionShapeIndex=wall_coll3,
                                      basePosition=[4.5, -5.5, 0])
            self.walls_id += [wall1, wall2, wall3]
        elif self.scene_id == "S":
            wall_coll1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([3, 0.5, 1]))
            wall_coll2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([3, 0.75, 1]))
            wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                      basePosition=[2, 2, 0])
            wall2 = p.createMultiBody(baseCollisionShapeIndex=wall_coll2,
                                      basePosition=[-1.5, -2, 0])
            self.walls_id += [wall1, wall2]
        elif self.scene_id == "O":
            wall_coll1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([2, 2.25, 1]))
            wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                      basePosition=[0, -0.25, 0])
        elif self.scene_id == "bend":
            wall_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([3, 0.2, 1]))
            wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[-2, 0, 0])
            self.walls_id += [wall1]
        elif self.scene_id == "X_large":
            wall_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([4.5, 4.5, 1]))
            wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[5.5, 5.5, 0])
            wall2 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[-5.5, 5.5, 0])
            wall3 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[-5.5, -5.5, 0])
            wall4 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[5.5, -5.5, 0])
            self.walls_id += [wall1, wall2, wall3, wall4]
