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
from gibson2.scenes.indoor_scene import IndoorScene

class WallScene(IndoorScene):
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
        super(WallScene, self).__init__(
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

    def load_floor_metadata(self):
        """
        Load floor metadata
        """
        floor_height_path = os.path.join(
            get_scene_path(self.scene_id), 'floors.txt')
        if not os.path.isfile(floor_height_path):
            raise Exception(
                'floor_heights.txt cannot be found in model: {}'.format(self.scene_id))
        with open(floor_height_path, 'r') as f:
            self.floor_heights = sorted(list(map(float, f.readlines())))
            logging.debug('Floors {}'.format(self.floor_heights))

    def load_scene_mesh(self):
        """
        Load scene mesh
        """
        filename = os.path.join(get_scene_path(
            self.scene_id), "mesh_z_up_downsampled.obj")
        if not os.path.isfile(filename):
            filename = os.path.join(get_scene_path(
                self.scene_id), "mesh_z_up.obj")

        collision_id = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=filename,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        if self.pybullet_load_texture:
            visual_id = p.createVisualShape(
                p.GEOM_MESH,
                fileName=filename)
        else:
            visual_id = -1

        self.mesh_body_id = p.createMultiBody(
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id)
        p.changeDynamics(self.mesh_body_id, -1, lateralFriction=1)

        if self.pybullet_load_texture:
            texture_filename = get_texture_file(filename)
            if texture_filename is not None:
                texture_id = p.loadTexture(texture_filename)
                p.changeVisualShape(
                    self.mesh_body_id,
                    -1,
                    textureUniqueId=texture_id)

    def load_border_walls(self):

        """
        Load surrounding walls
        """
        wall_coll1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([5, 0.2, 1]))
        wall_coll2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([0.2, 5, 1]))
        wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                  basePosition=[0, 5.2, 0])
        wall2 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                  basePosition=[0, -5.2, 0])
        wall3 = p.createMultiBody(baseCollisionShapeIndex=wall_coll2,
                                  basePosition=[5.2, 0, 0])
        wall4 = p.createMultiBody(baseCollisionShapeIndex=wall_coll2,
                                  basePosition=[-5.2, 0, 0])
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
        elif self.scene_id == "straight_narrow":
            wall_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([5, 2, 1]))
            wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[0, 3, 0])
            wall2 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[0, -3, 0])
            self.walls_id += [wall1, wall2]
        elif self.scene_id == "H":
            wall_coll1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([1.5, 1, 1]))
            wall_coll2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([2.5, 1, 1]))
            wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                      basePosition=[-3.5, 0, 0])
            wall2 = p.createMultiBody(baseCollisionShapeIndex=wall_coll2,
                                      basePosition=[2.5, 0, 0])
            self.walls_id += [wall1, wall2]
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
        elif self.scene_id == "cross_narrow":
            wall_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([2, 2, 1]))
            wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[3, 3, 0])
            wall2 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[-3, 3, 0])
            wall3 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[-3, -3, 0])
            wall4 = p.createMultiBody(baseCollisionShapeIndex=wall_coll,
                                      basePosition=[3, -3, 0])
            self.walls_id += [wall1, wall2, wall3, wall4]
        elif self.scene_id == "cross":
            wall_coll1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([1.5, 0.2, 1]))
            wall_coll2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([0.2, 1.5, 1]))
            vis1 = p.createVisualShape(p.GEOM_BOX, halfExtents=np.array([1.5, 0.2, 1]), rgbaColor=np.array([0, 0, 1, 1]))
            vis2 = p.createVisualShape(p.GEOM_BOX, halfExtents=np.array([0.2, 1.5, 1]), rgbaColor=np.array([0, 0, 1, 1]))
            wall1 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                      baseVisualShapeIndex=vis1,
                                      basePosition=[-3.5, 2.2, 0])
            wall2 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                      baseVisualShapeIndex=vis1,
                                      basePosition=[-3.5, -2.2, 0])
            wall3 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                      baseVisualShapeIndex=vis1,
                                      basePosition=[3.5, 2.2, 0])
            wall4 = p.createMultiBody(baseCollisionShapeIndex=wall_coll1,
                                      baseVisualShapeIndex=vis1,
                                      basePosition=[3.5, -2.2, 0])
            wall5 = p.createMultiBody(baseCollisionShapeIndex=wall_coll2,
                                      baseVisualShapeIndex=vis2,
                                      basePosition=[2.2, -3.5, 0])
            wall6 = p.createMultiBody(baseCollisionShapeIndex=wall_coll2,
                                      baseVisualShapeIndex=vis2,
                                      basePosition=[-2.2, -3.5, 0])
            wall7 = p.createMultiBody(baseCollisionShapeIndex=wall_coll2,
                                      baseVisualShapeIndex=vis2,
                                      basePosition=[2.2, 3.5, 0])
            wall8 = p.createMultiBody(baseCollisionShapeIndex=wall_coll2,
                                      baseVisualShapeIndex=vis2,
                                      basePosition=[-2.2, 3.5, 0])
            self.walls_id += [wall1, wall2, wall3, wall4, wall5, wall6, wall7, wall8]

    def load_floor_plane(self):
        """
        Load floor plane
        """
        plane_name = os.path.join(
            pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        floor_body_id = p.loadMJCF(plane_name)[0]
        self.floor_body_ids.append(floor_body_id)

    def load(self):
        """
        Load the scene (including scene mesh and floor plane) into pybullet
        """
        self.load_border_walls()
        self.load_walls()
        self.load_floor_plane()
        self.load_trav_map(get_scene_path(self.scene_id))

        ids = self.floor_body_ids
        for walls in self.walls_id:
            ids.append(walls)
        # return [self.walls_id] + self.floor_body_ids
        return ids

    def get_random_floor(self):
        """
        Get a random floor

        :return: random floor number
        """
        return np.random.randint(0, high=len(self.floor_heights))

    def reset_floor(self, floor=0, additional_elevation=0.02, height=None):
        """
        Resets the floor plane to a new floor

        :param floor: Integer identifying the floor to move the floor plane to
        :param additional_elevation: Additional elevation with respect to the height of the floor
        :param height: Alternative parameter to control directly the height of the ground plane
        """
        height = height if height is not None \
            else self.floor_heights[floor] + additional_elevation
        p.resetBasePositionAndOrientation(self.floor_body_ids[0],
                                          posObj=[0, 0, height],
                                          ornObj=[0, 0, 0, 1])

    def get_floor_height(self, floor=0):
        """
        Return the current floor height (in meter)

        :return: current floor height
        """
        return self.floor_heights[floor]
