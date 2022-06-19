from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.simulator import Simulator
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
from gibson2.render.profiler import Profiler
import gibson2
import os

def main():
    config = parse_config(os.path.join(gibson2.example_config_path, 'turtlebot_demo.yaml'))
    settings = MeshRendererSettings()
    s = Simulator(mode='gui',
                  image_width=256,
                  image_height=256,
                  rendering_settings=settings)

    scene = StaticIndoorScene('Rs',
                              build_graph=True,
                              pybullet_load_texture=True)
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    for i in range(10000):
        with Profiler('Simulator step'):
            turtlebot.apply_action([0.1, -0.1])
            s.step()
            lidar = s.renderer.get_lidar_all()
            print(lidar.shape)
            # TODO: visualize lidar scan

    s.disconnect()


if __name__ == '__main__':
    main()
