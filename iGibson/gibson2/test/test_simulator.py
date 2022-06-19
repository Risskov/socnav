from gibson2.scenes.stadium_scene import StadiumScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator

from gibson2.utils.assets_utils import download_assets


def test_simulator():
    download_assets()
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    obj = YCBObject('006_mustard_bottle')

    for i in range(10):
        s.import_object(obj)

    obj = YCBObject('002_master_chef_can')
    for i in range(10):
        s.import_object(obj)

    for i in range(1000):
        s.step()
    s.disconnect()
