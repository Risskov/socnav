from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
import os
import gibson2
import GPUtil
import time
from gibson2.utils.assets_utils import download_assets
from gibson2.utils.assets_utils import get_ig_model_path
from PIL import Image
import matplotlib.pyplot as plt

def test_render_pbr():
    hdr_texture = os.path.join(gibson2.ig_dataset_path, 'scenes', 'background', 'quattro_canti_4k.hdr')
    model_path = os.path.join(get_ig_model_path('sink', 'sink_1'), 'shape', 'visual')
    settings = MeshRendererSettings(msaa=True, enable_shadow=True, env_texture_filename=hdr_texture, env_texture_filename3=hdr_texture)
    renderer = MeshRenderer(width=1024, height=1024, vertical_fov=90, rendering_settings=settings)
    renderer.set_light_position_direction([0,0,10], [0,0,0])
    i = 0

    for fn in os.listdir(model_path):
        if fn.endswith('obj'):
            renderer.load_object(os.path.join(model_path, fn), scale=[1, 1, 1])
            renderer.add_instance(i)
            i += 1
            renderer.instances[-1].use_pbr = True
            renderer.instances[-1].use_pbr_mapping = True


    renderer.set_camera([1.5,1.5,1.5], [0,0,0], [0, 0, 1], cache=True)
    frame = renderer.render(modes=('rgb', 'normal'))
    Image.fromarray((255*np.concatenate(frame, axis=1)[:,:,:3]).astype(np.uint8)).save('test_render.png')
    
    renderer.set_camera([1.49,1.49,1.49], [0,0.05,0.05], [0, 0, 1], cache=True) # simulate camera movement
    frame = renderer.render(modes=('optical_flow', 'scene_flow'))
    plt.subplot(1,2,1)
    plt.imshow(np.abs(frame[0][:,:,:3]) / np.max(np.abs(frame[0][:,:,:3])))
    plt.subplot(1,2,2)
    plt.imshow(np.abs(frame[1][:,:,:3]) / np.max(np.abs(frame[1][:,:,:3])))
    plt.savefig('test_render_flow.png')
    renderer.release()


def test_render_pbr_optimized():
    hdr_texture = os.path.join(gibson2.ig_dataset_path, 'scenes', 'background', 'quattro_canti_4k.hdr')
    model_path = os.path.join(get_ig_model_path('sink', 'sink_1'), 'shape', 'visual')
    settings = MeshRendererSettings(msaa=True, enable_shadow=True, env_texture_filename=hdr_texture, env_texture_filename3=hdr_texture,
        optimized=True)
    renderer = MeshRenderer(width=1024, height=1024, vertical_fov=90, rendering_settings=settings)
    renderer.set_light_position_direction([0,0,10], [0,0,0])
    i = 0

    for fn in os.listdir(model_path):
        if fn.endswith('obj'):
            renderer.load_object(os.path.join(model_path, fn), scale=[1, 1, 1])
            renderer.add_instance(i)
            i += 1
            renderer.instances[-1].use_pbr = True
            renderer.instances[-1].use_pbr_mapping = True

    renderer.set_camera([1.5,1.5,1.5], [0,0,0], [0, 0, 1], cache=True)
    frame = renderer.render(modes=('rgb', 'normal'))

    Image.fromarray((255*np.concatenate(frame, axis=1)[:,:,:3]).astype(np.uint8)).save('test_render_optimized.png')
    renderer.set_camera([1.49,1.49,1.49], [0,0.05,0.05], [0, 0, 1], cache=True) # simulate camera movement
    frame = renderer.render(modes=('optical_flow', 'scene_flow'))
    plt.subplot(1,2,1)
    plt.imshow(np.abs(frame[0][:,:,:3]) / np.max(np.abs(frame[0][:,:,:3])))
    plt.subplot(1,2,2)
    plt.imshow(np.abs(frame[1][:,:,:3]) / np.max(np.abs(frame[1][:,:,:3])))
    plt.savefig('test_render_optimized_flow.png')

    renderer.release()

