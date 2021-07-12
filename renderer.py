import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread

from utils import *
from renderer_utils import *
from args import make_args

import torch
from pytorch3d.io import load_obj

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    TexturesAtlas,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    BlendParams
)

class PhongRenderer():
    def __init__(self, args, image_size, camera_dist, elevation, azim_angle):
        self.args = args
        self.image_size = image_size
        self.camera_dist = camera_dist
        self.elevation = elevation
        self.azim_angle = azim_angle
        self.view_num = len(elevation)
        assert len(elevation) == len(azim_angle)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

    def center_origin(self, verts, faces=None):
        verts = verts - torch.mean(verts, 0)
        return verts

    def set_textures(self, verts, faces=None):
        verts_rgb = torch.ones_like(verts)[None]
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        return textures

    def __call__(self, obj_filename):
        # Load obj file & Alignment
        verts, faces, _ = load_obj(obj_filename, device=self.device)
        verts = self.center_origin(verts)

        # Set vertex texture None
        textures = self.set_textures(verts)

        # Load a mesh & Extend the mesh to the number of view
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=textures)
        meshes = mesh.extend(self.view_num)
        # print("Vertices: {}, Faces: {}, View Num: {}".format(verts.shape[0], faces.verts_idx.shape[0], self.view_num))

        # Setting Rasterizer & Shader
        R, T = look_at_view_transform(dist=self.camera_dist, elev=self.elevation, azim=self.azim_angle)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        raster_setting = RasterizationSettings(
            image_size = self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_setting
        )

        blend_params = BlendParams(background_color=(0,0,0)) # Black back-ground
        shader = SoftPhongShader(device=self.device, cameras=cameras, blend_params=blend_params)

        # Setting Renderer using defined Rasterizer & Shader
        renderer = MeshRenderer(rasterizer, shader)

        # Rendering & Save images as "png"
        rendered_images = renderer(meshes)

        obj_file_id = (obj_filename.split("/")[-1]).split(".")[0]
        
        if self.args.save_view:
            output_path = self.args.rendering_output_path + "/" + obj_file_id + "/views"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for i in range(rendered_images.shape[0]):
                rendered_images = rendered_images[i, ..., :3].detach().squeeze().cpu()
                img_file = obj_file_id + "_" + str(i) + ".png"
                Image.fromarray((img.numpy()*255).astype(np.uint8)).save(output_path + "/" + img_file)

        # silhouette
        if self.args.rendering_silhouette:
            silhouettes = rendered_images.clone()
            silhouettes[silhouettes!=0.] = 1.
            if self.args.save_silhouette:
                output_path = self.args.rendering_output_path + "/" + obj_file_id + "/silhouette"
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                for i in range(silhouettes.shape[0]):
                    sil = silhouettes[i, ..., :3].detach().squeeze().cpu()
                    sil_file = obj_file_id + "_" + str(i) + ".png"
                    Image.fromarray((sil.numpy()*255).astype(np.uint8)).save(output_path + "/" + sil_file)

            return rendered_images, silhouettes

        else:
            return rendered_images

if __name__ == "__main__":
    
    # Load arguments
    args = make_args()

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Argument define for example test
    params = {
        "image_size": 256,
        "camera_dist": 1.8,   
        "elevation": [-45,-45,-45,-45,0,0,0,0,45,45,45,45],
        "azim_angle": [0,90,180,270]*3,
        "obj_filename": "./input/m0.obj"
        }
    args = make_args()
    renderer = PhongRenderer(args, params["image_size"], params["camera_dist"], params["elevation"], params["azim_angle"])
    images = renderer(params["obj_filename"])

