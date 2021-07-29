import bpy
import os
import math
import numpy as np
import glob

from os import listdir
from os.path import isfile, join

context = bpy.context

models_path = "ModelNet40"
models = sorted(glob.glob(models_path+"/*/*/*.off"))
nviews = 12

scene = bpy.context.scene

missed_models_list = []

for model_path in models:
    print(model_path)
    
    command = 'off2obj '+model_path[:-4]+'.off -o '+model_path[:-4]+'.obj'
    model_path = model_path[:-4]+'.obj'
    os.system(command)
    
    try:
        bpy.ops.import_scene.obj(filepath=model_path, filter_glob="*.obj")
    except:
        missed_models_list.append(model_path)
        continue
        
    imported = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    
    maxDimension = 5.0
    scaleFactor = maxDimension / max(imported.dimensions)
    imported.scale = (scaleFactor,scaleFactor,scaleFactor)
    imported.location = (0, 0, 0)
    
    imported.rotation_mode = 'XYZ'
    
    views = np.linspace(0, 2*np.pi, nviews, endpoint=False)
    print (views)
    
    for i in range(nviews):
        imported.rotation_euler[2] = views[i]
        imported.rotation_euler[0] = np.pi
        filename = model_path.split("/")[-1]
        print (filename)
        bpy.ops.view3d.camera_to_view_selected()
        context.scene.render.filepath = model_path+"_whiteshaded_v"+str(i)+".png"
        bpy.ops.render.render( write_still=True )
        
    meshes_to_remove = []
    for ob in bpy.context.selected_objects:
        meshes_to_remove.append(ob.data)
    bpy.ops.object.delete()
    # Remove the meshes from memory too
    for mesh in meshes_to_remove:
        bpy.data.meshes.remove(mesh)
    
    imported = None
    del imported