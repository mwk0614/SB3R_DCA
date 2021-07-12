import glob
import meshio

def off2obj(input_off, output_obj_path):
    mesh = meshio.read(input_off, file_format="off")
    output_objFile_path = output_obj_path + "/" + (input_off.split("/")[-1]).split(".")[0] + ".obj"
    # import ipdb; ipdb.set_trace(context=21)
    mesh.write(output_objFile_path)

off_list = glob.glob("./models_off/*")
output_obj_dir = "./models_obj"

for off in off_list:
    off2obj(off, output_obj_dir)
