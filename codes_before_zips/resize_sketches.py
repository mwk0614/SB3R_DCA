import os
import time
import glob
from PIL import Image

from utils import resize_image

root_dir = "SHREC2013/TRAINING_SKETCHES/TRAINING_SKETCHES"
write_root_dir = "SHREC2013/TRAINING_SKETCHES_resized/TRAINING_SKETCHES"
class_dir_list = sorted(glob.glob(root_dir + "/*"))
print(len(class_dir_list))
cls_count = 0
for cdl in class_dir_list:
    cls_count += 1
    print(cdl)
    img_dir = cdl+"/train"
    write_img_dir = write_root_dir + "/" + cdl.split("/")[-1]+"/train"

    if not os.path.exists(write_img_dir):
        os.makedirs(write_img_dir)

    img_list = sorted(glob.glob(cdl+"/train/*"))
    img_count = 0
    for img_path in img_list:
        img_count += 1
        img_resized = resize_image(img_path)
        # import ipdb; ipdb.set_trace()
        img_resized.save(write_img_dir+"/"+img_path.split("/")[-1])
        if img_count == len(img_list):
            print("{} Image Done".format(cdl))
    if cls_count == len(class_dir_list):
        print("All Class Done")