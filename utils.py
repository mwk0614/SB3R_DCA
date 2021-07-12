import meshio
from PIL import Image

def read_image(filepath):
    raise NotImplementedError

def read_cla(cla_file):
    with open(cla_file, 'r') as f:
        contents = list()
        for line in f:
            contents.append(line.strip())
    split_idx = [idx for idx, el in enumerate(contents) if el == ""]

    class_list = list()
    for i in range(len(split_idx)-1):
        start = split_idx[i]
        end = split_idx[i+1]
        class_list.append(contents[start+1:end])
    
    class_model = {}
    class_number_check = list()
    model_number_check = list()
    for pair in class_list:
        class_name = pair[0].split(" ")[0]
        if "\t" in class_name:
            class_name = class_name.replace("\t","")
        model_id = ["m"+idx for idx in pair[1:]]
        class_model[class_name] = model_id
        class_number_check.append(class_name)
        model_number_check += model_id
        
    assert len(class_number_check) == int(contents[1].split(" ")[0]) 
    assert len(model_number_check) == int(contents[1].split(" ")[1])

    return class_model

if __name__ == "__main__":
    cla_file = "./cla_files/SHREC13_SBR_Model.cla"
    class_model = read_cla(cla_file)
    
