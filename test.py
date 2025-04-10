import os
import torch

path =  "D:\Download\dataverse_files\Train"
def get_class_counts(path):
    class_counts = []
    for file in os.listdir(path):
        class_name = file
        file_path = os.path.join(path, file)
        image_len = len(os.listdir(file_path))
        print(f"{class_name}: {image_len}")
        class_counts.append(image_len)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    return weights
