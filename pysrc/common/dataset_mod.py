import torch.utils.data as data
from PIL import Image
import numpy as np

class Originaldataset(data.Dataset):
    def __init__(self, data_list, transform, phase):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        img_path = self.data_list[index][3]
        acc_str_list = self.data_list[index][ :3]
        acc_list = [float(num) for num in acc_str_list]

        img_pil = Image.open(img_path)
        acc_numpy = np.array(acc_list)

        img_trans, acc_trans = self.transform(img_pil, acc_numpy, phase=self.phase)
        return img_trans, acc_trans