import json
import cv2
import numpy as np
import os
import torch
import einops

import PIL
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self,depth_path, image_path, prompt,number,epochs):
        self.data = []
        
        for i in range(0,epochs):
            for n in range(number+1):
                target_path= os.path.join(image_path, "{}_after.png".format(n))
                source_path= os.path.join(depth_path, "{}.png".format(n))

                self.data.append({"source": source_path, "target": target_path, "prompt": prompt[n]})
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)


        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

