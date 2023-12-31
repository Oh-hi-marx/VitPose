import argparse
import os.path as osp

import torch
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np


from time import time
from PIL import Image
from torchvision.transforms import transforms

from VitPose.models.model import ViTPose
from VitPose.utils.visualization import draw_points_and_skeleton, joints_dict
from VitPose.utils.dist_util import get_dist_info, init_dist
from VitPose.utils.top_down_eval import keypoints_from_heatmaps

from VitPose.configs.ViTPose_huge_coco_256x192 import model as model_cfg
from VitPose.configs.ViTPose_huge_coco_256x192 import data_cfg
#from VitPose.configs.ViTPose_base_coco_256x192 import model as model_cfg
#from VitPose.configs.ViTPose_base_coco_256x192 import data_cfg

from VitPose.yolov8 import yolov8
__all__ = ['inference']


@torch.no_grad()

class ViT:
    def __init__(self):
        self.device = 'cuda'
        CKPT_PATH = "./VitPose/checkpoints/vitpose-b-multi-coco.pth"
        CKPT_PATH = "./VitPose/checkpoints/vitpose-h-multi-coco.pth"
        # Prepare model
        self.vit_pose = ViTPose(model_cfg)


        ckpt = torch.load(CKPT_PATH )
        if 'state_dict' in ckpt:
            self.vit_pose.load_state_dict(ckpt['state_dict'])
        else:
            self.vit_pose.load_state_dict(ckpt)
        self.vit_pose.to(self.device)


        self.img_size = data_cfg['image_size']
        self.yolo = yolov8(filter = [0])

    def detect(self, img, padding ):
        boxes = self.yolo.pred(img)
        crops = self.yolo.cropall(img,boxes, padding)
        return crops

    def detectPose(self, img, padding = 0.1):
        crops = self.detect(img, padding)
        allPoints =[]
        plots = []
        cropBoxs =[]
        for crop in crops:
            cropBox = crop[-1]
            crop = crop[0]

            points, plot = self.inference(crop)
            allPoints.append(points)
            plots.append(plot)
            cropBoxs.append(cropBox)
        return allPoints, plots,cropBoxs

    def inference(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        org_w, org_h = img.size

        img_tensor = transforms.Compose (
            [transforms.Resize((self.img_size[1], self.img_size[0])),
             transforms.ToTensor()]
        )(img).unsqueeze(0).to(self.device)

        # Feed to model
        tic = time()
        heatmaps = self.vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4
        elapsed_time = time()-tic
        print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time**-1: .1f} fps]\n")

        # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]),
                                               unbiased=True, use_udp=True)
        points = np.concatenate([points[:, :, ::-1], prob], axis=2)

        # Visualization

        for pid, point in enumerate(points):
            img = np.array(img)[:, :, ::-1] # RGB to BGR for cv2 modules
            img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                           points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                           points_palette_samples=10, confidence_threshold=0.4)



        return points, img


if __name__ == "__main__":



    img = Image.open('examples/img1.jpg')

    vit = ViT()
    for i in range(10):
        points, plot = vit.inference(img)
    cv2.imwrite("plot.jpg", plot)
