import torch
import torchvision.transforms.functional as TF
import numpy as np

class RGBDTransform:
    def __init__(self, image_size):
        self.target_size = image_size # (H, W)

    def __call__(self, rgb_image, depth_image, masks, bounding_boxes):
        rgb_tensor = TF.to_tensor(rgb_image)
        
        rgb_tensor = TF.normalize(rgb_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # base on ImageNet stats

        return rgb_tensor, depth_image, masks, bounding_boxes