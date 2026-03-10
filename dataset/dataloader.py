import os
import cv2
import torch
import numpy as np
import random
from pathlib import Path
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class ZeroGraspTransform:
    def __init__(self, image_size, is_train=True):
        self.target_size = image_size
        self.is_train = is_train

    def __call__(self, rgb, depth, normal, masks, boxes, poses=None):
        # Convert inputs to tensor format
        rgb_t = TF.to_tensor(rgb)
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        normal_t = TF.to_tensor(normal)

        # Photometric augmentation (Train only)
        # Note: Geometric augmentation (e.g., hflip) is disabled to preserve 6-DoF pose integrity
        if self.is_train and random.random() > 0.5:
            rgb_t = TF.adjust_brightness(rgb_t, random.uniform(0.8, 1.2))
            rgb_t = TF.adjust_contrast(rgb_t, random.uniform(0.8, 1.2))
            rgb_t = TF.adjust_saturation(rgb_t, random.uniform(0.8, 1.2))

        # Standard normalization
        rgb_t = TF.normalize(rgb_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normal_t = TF.normalize(normal_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # Depth scaling (assuming mm scale, convert to meters)
        depth_t = depth_t / 1000.0
        
        return rgb_t, depth_t, normal_t, masks, boxes, poses


class ZeroGraspDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.files = sorted(list(self.root_dir.glob("*-color.png")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Path resolution
        color_path = str(self.files[idx])
        base_name = color_path.replace("-color.png", "")
        
        depth_path = f"{base_name}-depth.png"
        normal_path = f"{base_name}-normal.png"
        label_path = f"{base_name}-label.png"
        meta_path = f"{base_name}-meta.npy" 
        
        # 2. Load visual data
        img_rgb = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        img_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        img_normal = cv2.cvtColor(cv2.imread(normal_path), cv2.COLOR_BGR2RGB)
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # 3. Parse instance masks and bounding boxes
        obj_ids = np.unique(label_img)
        obj_ids = obj_ids[obj_ids != 0]
        
        masks = label_img == obj_ids[:, None, None]
        boxes = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            boxes.append([np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])])
        
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(obj_ids, dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8)
        }
        
        # 4. Load kinematics and intrinsic metadata
        if os.path.exists(meta_path):
            meta_data = np.load(meta_path, allow_pickle=True).item()
            target["intrinsics"] = torch.as_tensor(meta_data.get('intrinsics', np.eye(3)), dtype=torch.float32)
            target["poses"] = torch.as_tensor(meta_data.get('poses', []), dtype=torch.float32)
        
        # 5. Apply transformations
        if self.transform:
            img_rgb, img_depth, img_normal, target["masks"], target["boxes"], poses = self.transform(
                img_rgb, img_depth, img_normal, target["masks"], target["boxes"], target.get("poses")
            )
            if poses is not None:
                target["poses"] = poses
            
        # 6. Early Fusion: Concatenate [RGB(3) + Depth(1) + Normal(3)] -> 7 Channels
        img_multimodal = torch.cat([img_rgb, img_depth, img_normal], dim=0)
        
        return img_multimodal, target