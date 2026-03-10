import os
import cv2
import torch
import numpy as np
import random
import json
from pathlib import Path
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class ZeroGraspTransform:
    def __init__(self, image_size, is_train=True):
        self.target_size = tuple(image_size) # (H, W)
        self.is_train = is_train

    def __call__(self, rgb, depth, normal, masks, boxes, poses=None):
        # 1. Chuyển đổi sang Tensor
        rgb_t = TF.to_tensor(rgb)
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        normal_t = TF.to_tensor(normal)
        
        orig_h, orig_w = rgb_t.shape[1], rgb_t.shape[2]
        target_h, target_w = self.target_size

        # 2. ĐỒNG BỘ KHÔNG GIAN (SPATIAL ALIGNMENT & RESIZE)
        # Sử dụng Bilinear cho ảnh màu/pháp tuyến và Nearest cho Depth/Mask để tránh mất mát giá trị vật lý
        rgb_t = TF.resize(rgb_t, self.target_size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        depth_t = TF.resize(depth_t, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
        normal_t = TF.resize(normal_t, self.target_size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        # 3. Nội suy (Scale) tọa độ Bounding Boxes theo tỷ lệ mới
        if len(boxes) > 0:
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            boxes[:, 0] *= scale_x  # xmin
            boxes[:, 2] *= scale_x  # xmax
            boxes[:, 1] *= scale_y  # ymin
            boxes[:, 3] *= scale_y  # ymax

        # 4. Resize Masks nhị phân
        if masks.numel() > 0:
            # Pytorch resize yêu cầu shape [N, 1, H, W]
            masks_t = masks.unsqueeze(1).float()
            masks_t = TF.resize(masks_t, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
            masks_t = masks_t.squeeze(1).to(torch.uint8)
        else:
            masks_t = masks.to(torch.uint8)

        # 5. Data Augmentation (Photometric)
        if self.is_train and random.random() > 0.5:
            rgb_t = TF.adjust_brightness(rgb_t, random.uniform(0.8, 1.2))
            rgb_t = TF.adjust_contrast(rgb_t, random.uniform(0.8, 1.2))
            rgb_t = TF.adjust_saturation(rgb_t, random.uniform(0.8, 1.2))

        # 6. Chuẩn hóa Tensor
        rgb_t = TF.normalize(rgb_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normal_t = TF.normalize(normal_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        depth_t = depth_t / 1000.0  # Scale từ mm sang mét
        
        return rgb_t, depth_t, normal_t, masks_t, boxes, poses


class ZeroGraspDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.files = sorted(list(self.root_dir.rglob("*.depth.png")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Trích xuất đường dẫn
        depth_path = str(self.files[idx])
        base_name_full = depth_path.replace(".depth.png", "")
        
        parent_dir = Path(base_name_full).parent
        base_stem = Path(base_name_full).name
        
        # 2. Dò tìm RGB và thiết lập đường dẫn
        rgb_candidates = list(parent_dir.glob(f"{base_stem}*.jpg"))
        if not rgb_candidates:
            raise FileNotFoundError(f"[DataLoader] Không tìm thấy ảnh màu cho: {base_stem}")
        color_path = str(rgb_candidates[0])
        
        normal_path = f"{base_name_full}.normal.png"
        camera_path = f"{base_name_full}.camera.json"
        grasp_path = f"{base_name_full}.grasp.npz"
        gt_path = f"{base_name_full}.gt.json"
        
        # 3. Đọc dữ liệu với Safety Checks
        img_rgb_raw = cv2.imread(color_path)
        if img_rgb_raw is None:
            raise ValueError(f"Lỗi đọc ảnh RGB: {color_path}")
        img_rgb = cv2.cvtColor(img_rgb_raw, cv2.COLOR_BGR2RGB)
        
        img_depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if img_depth_raw is None:
            raise ValueError(f"Lỗi đọc ảnh Depth: {depth_path}")
        img_depth = img_depth_raw
        
        if os.path.exists(normal_path):
            img_normal_raw = cv2.imread(normal_path)
            if img_normal_raw is None:
                 raise ValueError(f"Lỗi đọc ảnh Normal: {normal_path}")
            img_normal = cv2.cvtColor(img_normal_raw, cv2.COLOR_BGR2RGB)
        else:
            # Fallback nếu chưa sinh Normal Map (khớp với resolution của RGB)
            img_normal = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.uint8)

        # 4. Đọc nhãn JSON
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
            
        boxes = []
        obj_ids = []
        
        if "objects" in gt_data:
            for obj in gt_data["objects"]:
                bbox = obj.get("bbox", [0, 0, 10, 10])
                boxes.append(bbox)
                obj_ids.append(obj.get("class_id", 1))
        else:
            boxes.append([0, 0, img_rgb.shape[1], img_rgb.shape[0]])
            obj_ids.append(1)

        masks = torch.zeros((len(boxes), img_rgb.shape[0], img_rgb.shape[1]), dtype=torch.uint8)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(obj_ids, dtype=torch.int64),
            "masks": masks
        }
        
        # 5. Đọc Metadata
        if os.path.exists(camera_path):
            with open(camera_path, 'r') as f:
                cam_data = json.load(f)
            target["intrinsics"] = torch.as_tensor(cam_data.get('intrinsic_matrix', np.eye(3)), dtype=torch.float32)
            
        if os.path.exists(grasp_path):
            grasp_data = np.load(grasp_path, allow_pickle=True)
            key = 'poses' if 'poses' in grasp_data.files else grasp_data.files[0]
            target["poses"] = torch.as_tensor(grasp_data[key], dtype=torch.float32)
        
        # 6. Transform & Resize
        if self.transform:
            img_rgb, img_depth, img_normal, target["masks"], target["boxes"], poses = self.transform(
                img_rgb, img_depth, img_normal, target["masks"], target["boxes"], target.get("poses")
            )
            if poses is not None:
                target["poses"] = poses
            
        # 7. Early Fusion: Lúc này cả 3 tensor đều có chung kích thước là target_size [480, 640]
        img_multimodal = torch.cat([img_rgb, img_depth, img_normal], dim=0)
        
        return img_multimodal, target