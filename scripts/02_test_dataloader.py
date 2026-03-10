import os
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from conf.config import cfg
from dataset.dataloader import ZeroGraspDataset, ZeroGraspTransform
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)
    
    return images, targets

def test_dataloader():
    print("="*60)
    print("[INFO] Starting DataLoader test...")

    transform = ZeroGraspTransform(image_size=cfg.dataset.image_size, is_train=True)
    
    dataset_path = os.path.join(PROJECT_ROOT, cfg.dataset.root_dir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Cannot find dataset directory at {dataset_path}. Please check the path in config.yaml.")

    dataset = ZeroGraspDataset(root_dir=dataset_path, transform=transform)
    
    print(f"[INFO] Loading successfully. Number of samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=2,
        collate_fn=custom_collate_fn,
        drop_last=False
    )

    try:
        images, targets = next(iter(dataloader))
    except Exception as e:
        print(f"\n[FATAL] There was an error: {e}")
        return

    print(f"\n[THÔNG TIN BATCH] Batch Size: {len(images)}")
    print(f"\n[THÔNG TIN BATCH] Batch Size: {len(images)}")
    
    # 4. Kiểm tra chiều dữ liệu đa phương thức (7 Channels)
    # Kỳ vọng: [Batch_Size, 7, Height, Width]
    print(f"\n--- Phân tích Tensor Hình ảnh Đa phương thức ---")
    print(f"Kích thước (Shape)  : {images.shape}")
    print(f"Kiểu dữ liệu (Dtype): {images.dtype}")
    
    if images.shape[1] == 7:
        print("[ĐẠT] Hình ảnh đã gộp đúng 7 kênh (3 RGB + 1 Depth + 3 Normal).")
    else:
        print(f"[LỖI] Số kênh không đúng. Đang có {images.shape[1]} kênh.")

    # 5. Kiểm tra cấu trúc nhãn (Ground-truth Targets)
    print(f"\n--- Phân tích Cấu trúc Nhãn (Target Dicts) ---")
    for i, target in enumerate(targets):
        num_objects = len(target['labels'])
        print(f"\nẢnh {i+1} trong Batch chứa {num_objects} vật thể:")
        
        # Kiểm tra Bounding Boxes
        boxes = target['boxes']
        print(f"  - Bounding Boxes : Shape {boxes.shape}, Dtype {boxes.dtype}")
        
        # Kiểm tra Labels
        labels = target['labels']
        print(f"  - Labels         : Shape {labels.shape}, Dtype {labels.dtype}")
        
        # Kiểm tra Masks nhị phân
        masks = target['masks']
        print(f"  - Masks (2D)     : Shape {masks.shape}, Dtype {masks.dtype}")
        
        # Kiểm tra Ma trận Camera (Intrinsics)
        if 'intrinsics' in target:
            intrinsics = target['intrinsics']
            print(f"  - Camera K Matrix: Shape {intrinsics.shape}")
        
        if 'poses' in target:
            poses = target['poses']
            print(f"  - 6-DoF Poses    : Shape {poses.shape}")
        
        assert boxes.shape[0] == labels.shape[0] == masks.shape[0], \
            f"LỖI: Mất đồng bộ số lượng vật thể trong nhãn của ảnh {i+1}!"
            
    print("\n" + "="*60)
    print(" KẾT LUẬN: BỘ NẠP DỮ LIỆU HOẠT ĐỘNG HOÀN HẢO.")
    print("="*60)

if __name__ == "__main__":
    test_dataloader()