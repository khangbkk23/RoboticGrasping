import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def compute_surface_normals(depth_map):
    depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0)
    
    zx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    zy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    
    normal = np.dstack((-zx, -zy, np.ones_like(depth_map)))
    
    n = np.linalg.norm(normal, axis=2)

    n[n == 0] = 1e-5 
    
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    
    normal = ((normal + 1) * 0.5 * 255).astype(np.uint8)
    return normal

def process_and_save_normals(root_dir):
    root_path = Path(root_dir)
    depth_files = list(root_path.glob("*-depth.png"))
    
    print(f"Bắt đầu xử lý {len(depth_files)} tệp chiều sâu...")
    
    for depth_path in tqdm(depth_files, desc="Computing Normals"):
        depth_map = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        
        depth_map_float = depth_map.astype(np.float64)
        
        # Tính toán Normal
        normal_map = compute_surface_normals(depth_map_float)
        
        normal_path = str(depth_path).replace("-depth.png", "-normal.png")
        cv2.imwrite(normal_path, normal_map)

if __name__ == "__main__":
    process_and_save_normals("./data/train_tiny")