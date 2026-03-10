import os
from pathlib import Path

def diagnose_dataset(root_dir="./data/train_tiny"):
    print("="*50)
    print(f" CHẨN ĐOÁN HỆ THỐNG LƯU TRỮ: {root_dir}")
    print("="*50)
    
    p = Path(root_dir)
    if not p.exists():
        print(f"[LỖI] Thư mục {root_dir} hoàn toàn không tồn tại.")
        return
        
    all_files = list(p.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    
    print(f"Tổng số tệp tin vật lý   : {len(files_only)}")
    
    if len(files_only) == 0:
        print("[LỖI] Thư mục trống. Quá trình tải dữ liệu đã thất bại.")
        return

    # Thống kê định dạng tệp
    tars = list(p.rglob("*.tar"))
    gzs = list(p.rglob("*.gz"))
    pngs = list(p.rglob("*.png"))
    jpgs = list(p.rglob("*.jpg"))
    
    print(f"\n[THỐNG KÊ ĐỊNH DẠNG]")
    print(f"- Tệp nén (.tar)         : {len(tars)}")
    print(f"- Tệp nén (.tar.gz)      : {len(gzs)}")
    print(f"- Hình ảnh (.png)        : {len(pngs)}")
    print(f"- Hình ảnh (.jpg)        : {len(jpgs)}")
    
    print(f"\n[MẪU TÊN TỆP THỰC TẾ (5 tệp đầu tiên)]")
    for f in files_only[:5]:
        print(f"  -> {f.name}")

if __name__ == "__main__":
    diagnose_dataset()