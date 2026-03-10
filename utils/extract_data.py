import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_archives(data_dir: str, remove_after: bool = False):
    target_path = Path(data_dir)
    if not target_path.exists():
        raise FileNotFoundError(f"[Error] Target directory does not exist: {data_dir}")

    # Aggregating both .tar and .tar.gz files in the directory hierarchy
    archives = list(target_path.rglob("*.tar.gz")) + list(target_path.rglob("*.tar"))
    
    if not archives:
        print(f"[Info] No archive files detected in {data_dir}. System exiting.")
        return

    print(f"[Info] Detected {len(archives)} archive files. Initiating native I/O extraction...")

    error_logs = []

    for archive_path in tqdm(archives, desc="Extracting Shards", unit="file"):
        try:
            # Invoking OS-level tar command for optimal I/O throughput
            cmd = ["tar", "-xf", str(archive_path), "-C", str(archive_path.parent)]
            
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Post-extraction cleanup to reclaim disk space
            if remove_after:
                archive_path.unlink()
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to extract {archive_path.name}. stderr: {e.stderr.strip()}"
            error_logs.append(error_msg)
        except Exception as e:
            error_logs.append(f"Unexpected OS error on {archive_path.name}: {str(e)}")

    if error_logs:
        print("\n[Warning] Extraction completed with errors:")
        for log in error_logs:
            print(f"  - {log}")
    else:
        print("\n[Success] All archives extracted successfully ensuring spatial data integrity.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-throughput Archive Extractor for ZeroGrasp Dataset.")
    parser.add_argument(
        "--dir", 
        type=str, 
        default="./data/train_tiny", 
        help="Root directory containing the compressed shards."
    )
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="Flag to strictly remove original .tar/.tar.gz files after successful extraction to save VRAM/Disk."
    )
    
    args = parser.parse_args()
    extract_archives(args.dir, args.clean)