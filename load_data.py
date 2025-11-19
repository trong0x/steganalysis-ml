import os
import random
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COVER_DIR = os.path.join(BASE_DIR, 'data', 'ảnh', 'ảnh sạch up')
STEGO_DIR = os.path.join(BASE_DIR, 'data', 'ảnh', 'ảnh đã ẩn mã up')
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
def get_files_from_dir(directory: str) -> List[str]:
    """Hàm phụ trợ: Lấy danh sách toàn bộ file ảnh trong thư mục"""
    if not os.path.exists(directory):
        print(f"CẢNH BÁO: Không tìm thấy thư mục tại đường dẫn:")
        print(f"{directory}")
        print("   (Hãy chắc chắn bạn đã tạo folder này và đặt tên chính xác y hệt trong code)")
        return [] 
    files = []
    for f in os.listdir(directory):
        if f.lower().endswith(VALID_EXTENSIONS):
            full_path = os.path.join(directory, f)
            files.append(full_path)
            
    return files

def prepare_dataset(
    train_ratio: float = 0.8, 
    seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Hàm chính để chuẩn bị dữ liệu cho hệ thống.
    
    Args:
        train_ratio: Tỉ lệ chia tập Train (Ví dụ 0.8 là 80%).
        seed: Số ngẫu nhiên cố định (để kết quả giống nhau mỗi lần chạy).
        
    Returns:
        train_set, test_set: Hai danh sách chứa các cặp (đường_dẫn, nhãn).
    """
    
    print("ĐANG KHỞI TẠO DỮ LIỆU...")
    print(f"Thư mục gốc dự án: {BASE_DIR}")
    covers = get_files_from_dir(COVER_DIR)
    stegos = get_files_from_dir(STEGO_DIR)
    labeled_data = []
    
    for path in covers:
        labeled_data.append((path, 0)) 
        
    for path in stegos:
        labeled_data.append((path, 1)) 
        
    total_files = len(labeled_data)

    if total_files == 0:
        print("LỖI: Không tìm thấy ảnh nào trong cả 2 thư mục!")
        return [], []
        
    print(f"Tìm thấy tổng cộng: {total_files} ảnh")
    print(f"   - Ảnh sạch (nhãn 0): {len(covers)}")
    print(f"   - Ảnh ẩn mã (nhãn 1): {len(stegos)}")

    # --- BƯỚC 2: TRỘN NGẪU NHIÊN (SHUFFLE) ---
    random.seed(seed) # Quan trọng để tái lập kết quả
    random.shuffle(labeled_data)
    print("Đã trộn ngẫu nhiên dữ liệu.")

    # --- BƯỚC 3: CHIA THEO TỈ LỆ (SPLIT) ---
    split_idx = int(total_files * train_ratio)
    
    train_set = labeled_data[:split_idx]
    test_set = labeled_data[split_idx:]
    
    print("-" * 40)
    print(f"Hoàn tất chia dữ liệu (Train: {train_ratio*100}% / Test: {(1-train_ratio)*100:.0f}%)")
    print(f" Tập Train (Học): {len(train_set)} mẫu")
    print(f" Tập Test (Thi) : {len(test_set)} mẫu")
    print("-" * 40)
    
    return train_set, test_set

if __name__ == "__main__":
    try:
        train_data, test_data = prepare_dataset(train_ratio=0.8)
        
        if len(train_data) > 0:
            print("\n [DEMO] Kiểm tra 3 mẫu đầu tiên trong tập Train:")
            for path, label in train_data[:3]:
                label_name = "ẨN MÃ (1)" if label == 1 else "SẠCH (0)"
                filename = os.path.basename(path)
                print(f" File: {filename}  --->  Nhãn: {label_name}")
                
    except Exception as e:
        print(f"\n Có lỗi xảy ra: {e}")
