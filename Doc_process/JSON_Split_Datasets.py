import os
import shutil
import random

def split_dataset(image_dir, json_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 1. 獲取所有圖片文件名（不含後綴）
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    file_names = [os.path.splitext(f)[0] for f in image_files]
    
    # 確保每個圖片都有對應的 JSON
    valid_files = []
    for name in file_names:
        if os.path.exists(os.path.join(json_dir, f"{name}.json")):
            valid_files.append(name)
        else:
            print(f"警告: 找不到 {name} 的標註文件，已跳過。")

    # 2. 隨機打亂
    random.seed(42) # 固定隨機種子，確保結果可複現
    random.shuffle(valid_files)

    # 3. 計算劃分索引
    total = len(valid_files)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        'train': valid_files[:train_end],
        'val': valid_files[train_end:val_end],
        'test': valid_files[val_end:]
    }

    # 4. 創建目錄並移動文件
    for split_name, filenames in splits.items():
        # 創建子文件夾：output/train/images 和 output/train/jsons
        img_output = os.path.join(output_dir, split_name, 'images')
        json_output = os.path.join(output_dir, split_name, 'jsons')
        os.makedirs(img_output, exist_ok=True)
        os.makedirs(json_output, exist_ok=True)

        for name in filenames:
            # 找到原始圖片後綴
            orig_img_ext = ""
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                if os.path.exists(os.path.join(image_dir, f"{name}{ext}")):
                    orig_img_ext = ext
                    break
            
            # 複製文件 (用 copy 避免弄丟原始數據)
            shutil.copy(os.path.join(image_dir, f"{name}{orig_img_ext}"), os.path.join(img_output, f"{name}{orig_img_ext}"))
            shutil.copy(os.path.join(json_dir, f"{name}.json"), os.path.join(json_output, f"{name}.json"))

    print(f"數據集劃分完成！")
    print(f"總計: {total} | 訓練集: {len(splits['train'])} | 驗證集: {len(splits['val'])} | 測試集: {len(splits['test'])}")

# --- 使用設定 ---
image_path = r'H:\DataSets\新和木板图片20250324(裁切后)'  # 原始圖片路徑
json_path = r'H:\DataSets\新和木板图片20250324(裁切后)' # 原始 JSON 路徑
output_path = r'H:\DataSets\Wood_Data' # 劃分後的保存路徑

split_dataset(image_path, json_path, output_path)