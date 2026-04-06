import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class XAnyLabelingDataset(Dataset):
    def __init__(self, root_dir, img_size=(256, 256), is_rgb=False):
        self.img_dir = os.path.join(root_dir, 'images')
        self.json_dir = os.path.join(root_dir, 'jsons')
        self.img_names = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        self.img_size = img_size
        self.is_rgb = is_rgb

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name_ext = self.img_names[idx]
        name = os.path.splitext(name_ext)[0]
        
        # 圖片處理
        mode = cv2.IMREAD_COLOR if self.is_rgb else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(os.path.join(self.img_dir, name_ext), mode)
        h_orig, w_orig = img.shape[:2]
        img = cv2.resize(img, self.img_size)

        # JSON 解析為 Mask
        json_path = os.path.join(self.json_dir, f"{name}.json")
        mask = np.zeros(self.img_size, dtype=np.uint8)
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data['shapes']:
                pts = np.array(shape['points'], dtype=np.float32)
                pts[:, 0] *= (self.img_size[0] / w_orig)
                pts[:, 1] *= (self.img_size[1] / h_orig)
                cv2.fillPoly(mask, [pts.astype(np.int32)], color=1)

        # 轉 Tensor
        img_t = torch.from_numpy(img).float() / 255.0
        if not self.is_rgb: img_t = img_t.unsqueeze(0) # (1, H, W)
        else: img_t = img_t.permute(2, 0, 1) # (3, H, W)
        
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t