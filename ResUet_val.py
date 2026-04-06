import torch
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader
from Net.ResUnet import OptimizedResUNet  # 確保文件名正確
from Doc_process.JSON_process import XAnyLabelingDataset # 確保文件名正確

def validate():
    # 1. 配置與路徑
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "checkpoints/best_resunet.pth"
    VAL_DATA_PATH = r"H:\DataSets\Wood_Data\test" # 指向你的驗證集
    SAVE_DIR = "test_results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. 加載數據與模型
    dataset = XAnyLabelingDataset(VAL_DATA_PATH, img_size=(256, 256), is_rgb=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = OptimizedResUNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    total_dice, total_iou = 0.0, 0.0
    print(f"開始驗證，樣本總數: {len(loader)}")

    # 3. 推理與評估
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            # 模型預測
            outputs = model(imgs)
            pred = torch.sigmoid(outputs)
            pred = (pred > 0.5).float() # 二值化

            # 計算指標
            intersection = (pred * masks).sum()
            union = (pred + masks).sum() - intersection
            dice = (2. * intersection + 1e-6) / (pred.sum() + masks.sum() + 1e-6)
            iou = (intersection + 1e-6) / (union + 1e-6)
            
            total_dice += dice.item()
            total_iou += iou.item()

            # 4. 可視化保存 (每隔幾張存一張，或者全部保存)
            if i < 20: # 僅保存前 20 張作為示例
                img_np = imgs[0].cpu().numpy().transpose(1, 2, 0) * 255
                mask_np = masks[0].cpu().numpy().transpose(1, 2, 0) * 255
                pred_np = pred[0].cpu().numpy().transpose(1, 2, 0) * 255
                
                # 橫向拼接：原圖 | 真實標注 | 模型預測
                combined = np.hstack([img_np, mask_np, pred_np]).astype(np.uint8)
                cv2.imwrite(os.path.join(SAVE_DIR, f"result_{i}.png"), combined)

    # 5. 輸出最終結果
    print("-" * 30)
    print(f"驗證完成！")
    print(f"平均 Dice 係數: {total_dice/len(loader):.4f}")
    print(f"平均 IoU (mIoU): {total_iou/len(loader):.4f}")
    print(f"可視化結果已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    validate()