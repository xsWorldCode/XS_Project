import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # 用於可視化
import pandas as pd # 用於記錄數據
import os

from Net.ResUnet import OptimizedResUNet
from JSON_process import XAnyLabelingDataset

# --- 新增指標計算函數 ---
def get_metrics(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    
    # Dice = 2 * Intersection / (Total Area)
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    # IoU = Intersection / Union
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return dice.item(), iou.item()

def train_model():
    # 1. 超參數與配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    LR = 1e-4
    EPOCHS = 200
    DATA_PATH = r"H:\DataSets\Wood_Data\train"
    LOG_DIR = "runs/wood_segmentation"
    
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    writer = SummaryWriter(LOG_DIR)
    
    # 2. 初始化數據
    dataset = XAnyLabelingDataset(DATA_PATH, img_size=(256, 256), is_rgb=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = OptimizedResUNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    results = [] # 用於保存 CSV 的數據列表
    best_loss = float('inf')

    # 3. 訓練循環
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, epoch_dice, epoch_iou = 0.0, 0.0, 0.0
        
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 計算當前 Batch 的指標
            d_score, i_score = get_metrics(outputs, masks)
            epoch_loss += loss.item()
            epoch_dice += d_score
            epoch_iou += i_score
            
        # 計算本輪平均值
        avg_loss = epoch_loss / len(loader)
        avg_dice = epoch_dice / len(loader)
        avg_iou = epoch_iou / len(loader)

        # 4. 打印指標與記錄
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
        
        # 寫入 TensorBoard
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("Metric/Dice", avg_dice, epoch)
        writer.add_scalar("Metric/IoU", avg_iou, epoch)
        
        # 寫入列表供 CSV 使用
        results.append([epoch+1, avg_loss, avg_dice, avg_iou])

        # 保存最優模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_resunet.pth")

    # 5. 數據持久化
    df = pd.DataFrame(results, columns=['epoch', 'loss', 'dice', 'iou'])
    df.to_csv(os.path.join(LOG_DIR, "train_log.csv"), index=False)
    writer.close()
    print("訓練結束，數據已記錄。")

if __name__ == "__main__":
    train_model()