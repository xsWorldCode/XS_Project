import torch
import torch.nn.functional as F
import os
from Resnet import ResNet54

def eval_pruning_quality():
    device = torch.device("cpu")
    torch.manual_seed(42)  # 固定输入
    dummy_input = torch.randn(1, 3, 224, 224)

    # 1. 加载原始 FP32 模型 (270MB)
    print(">>> 正在初始化原始模型...")
    model_orig = ResNet54(num_classes=2).eval()
    orig_path = r"F:\python_code\checkpoints\best_model.pth"
    
    checkpoint = torch.load(orig_path, map_location=device)
    raw = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    # 修复你的 Key 映射 (layer -> layers)
    fixed = {k.replace('module.', '').replace('layer2.', 'layers.0.').replace('layer3.', 'layers.1.').replace('layer4.', 'layers.2.').replace('layer5.', 'layers.3.'): v for k, v in raw.items()}
    model_orig.load_state_dict(fixed, strict=False)

    # 2. 加载剪枝 FP32 模型 (90MB)
    print(">>> 正在初始化 90MB 剪枝模型...")
    model_pruned = ResNet54(num_classes=2).eval()
    pruned_path = "resnet54_pruned.pth"
    
    if not os.path.exists(pruned_path):
        print(f"错误：找不到剪枝文件 {pruned_path}，请先运行剪枝脚本！")
        return
    model_pruned.load_state_dict(torch.load(pruned_path, map_location=device))

    # 3. 提取特征向量
    print(">>> 正在对比特征向量一致性...")
    with torch.no_grad():
        # 原始特征流
        x_o = model_orig.relu(model_orig.bn1(model_orig.conv1(dummy_input)))
        for layer in model_orig.layers: x_o = layer(x_o)
        feat_orig = torch.flatten(model_orig.avgpool(x_o), 1)

        # 剪枝特征流
        x_p = model_pruned.relu(model_pruned.bn1(model_pruned.conv1(dummy_input)))
        for layer in model_pruned.layers: x_p = layer(x_p)
        feat_pruned = torch.flatten(model_pruned.avgpool(x_p), 1)

    # 4. 计算余弦相似度
    similarity = F.cosine_similarity(feat_orig, feat_pruned).item()

    print("\n" + "="*50)
    print(f"{'对比项':<20} | {'数值':<20}")
    print("-" * 50)
    print(f"{'特征向量维度':<20} | {feat_orig.shape[1]:<20}")
    print(f"{'余弦相似度':<20} | {similarity:.6f}")
    
    # 质量判定
    if similarity > 0.98:
        status = "极佳 (剪枝几乎无损)"
    elif similarity > 0.90:
        status = "良好 (存在轻微偏差)"
    else:
        status = "警告 (精度下降风险大)"
    print(f"{'剪枝质量评估':<20} | {status:<20}")
    print("="*50)

if __name__ == "__main__":
    eval_pruning_quality()