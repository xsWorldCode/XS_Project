import torch
import torch.nn.utils.prune as prune
import os
from Resnet import ResNet54, Bottleneck

def prune_resnet54():
    device = torch.device("cpu")
    model = ResNet54(num_classes=2)
    model_path = r"F:\python_code\checkpoints\best_model.pth"
    
    if not os.path.exists(model_path):
        print("找不到原始权重文件")
        return

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    raw_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    fixed_dict = {k.replace('module.', '').replace('layer2.', 'layers.0.').replace('layer3.', 'layers.1.').replace('layer4.', 'layers.2.').replace('layer5.', 'layers.3.'): v for k, v in raw_dict.items()}
    model.load_state_dict(fixed_dict, strict=False)

    print(">>> 正在对模型进行全局非结构化剪枝 (50%)...")
    # 遍历所有卷积层进行剪枝
    parameters_to_prune = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            parameters_to_prune.append((m, 'weight'))

    # 使用全局剪枝，保留最重要的 50% 权重
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.5,
    )

    # 使剪枝永久化（移除 mask）
    for m, name in parameters_to_prune:
        prune.remove(m, name)

    # 保存剪枝后的模型
    save_path = "resnet54_pruned.pth"
    # 使用压缩字典保存，体积会显著缩小
    torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=True)
    
    print(f"剪枝完成！模型已保存至: {save_path}")
    print(f"原始大小: {os.path.getsize(model_path)/1024/1024:.2f} MB")
    print(f"剪枝后大小: {os.path.getsize(save_path)/1024/1024:.2f} MB")

if __name__ == "__main__":
    prune_resnet54()