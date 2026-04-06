import torch
import torch.nn as nn
import os
from copy import deepcopy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Resnet import ResNet54, Bottleneck 

# 强制使用 fbgemm 后端（Windows x86 标配）
torch.backends.quantized.engine = 'fbgemm'

def get_calibration_loader(data_root, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if not os.path.exists(data_root):
        print(f"❌ 找不到数据路径: {data_root}")
        return None
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def run_quantize_pruned():
    print(">>> 步骤1: 加载已剪枝的模型权重...")
    model = ResNet54(num_classes=2)
    # ！！！这里加载你之前剪枝生成的 90MB 文件 ！！！
    pruned_path = "resnet54_pruned.pth"
    
    if not os.path.exists(pruned_path):
        print(f"❌ 错误：找不到剪枝后的权重 {pruned_path}，请先运行剪枝脚本！")
        return

    # 加载剪枝后的权重
    model.load_state_dict(torch.load(pruned_path, map_location='cpu'))
    model.eval()
    
    print(">>> 步骤2: 执行算子融合 (Fusion)...")
    # 融合 Conv+BN+ReLU 提高推理速度
    torch.ao.quantization.fuse_modules(model, ['conv1', 'bn1', 'relu'], inplace=True)
    for m in model.modules():
        if isinstance(m, Bottleneck):
            torch.ao.quantization.fuse_modules(m, ['conv1', 'bn1', 'relu1'], inplace=True)
            torch.ao.quantization.fuse_modules(m, ['conv2', 'bn2', 'relu2'], inplace=True)
            torch.ao.quantization.fuse_modules(m, ['conv3', 'bn3'], inplace=True)
            if m.downsample:
                torch.ao.quantization.fuse_modules(m.downsample, ['0', '1'], inplace=True)

    print(">>> 步骤3: 准备量化并进行校准 (Calibration)...")
    # 使用 fbgemm 配置
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.ao.quantization.prepare(model)
    
    # 必须通过真实数据来校准，让量化器知道 0 以外的权重分布
    calib_loader = get_calibration_loader(r"F:\python_code\DataSets\split\val")
    if calib_loader:
        with torch.no_grad():
            for i, (images, _) in enumerate(calib_loader):
                model_prepared(images)
                if i >= 20: break
        print("✅ 校准完成")
    else:
        print("❌ 校准失败，缺少数据")
        return

    print(">>> 步骤4: 最终转换并导出 INT8 模型...")
    # 这一步会将 Float32 转换为真正的 Int8 存储
    model_int8 = torch.ao.quantization.convert(model_prepared)
    
    save_path = "resnet54_pruned_int8.pth"
    torch.save(model_int8.state_dict(), save_path)
    print(f"🎉 大功告成！剪枝+量化模型已保存至: {save_path}")
    print(f"最终预测体积: ~20-25 MB")

if __name__ == "__main__":
    run_quantize_pruned()