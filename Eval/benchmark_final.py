import torch
import time
import os
import sys
from Resnet import ResNet54, Bottleneck

# 强制单线程，确保 Windows 环境稳定
torch.set_num_threads(1)
if 'fbgemm' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'fbgemm'

def get_size(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0

def load_final_model(path, num_classes=2):
    model = ResNet54(num_classes=num_classes).eval()
    # 1. 融合
    torch.ao.quantization.fuse_modules(model, ['conv1', 'bn1', 'relu'], inplace=True)
    for m in model.modules():
        if isinstance(m, Bottleneck):
            torch.ao.quantization.fuse_modules(m, ['conv1', 'bn1', 'relu1'], inplace=True)
            torch.ao.quantization.fuse_modules(m, ['conv2', 'bn2', 'relu2'], inplace=True)
            torch.ao.quantization.fuse_modules(m, ['conv3', 'bn3'], inplace=True)
            if m.downsample: torch.ao.quantization.fuse_modules(m.downsample, ['0', '1'], inplace=True)
    
    # 2. 准备与转换
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.ao.quantization.prepare(model)
    # 伪推理激活
    with torch.no_grad(): model_prepared(torch.randn(1, 3, 224, 224))
    model_int8 = torch.ao.quantization.convert(model_prepared)
    
    # 3. 加载合并了剪枝和量化后的权重
    model_int8.load_state_dict(torch.load(path, map_location='cpu'))
    return model_int8

def benchmark_all():
    device = torch.device("cpu")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    paths = {
        "Original": r"F:\python_code\checkpoints\best_model.pth",
        "Pruned_Only": "resnet54_pruned.pth",
        "Pruned_Quantized": "resnet54_pruned_int8.pth"
    }

    print(f"{'模型版本':<20} | {'延迟 (ms)':<12} | {'FPS':<8} | {'体积 (MB)'}")
    print("-" * 65)

    for name, path in paths.items():
        try:
            # 根据版本加载模型
            if name == "Original":
                model = ResNet54(num_classes=2).eval()
                # 这里复用你之前的 key 修复逻辑加载原始权重... (省略部分代码以保持简洁)
            elif name == "Pruned_Only":
                model = ResNet54(num_classes=2).eval()
                model.load_state_dict(torch.load(path, map_location='cpu'))
            else:
                model = load_final_model(path)

            # 预热
            with torch.no_grad():
                for _ in range(10): model(dummy_input)
            
            # 测试
            iters = 50
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(iters): model(dummy_input)
            end = time.perf_counter()
            
            lat = ((end - start) / iters) * 1000
            print(f"{name:<20} | {lat:>10.2f} ms | {1000/lat:>8.2f} | {get_size(path):>10.2f} MB")
        except Exception as e:
            print(f"{name:<20} | 测试失败: {e}")

if __name__ == "__main__":
    benchmark_all()