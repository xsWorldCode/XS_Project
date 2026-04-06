import torch
from Resnet import ResNet54

# 1. 实例化你的 ResNet54
model = ResNet54(num_classes=2)

# 2. 加载全量检查点
checkpoint = torch.load(r"F:\python_code\checkpoints\best_model.pth", map_location="cpu")

# 3. 核心修复：只提取模型权重部分
# 报错信息显示权重的 key 名叫 "model_state_dict"
state_dict = checkpoint["model_state_dict"]

# 4. 加载到模型中
model.load_state_dict(state_dict)

# 5. 设置为评估模式并导出
model.eval()
example_input = torch.rand(1, 3, 224, 224)
traced_module = torch.jit.trace(model, example_input)
traced_module.save("model.pt")

print("✅ model.pt 导出成功！")