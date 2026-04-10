猫狗分类模型压缩与 TensorRT 部署

基于 ResNet54 的猫狗图像分类模型，通过结构化剪枝、微调和 TensorRT FP16 量化，在几乎不损失精度的情况下将模型体积压缩 70% 以上，并在 RTX 5060 Ti 上实现高效推理。

## 项目结构
├── models/ # 模型定义（ResNet54）
├── pruning/ # 结构化/非结构化剪枝脚本
├── eval/ # 精度与相似度评估
├── export/ # 导出 ONNX 模型
├── deploy/ # TensorRT 推理引擎加载与预测
├── checkpoints/ # 存放原始模型权重（不提交）
├── DataSets/ # 猫狗数据集（不提交）
└── requirements.txt # 依赖列表

text

## 环境配置

- Python 3.8+
- PyTorch 2.x
- TensorRT 10.x
- ONNX / onnxruntime-gpu
- OpenCV, numpy

安装依赖：
```bash
pip install -r requirements.txt
数据集准备
数据集应按照以下结构组织：

text
DataSets/
  split/
    train/
      cat/       # 猫图片
      dog/       # 狗图片
    val/
      cat/
      dog/
快速开始
1. 训练原始模型（可选）
若你已有训练好的权重（checkpoints/model.pth），可直接进入剪枝步骤。否则可自行训练 ResNet54。

2. 结构化剪枝与微调
bash
python pruning/structured_prune.py          # 生成 Pruned_Structured.pth
# 然后运行微调脚本（需自行准备训练数据加载器）
python finetune.py                          # 输出 Pruned_Structured_FineTuned.pth
3. 评估剪枝效果
bash
python eval/cosine_similarity_eval.py       # 特征余弦相似度
python eval/classify_accuracy.py            # 分类准确率
4. 导出 ONNX 模型
bash
python export/export_onnx.py
# 输出 model_static.onnx
5. 转换为 TensorRT FP16 引擎
bash
trtexec --onnx=model_static.onnx --saveEngine=model_fp16.engine --fp16
6. 推理预测
bash
python deploy/infer_tensorrt.py
修改脚本中的 IMAGE_PATH 为你的猫/狗图片路径，即可输出分类结果。

实验结果
模型	准确率	模型文件大小	推理设备
原始 ResNet54 (FP32)	71.4%	269 MB	CPU/GPU
结构化剪枝 50% + 微调	68.0%	56 MB	GPU
剪枝 + FP16 量化	68.0%	45 MB	RTX 5060 Ti (FP16)
余弦相似度（剪枝后 vs 原始）: 0.9988

许可证
MIT License
