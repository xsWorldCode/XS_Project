# export_onnx.py - 只导出ONNX模型
import torch
import onnx
import os

def main():
    print("=" * 50)
    print("🐱🐶 导出猫狗分类模型为ONNX格式")
    print("=" * 50)
    
    # 配置
    MODEL_PATH = 'checkpoints/best_model.pth'
    ONNX_PATH = 'pt_to_onnx/model.onnx'
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在")
        print(f"   请确保 {MODEL_PATH} 存在")
        print(f"   需要先训练模型或放置训练好的权重文件")
        return
    
    print(f"📦 加载模型: {MODEL_PATH}")
    
    # 1. 加载PyTorch模型
    try:
        # 导入你的模型定义
        from Resnet import resnet54
        
        # 加载检查点
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        model = resnet54(num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ 模型加载成功")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n💡 可能的原因:")
        print("   1. net.py 不存在或路径错误")
        print("   2. 模型结构不匹配")
        print("   3. 文件损坏")
        return
    
    # 2. 创建虚拟输入
    print("\n🔧 准备导出ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 3. 导出ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_PATH,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX导出成功: {ONNX_PATH}")
        
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        return
    
    # 4. 验证ONNX模型
    print("\n🔍 验证ONNX模型...")
    try:
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX模型验证通过")
        
        # 显示模型信息
        print(f"\n📊 ONNX模型信息:")
        print(f"   输入: {onnx_model.graph.input[0].name}")
        print(f"   输出: {onnx_model.graph.output[0].name}")
        
        # 获取输入输出形状
        for inp in onnx_model.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)  # 动态维度
                else:
                    shape.append(dim.dim_value)
            print(f"   输入形状: {shape}")
        
    except Exception as e:
        print(f"⚠️  ONNX验证警告: {e}")
    
    # 5. 测试推理
    print("\n🧪 测试ONNX推理...")
    try:
        import onnxruntime as ort
        import numpy as np
        
        ort_session = ort.InferenceSession(ONNX_PATH)
        
        # 准备测试输入
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        
        # 推理
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"✅ ONNX推理测试通过")
        print(f"   输入形状: {test_input.shape}")
        print(f"   输出形状: {ort_outputs[0].shape}")
        
        # 显示类别概率
        if 'classes' in checkpoint:
            classes = checkpoint['classes']
            probs = torch.softmax(torch.tensor(ort_outputs[0]), dim=1)
            print(f"\n📈 类别概率示例:")
            for i, cls in enumerate(classes):
                print(f"   {cls}: {probs[0][i]:.2%}")
        
    except ImportError:
        print("⚠️  未安装onnxruntime，跳过推理测试")
        print("   安装: pip install onnxruntime")
    except Exception as e:
        print(f"⚠️  推理测试失败: {e}")
    
    # 6. 文件信息
    print("\n" + "=" * 50)
    print("🎉 转换完成!")
    print("=" * 50)
    
    if os.path.exists(ONNX_PATH):
        size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
        print(f"📄 生成文件: {ONNX_PATH}")
        print(f"📏 文件大小: {size_mb:.2f} MB")
        
        print(f"\n💡 下一步:")
        print(f"   1. 直接使用: python 脚本加载 {ONNX_PATH}")
        print(f"   2. 转换为TensorRT: 运行 export_engine.py")
        print(f"   3. 部署到其他平台")
    
    print(f"\n✅ 所有操作完成!")

if __name__ == "__main__":
    main()