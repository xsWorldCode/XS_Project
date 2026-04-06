import tensorrt as trt
import os

def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # --- 关键修改：配置优化配置文件 (Optimization Profile) ---
    profile = builder.create_optimization_profile()
    
    # 假设你的模型输入节点名称是 "input" (如果不确定，看下文查看方法)
    # 参数含义：[Batch, Channel, Height, Width]
    # 我们统一固定为 [1, 3, 224, 224]
    input_name = "input" 
    input_shape = (1, 3, 224, 224)
    profile.set_shape(input_name, input_shape, input_shape, input_shape) 
    config.add_optimization_profile(profile)
    # -------------------------------------------------------

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    print(f"正在解析 ONNX 文件: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ONNX 解析失败:')
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    # 再次确认输入名称（防止 ONNX 里的名称不是 "input"）
    real_input_name = network.get_input(0).name
    if real_input_name != input_name:
        print(f"警告: 检测到输入名称为 '{real_input_name}'，正在重新配置...")
        # 如果名称不对，重新设置一次
        profile.set_shape(real_input_name, input_shape, input_shape, input_shape)

    print(f"正在针对 RTX 5060 Ti (Blackwell) 构建 Engine...")
    plan = builder.build_serialized_network(network, config)
    
    if plan:
        with open(engine_path, 'wb') as f:
            f.write(plan)
        print(f"成功生成 Engine: {engine_path}")
    else:
        print("构建失败，请检查日志。")

if __name__ == "__main__":
    build_engine("F:\python_code\pt_to_onnx\model.onnx", "F:\python_code\onnx_to_engine\model.engine")