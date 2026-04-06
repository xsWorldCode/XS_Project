import tensorrt as trt
import numpy as np
import ctypes

# 載入 cudart 庫以手動分配顯存（替代 pycuda）
def check_cuda_err(err):
    if err != 0:
        raise RuntimeError(f"CUDA Error: {err}")

# 加載 CUDA Runtime API
cuda_rt = ctypes.CDLL('cudart.dll') # Windows 下是 cudart.dll

def verify_engine_no_pycuda(engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # 獲取輸入輸出名稱和維度
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    
    # 準備數據
    h_input = np.random.random(input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)
    
    # 分配 GPU 顯存
    d_input = ctypes.c_void_p()
    d_output = ctypes.c_void_p()
    check_cuda_err(cuda_rt.cudaMalloc(ctypes.byref(d_input), h_input.nbytes))
    check_cuda_err(cuda_rt.cudaMalloc(ctypes.byref(d_output), h_output.nbytes))
    
    # 數據傳輸: Host -> Device
    check_cuda_err(cuda_rt.cudaMemcpy(d_input, h_input.ctypes.data, h_input.nbytes, 1)) # 1 是 HostToDevice
    
    # 設置推理地址
    context.set_tensor_address(input_name, d_input.value)
    context.set_tensor_address(output_name, d_output.value)
    
    # 推理
    print("開始推理...")
    context.execute_v3(0)
    
    # 數據傳輸: Device -> Host
    check_cuda_err(cuda_rt.cudaMemcpy(h_output.ctypes.data, d_output, h_output.nbytes, 2)) # 2 是 DeviceToHost
    
    print(f"✅ 驗證完成！")
    print(f"輸入 {input_name} 維度: {input_shape}")
    print(f"輸出 {output_name} 維度: {output_shape}")
    print(f"輸出前 5 個數值: {h_output.flatten()[:5]}")

    # 釋放顯存
    cuda_rt.cudaFree(d_input)
    cuda_rt.cudaFree(d_output)

if __name__ == "__main__":
    verify_engine_no_pycuda(r"F:\python_code\onnx_to_engine\model.engine")