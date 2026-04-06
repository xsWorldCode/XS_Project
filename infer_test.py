import cv2
import numpy as np
from polygraphy.backend.trt import TrtRunner, EngineFromBytes

def predict_real_image(engine_path, image_path):
    # 1. 讀取與前處理圖片
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) # 縮放到模型要求的尺寸
    img = img.astype(np.float32) / 255.0 # 歸一化 [0, 1]
    
    # HWC (OpenCV) 轉為 CHW (TensorRT 格式)
    img = np.transpose(img, (2, 0, 1)) 
    # 添加 Batch 維度 -> (1, 3, 224, 224)
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img) # 確保內存連續

    # 2. 使用 Polygraphy 加載並運行 Engine
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    # 創建 Runner
    with TrtRunner(EngineFromBytes(engine_data)) as runner:
        # 執行推理
        outputs = runner.infer({"input": img})
        
        # 3. 獲取輸出結果
        output_tensor = outputs["output"] # 根據你日誌顯示的節點名 "output"
        print(f"📊 模型原始輸出 (Logits): {output_tensor}")
        
        # 得到分類結果
        probs = np.exp(output_tensor) / np.sum(np.exp(output_tensor)) # 簡單的 Softmax
        class_id = np.argmax(probs)
        conf = probs[0][class_id]
        
        print(f"✅ 預測結果: 類別 {class_id}, 置信度: {conf:.2%}")

if __name__ == "__main__":
    # 替換成你自己的圖片路徑
    predict_real_image(r"F:\python_code\onnx_to_engine\model.engine", r"F:\python_code\DataSets\split\val\Cat\cat.665.jpg")