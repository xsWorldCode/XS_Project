# ---------------------------------------------------------
# 第一阶段：基础环境 (Base)
# 使用 NVIDIA 官方提供的 PyTorch 运行时镜像，已包含 CUDA 12.x
# ---------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:24.01-py3

# 设置工作目录，后续所有操作都在 /app 下进行
WORKDIR /app

# ---------------------------------------------------------
# 第二阶段：系统依赖安装
# ---------------------------------------------------------
# 防止 apt-get 弹出交互式界面
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# 第三阶段：Python 环境固化
# ---------------------------------------------------------
# 先拷贝 requirements.txt，利用 Docker 缓存机制，
# 只要这个文件没变，下次 build 就会跳过安装过程，速度极快。
COPY requirements.txt .

# 建议使用国内源提速
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ---------------------------------------------------------
# 第四阶段：代码集成
# ---------------------------------------------------------
# 将当前目录的所有文件拷贝到容器 /app 目录
COPY . .

# 暴露端口（如果你需要运行 Jupyter 或 Tensorboard）
EXPOSE 8888 6006

# 默认启动命令
CMD ["python", "main.py"]