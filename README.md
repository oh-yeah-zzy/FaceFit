# FaceFit

人脸匹配度识别工具 - 比较两张图片中人脸的相似度。

## 功能特点

- 输入两张人脸图片，输出相似度分数（0-100）
- 基于 InsightFace + ArcFace 模型，精度高
- 支持 CPU 和 GPU 运行
- 简洁的命令行界面

## 安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. (可选) 如果有NVIDIA GPU，安装GPU加速支持
pip install -r requirements-gpu.txt
```

## 使用方法

### 基本用法

```bash
python cli.py 图片1.jpg 图片2.jpg
```

### 命令行参数

```bash
# 使用GPU加速（默认）
python cli.py photo1.jpg photo2.jpg --gpu

# 强制使用CPU
python cli.py photo1.jpg photo2.jpg --cpu

# 使用轻量级模型（更快但精度略低）
python cli.py photo1.jpg photo2.jpg --model buffalo_s

# JSON格式输出
python cli.py photo1.jpg photo2.jpg --json

# 安静模式（只输出分数）
python cli.py photo1.jpg photo2.jpg -q

# 查看版本
python cli.py -v
```

### 输出示例

```
==================================================
           FaceFit 人脸匹配结果
==================================================

  相似度分数: 85.32 / 100
  原始相似度: 0.7064
  置信等级:   极高

  判定结果:   很可能是同一人

==================================================
```

## 作为Python模块使用

```python
from core import FaceMatcher

# 创建匹配器
matcher = FaceMatcher(model_name="buffalo_l", use_gpu=True)

# 比较两张图片
result = matcher.match("photo1.jpg", "photo2.jpg")

print(f"相似度分数: {result['score']}")
print(f"是否同一人: {result['is_same_person']}")
```

## 项目结构

```
FaceFit/
├── cli.py              # 命令行入口
├── core/
│   ├── __init__.py
│   └── face_matcher.py # 核心匹配逻辑
├── requirements.txt    # 依赖列表
├── requirements-gpu.txt # GPU依赖
└── README.md
```

## 模型说明

| 模型 | 精度 | 速度 | 适用场景 |
|------|------|------|----------|
| buffalo_l | 高 | 较慢 | 服务器/高精度需求 |
| buffalo_s | 中 | 快 | CPU/边缘设备 |

## 相似度阈值参考

| 分数范围 | 置信度 | 说明 |
|----------|--------|------|
| 80-100 | 极高 | 极可能是同一人 |
| 70-80 | 高 | 很可能是同一人 |
| 60-70 | 中等 | 可能是同一人 |
| 50-60 | 低 | 可能不是同一人 |
| <50 | 极低 | 很可能不是同一人 |
