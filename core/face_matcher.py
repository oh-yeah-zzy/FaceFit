"""
FaceFit 人脸匹配核心模块

使用 InsightFace + ArcFace 模型实现人脸特征提取和相似度计算。
支持 CPU 和 GPU 运行环境。
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import cv2


class FaceMatchError(Exception):
    """人脸匹配过程中的错误基类"""
    pass


class NoFaceDetectedError(FaceMatchError):
    """未检测到人脸"""
    pass


class MultipleFacesError(FaceMatchError):
    """检测到多个人脸（可选择性抛出）"""
    pass


class FaceMatcher:
    """
    人脸匹配器

    使用 InsightFace 的 ArcFace 模型进行人脸检测、特征提取和相似度计算。

    Attributes:
        model_name: 使用的模型包名称，默认 "buffalo_l"（高精度）或 "buffalo_s"（轻量级）
        providers: ONNX Runtime 推理提供者，默认自动选择（优先GPU）
        det_size: 检测器输入尺寸，默认 (640, 640)
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: Optional[list] = None,
        det_size: Tuple[int, int] = (640, 640),
        use_gpu: bool = True
    ):
        """
        初始化人脸匹配器

        Args:
            model_name: 模型包名称
                - "buffalo_l": 高精度模型，适合服务器环境
                - "buffalo_s": 轻量模型，适合CPU/边缘设备
            providers: ONNX推理提供者列表，None则自动选择
            det_size: 人脸检测器输入尺寸
            use_gpu: 是否优先使用GPU（如果可用）
        """
        self.model_name = model_name
        self.det_size = det_size

        # 配置推理提供者
        if providers is None:
            if use_gpu:
                # 优先尝试CUDA，不可用则回退到CPU
                self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                self.providers = ['CPUExecutionProvider']
        else:
            self.providers = providers

        # 延迟加载模型（首次使用时加载）
        self._app = None

    def _load_model(self):
        """延迟加载 InsightFace 模型"""
        if self._app is not None:
            return

        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "请先安装 insightface: pip install insightface onnxruntime"
            )

        # 初始化人脸分析器
        self._app = FaceAnalysis(
            name=self.model_name,
            providers=self.providers
        )

        # 准备检测器，设置检测尺寸
        self._app.prepare(ctx_id=0, det_size=self.det_size)

    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        加载图片文件

        Args:
            image_path: 图片路径

        Returns:
            BGR格式的numpy数组

        Raises:
            FileNotFoundError: 图片文件不存在
            ValueError: 无法读取图片
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {path}")

        # 使用OpenCV读取图片（BGR格式）
        img = cv2.imread(str(path))

        if img is None:
            raise ValueError(f"无法读取图片（可能格式不支持或文件损坏）: {path}")

        return img

    def _detect_face(
        self,
        image: np.ndarray,
        image_name: str = "image"
    ) -> np.ndarray:
        """
        检测人脸并提取特征向量

        Args:
            image: BGR格式的图片数组
            image_name: 图片标识名（用于错误信息）

        Returns:
            L2归一化的人脸特征向量（512维）

        Raises:
            NoFaceDetectedError: 未检测到人脸
        """
        self._load_model()

        # 检测人脸
        faces = self._app.get(image)

        if len(faces) == 0:
            raise NoFaceDetectedError(f"在 {image_name} 中未检测到人脸")

        # 如果检测到多个人脸，选择面积最大的那个（通常是主要人物）
        if len(faces) > 1:
            # 按人脸框面积排序，选择最大的
            faces = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True
            )

        # 返回最大人脸的特征向量（已经是L2归一化的）
        return faces[0].embedding

    def get_embedding(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        获取图片中人脸的特征向量

        Args:
            image_path: 图片路径

        Returns:
            512维的人脸特征向量
        """
        image = self._load_image(image_path)
        return self._detect_face(image, str(image_path))

    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        计算两个特征向量的余弦相似度

        Args:
            embedding1: 第一个特征向量
            embedding2: 第二个特征向量

        Returns:
            余弦相似度值，范围 [-1, 1]
        """
        # 确保向量已归一化
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 计算余弦相似度（归一化向量的点积）
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        return float(similarity)

    def match(
        self,
        image1_path: Union[str, Path],
        image2_path: Union[str, Path]
    ) -> dict:
        """
        比较两张图片中人脸的相似度

        这是主要的对外接口，输入两张图片路径，返回相似度信息。

        Args:
            image1_path: 第一张图片的路径
            image2_path: 第二张图片的路径

        Returns:
            包含匹配结果的字典:
            {
                "similarity": float,      # 原始余弦相似度 [-1, 1]
                "score": float,           # 映射到 [0, 100] 的分数
                "is_same_person": bool,   # 是否判定为同一人（阈值0.4）
                "confidence": str         # 置信度描述
            }
        """
        # 提取两张图片的人脸特征
        embedding1 = self.get_embedding(image1_path)
        embedding2 = self.get_embedding(image2_path)

        # 计算余弦相似度
        similarity = self.calculate_similarity(embedding1, embedding2)

        # 将相似度映射到 0-100 分数
        # 余弦相似度范围是 [-1, 1]，映射到 [0, 100]
        score = (similarity + 1) / 2 * 100

        # 判断是否为同一人（使用常见阈值0.4，对应余弦相似度）
        # 注意：实际应用中建议根据业务场景调整阈值
        threshold = 0.4
        is_same_person = similarity >= threshold

        # 生成置信度描述
        if similarity >= 0.6:
            confidence = "极高"
        elif similarity >= 0.5:
            confidence = "高"
        elif similarity >= 0.4:
            confidence = "中等"
        elif similarity >= 0.3:
            confidence = "低"
        else:
            confidence = "极低"

        return {
            "similarity": round(similarity, 4),
            "score": round(score, 2),
            "is_same_person": is_same_person,
            "confidence": confidence
        }

    def match_batch(
        self,
        image_pairs: list
    ) -> list:
        """
        批量比较多对图片

        Args:
            image_pairs: 图片路径对的列表 [(path1, path2), ...]

        Returns:
            匹配结果列表
        """
        results = []
        for path1, path2 in image_pairs:
            try:
                result = self.match(path1, path2)
                result["image1"] = str(path1)
                result["image2"] = str(path2)
                result["success"] = True
            except FaceMatchError as e:
                result = {
                    "image1": str(path1),
                    "image2": str(path2),
                    "success": False,
                    "error": str(e)
                }
            results.append(result)

        return results
