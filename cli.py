#!/usr/bin/env python3
"""
FaceFit CLI - 人脸匹配度识别命令行工具

使用方法:
    python cli.py image1.jpg image2.jpg
    python cli.py image1.jpg image2.jpg --gpu
    python cli.py image1.jpg image2.jpg --model buffalo_s
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from core.face_matcher import FaceMatcher, NoFaceDetectedError, FaceMatchError


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="facefit",
        description="FaceFit - 人脸匹配度识别工具",
        epilog="示例: python cli.py photo1.jpg photo2.jpg --gpu"
    )

    # 必需参数：两张图片路径
    parser.add_argument(
        "image1",
        type=str,
        help="第一张图片的路径"
    )

    parser.add_argument(
        "image2",
        type=str,
        help="第二张图片的路径"
    )

    # 可选参数
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="使用GPU加速（如果可用，默认启用）"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用CPU运行"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="buffalo_l",
        choices=["buffalo_l", "buffalo_s"],
        help="选择模型: buffalo_l (高精度) 或 buffalo_s (轻量级)，默认 buffalo_l"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="以JSON格式输出结果"
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="安静模式，只输出分数"
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="FaceFit v1.0.0"
    )

    return parser


def format_result(result: dict, json_output: bool = False, quiet: bool = False) -> str:
    """
    格式化输出结果

    Args:
        result: 匹配结果字典
        json_output: 是否以JSON格式输出
        quiet: 安静模式，只输出分数

    Returns:
        格式化后的字符串
    """
    if quiet:
        return str(result["score"])

    if json_output:
        import json
        return json.dumps(result, ensure_ascii=False, indent=2)

    # 默认友好格式输出
    lines = [
        "",
        "=" * 50,
        "           FaceFit 人脸匹配结果",
        "=" * 50,
        "",
        f"  相似度分数: {result['score']:.2f} / 100",
        f"  原始相似度: {result['similarity']:.4f}",
        f"  置信等级:   {result['confidence']}",
        "",
    ]

    # 判断结果
    if result["is_same_person"]:
        lines.append("  判定结果:   很可能是同一人")
    else:
        lines.append("  判定结果:   可能不是同一人")

    lines.extend([
        "",
        "=" * 50,
        ""
    ])

    return "\n".join(lines)


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 验证图片文件存在
    image1_path = Path(args.image1)
    image2_path = Path(args.image2)

    if not image1_path.exists():
        print(f"错误: 图片文件不存在 - {image1_path}", file=sys.stderr)
        sys.exit(1)

    if not image2_path.exists():
        print(f"错误: 图片文件不存在 - {image2_path}", file=sys.stderr)
        sys.exit(1)

    # 确定是否使用GPU
    use_gpu = args.gpu and not args.cpu

    if not args.quiet:
        device = "GPU" if use_gpu else "CPU"
        print(f"\n正在加载模型 ({args.model})，使用 {device} 运行...")

    try:
        # 创建匹配器
        matcher = FaceMatcher(
            model_name=args.model,
            use_gpu=use_gpu
        )

        # 执行匹配
        if not args.quiet:
            print("正在分析人脸...")

        result = matcher.match(image1_path, image2_path)

        # 输出结果
        output = format_result(result, args.json, args.quiet)
        print(output)

    except NoFaceDetectedError as e:
        print(f"\n错误: {e}", file=sys.stderr)
        print("提示: 请确保图片中有清晰可见的人脸", file=sys.stderr)
        sys.exit(2)

    except FaceMatchError as e:
        print(f"\n人脸匹配错误: {e}", file=sys.stderr)
        sys.exit(3)

    except ImportError as e:
        print(f"\n依赖缺失: {e}", file=sys.stderr)
        print("\n请安装必要的依赖:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        sys.exit(4)

    except Exception as e:
        print(f"\n未知错误: {e}", file=sys.stderr)
        sys.exit(5)


if __name__ == "__main__":
    main()
