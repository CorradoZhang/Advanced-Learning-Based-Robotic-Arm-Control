"""在 macOS 上按设备名打开腕部相机实时预览。

运行：python open_camera.py
在预览窗口按 Q 或 Esc 退出。
依赖：FFmpeg 中的 ffplay（可通过 brew install ffmpeg 安装）。
"""

import argparse
import os
import shutil
import sys


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("参数必须大于 0")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, help="相机设备名称")
    parser.add_argument("--width", type=positive_int, default=640)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--fps", type=positive_int, default=30)
    args = parser.parse_args()

    if sys.platform != "darwin":
        parser.error("此脚本使用 macOS AVFoundation 相机后端")

    ffplay = shutil.which("ffplay")
    if ffplay is None:
        parser.error("找不到 ffplay，请先执行 brew install ffmpeg")

    command = [
        ffplay,
        "-hide_banner",
        "-loglevel", "warning",
        "-f", "avfoundation",
        "-pixel_format", "uyvy422",
        "-framerate", str(args.fps),
        "-video_size", f"{args.width}x{args.height}",
        "-i", f"{args.camera}:none",
        "-window_title", f"SO101 - {args.camera}",
    ]
    print(f"打开 {args.camera}，{args.width}×{args.height}，{args.fps} FPS", flush=True)
    print("在预览窗口按 Q 或 Esc 退出。录制数据前请先关闭预览。", flush=True)
    os.execv(ffplay, command)


if __name__ == "__main__":
    main()
