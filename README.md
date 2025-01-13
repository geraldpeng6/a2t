# Audio/Video to SRT (音视频转字幕工具)

这是一个基于 Whisper 的音视频转字幕工具，支持中文和日语的转写，以及转写内容翻译为英文。

## 功能特点

- 支持多种音视频格式输入
- 支持中文和日语转写
- 支持将转写结果翻译为英文
- 支持指定时间范围进行处理
- 支持多种设备（CUDA/MPS/CPU）
- 自动检测最佳可用设备
- 输出标准 SRT 格式字幕文件

## 环境要求

- Python 3.8 或更高版本
- FFmpeg（用于音频处理）

## 安装

1. 克隆仓库：
```bash
git clone [repository-url]
cd a2t
```

2. 创建并激活虚拟环境（推荐）：
```bash
# 使用 conda
conda create -n a2t python=3.11
conda activate a2t

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

基本用法：
```bash
python audio_to_srt.py 视频文件.mp4
```

完整参数说明：
```bash
python audio_to_srt.py [-h] [--model {tiny,base,small,medium,large}] [--english]
                      [--language {zh,ja}] [--start START] [--end END]
                      audio_file
```

### 参数说明

- `audio_file`：输入的音视频文件路径
- `--model`, `-m`：选择 Whisper 模型大小（默认：base）
  - tiny：最快但准确度最低
  - base：平衡速度和准确度
  - small：准确度更好
  - medium：较高准确度
  - large：最高准确度但最慢
- `--english`, `-e`：将转写结果翻译为英文
- `--language`, `-l`：源语言，支持：
  - zh：中文（默认）
  - ja：日语
- `--start`, `-s`：开始时间（格式：HH:MM:SS、MM:SS 或秒数）
- `--end`, `-d`：结束时间（格式：HH:MM:SS、MM:SS 或秒数）

### 使用示例

1. 转写中文视频：
```bash
python audio_to_srt.py video.mp4
```

2. 转写日语视频：
```bash
python audio_to_srt.py video.mp4 -l ja
```

3. 转写并翻译为英文：
```bash
python audio_to_srt.py video.mp4 -l ja -e
```

4. 使用大模型提高准确度：
```bash
python audio_to_srt.py video.mp4 -m large
```

5. 只处理视频的特定时间段：
```bash
# 处理 1分30秒 到 2分钟 的部分
python audio_to_srt.py video.mp4 --start 1:30 --end 2:00

# 处理 1小时30分 到 2小时 的部分
python audio_to_srt.py video.mp4 --start 01:30:00 --end 02:00:00
```

## 输出文件

程序会在输入文件的同一目录下生成 SRT 格式的字幕文件，文件名格式为：
```
[原文件名]_[模型]_[语言]_[时间范围].srt
```

例如：
- `video_base_zh.srt`：中文转写
- `video_base_en.srt`：翻译为英文
- `video_base_ja_1-30_to_2-00.srt`：日语转写（指定时间段）

## 注意事项

1. 首次运行时会下载选定的 Whisper 模型
2. 转写速度取决于：
   - 选择的模型大小
   - 视频长度
   - 计算设备性能
3. 如果有 GPU（CUDA/MPS），程序会自动使用它来加速处理

## 常见问题

1. 如何选择合适的模型？
   - 短视频或对准确度要求不高：使用 tiny 或 base
   - 一般用途：使用 base 或 small
   - 需要高准确度：使用 medium 或 large

2. 为什么处理很慢？
   - 检查是否正在使用 CPU 而不是 GPU
   - 考虑使用更小的模型
   - 可以尝试只处理需要的时间段

3. 如何提高准确度？
   - 使用更大的模型（small/medium/large）
   - 确保音频质量良好
   - 正确选择源语言
