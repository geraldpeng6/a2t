# Audio to SRT Converter

这个程序使用 OpenAI 的 Whisper 模型将音频文件转换为带时间轴的 SRT 字幕文件，支持中文和英文。

## 环境配置

使用 Conda 创建新环境并安装依赖：

```bash
conda create -n a2t python=3.10
conda activate a2t
pip install -r requirements.txt
```

## 使用方法

基本用法：
```bash
python audio_to_srt.py 音频文件路径
```

完整参数：
```bash
python audio_to_srt.py 音频文件路径 [--output 输出文件路径] [--model 模型大小]
```

参数说明：
- `音频文件路径`：必需，输入的音频文件路径
- `--output` 或 `-o`：可选，输出的 SRT 文件路径（默认为输入文件同名，扩展名改为.srt）
- `--model` 或 `-m`：可选，Whisper 模型大小，可选值：tiny/base/small/medium/large（默认为base）

示例：
```bash
# 基本用法
python audio_to_srt.py input.mp3

# 指定输出文件
python audio_to_srt.py input.mp3 -o output.srt

# 使用大型模型（更准确但更慢）
python audio_to_srt.py input.mp3 -m large
```

## 支持的功能

- 支持多种音频格式（mp3, wav, m4a 等）
- 支持中文和英文语音识别
- 自动生成带时间轴的 SRT 格式字幕
- 支持 Apple Silicon (M1) GPU 加速
- 可选择不同大小的模型以平衡准确度和速度

## 注意事项

1. 首次运行时会自动下载选择的 Whisper 模型
2. 在 M1 Mac 上会自动使用 MPS (Metal Performance Shaders) 进行加速
3. 模型大小说明：
   - tiny: 最快但准确度最低
   - base: 平衡速度和准确度（推荐新手使用）
   - small: 比 base 更准确
   - medium: 更高的准确度
   - large: 最准确但最慢，需要更多内存
