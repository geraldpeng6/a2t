import whisper
import argparse
import datetime
import torch
import os
from tqdm import tqdm
import time

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    td = datetime.timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def create_srt(segments, output_file):
    """Create SRT file from segments"""
    with open(output_file, 'w', encoding='utf-8') as f:
        last_end = 0
        for segment in segments:
            # Write timestamps
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            
            # 如果与上一个字幕的间隔小于2秒，增加间隔
            if segment['start'] - last_end < 2:
                segment['start'] = last_end + 2
            
            # 确保每个字幕至少显示3秒
            if segment['end'] - segment['start'] < 3:
                segment['end'] = segment['start'] + 3
            
            f.write(f"{end_time}:")
            f.write(f"{segment['text'].strip()}\n")
            
            last_end = segment['end']

def transcribe_audio(audio_path, output_path, model_name="large", language=None):
    """
    Transcribe audio file to SRT format
    """
    start_time = time.time()
    print(f"正在加载模型 {model_name}...")
    
    # 加载模型并配置
    try:
        with tqdm(total=100, desc="加载模型", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            model = whisper.load_model(model_name)
            pbar.update(100)
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise
    
    print("\n开始转录音频...")
    try:
        with tqdm(total=100, desc="转录进度", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            def progress_callback(current, total):
                pbar.n = int(current * 100 / total)
                pbar.refresh()
            
            result = model.transcribe(
                audio_path,
                task="transcribe",
                verbose=False,
                language=language,
                beam_size=5,
                condition_on_previous_text=True,
                initial_prompt="以下是音频内容的转录：",
                progress_callback=progress_callback
            )
            pbar.n = 100
            pbar.refresh()
    except Exception as e:
        print(f"转录失败: {e}")
        print("使用基本配置重试...")
        try:
            result = model.transcribe(
                audio_path,
                language=language,
                verbose=False
            )
        except Exception as e:
            print(f"转录再次失败: {e}")
            raise
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n正在生成字幕文件...")
    create_srt(result["segments"], output_path)
    print(f"转录完成！字幕文件已保存到: {output_path}")
    print(f"总耗时: {int(total_time//60)}分{int(total_time%60)}秒")

def main():
    parser = argparse.ArgumentParser(description="使用 Whisper 将音频转换为 SRT 字幕")
    parser.add_argument("audio_path", help="输入音频文件的路径")
    parser.add_argument("--output", "-o", help="输出 SRT 文件的路径（默认：与输入文件同名加时间戳）")
    parser.add_argument("--model", "-m", default="large",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Whisper 模型大小（默认：large）")
    parser.add_argument("--language", "-l", default="zh",
                        help="指定语言（例如：'zh' 为中文，'en' 为英文）（默认：zh）")

    args = parser.parse_args()

    # If output path is not specified, use input filename with timestamp
    if not args.output:
        base_name = os.path.splitext(args.audio_path)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{base_name}_{timestamp}.srt"

    transcribe_audio(args.audio_path, args.output, args.model, args.language)

if __name__ == "__main__":
    main()
