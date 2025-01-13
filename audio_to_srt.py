#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音视频文件转写/翻译工具
可以将音视频文件转写为字幕文件(srt格式)，支持中日文转写和英文翻译
"""

import os
import time
import warnings
import argparse
from datetime import timedelta

import torch
import whisper
import librosa
import soundfile as sf

def format_timestamp(seconds):
    """将秒数转换为 SRT 时间戳格式 (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours = td.seconds//3600
    minutes = (td.seconds//60)%60
    seconds = td.seconds%60
    milliseconds = td.microseconds//1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def parse_time(time_str):
    """将时间字符串转换为秒数
    支持的格式：
    - HH:MM:SS (01:30:00)
    - MM:SS (5:30)
    - 秒数 (90)
    """
    if not time_str:
        return None
    try:
        parts = time_str.split(':')
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        else:  # 秒数
            return float(time_str)
    except:
        raise ValueError(f"无效的时间格式: {time_str}. 请使用 HH:MM:SS、MM:SS 或秒数")

def get_device_info():
    """检测可用的计算设备（CUDA/MPS/CPU）"""
    if torch.cuda.is_available():
        return "CUDA"
    elif torch.backends.mps.is_available():
        return "MPS"
    return "CPU"

def load_audio(audio_path, start_time=None, end_time=None):
    """加载音频文件，支持指定时间范围
    
    参数：
        audio_path: 音频文件路径
        start_time: 开始时间（可选）
        end_time: 结束时间（可选）
    
    返回：
        temp_path: 临时音频文件路径
        start_sec: 开始时间（秒）
    """
    # 获取音频时长
    duration = librosa.get_duration(path=audio_path)
    
    # 将时间字符串转换为秒数
    start_sec = parse_time(start_time) if start_time else 0
    end_sec = parse_time(end_time) if end_time else duration
    
    # 验证时间范围
    if start_sec is not None and end_sec is not None:
        if start_sec >= end_sec:
            raise ValueError("开始时间必须小于结束时间")
        if start_sec < 0:
            raise ValueError("开始时间不能为负数")
        if end_sec > duration:
            raise ValueError(f"结束时间超出音频时长 ({duration:.2f} 秒)")
    
    # 加载音频片段
    y, sr = librosa.load(audio_path, sr=16000, offset=start_sec, duration=end_sec-start_sec)
    
    # 保存临时文件
    temp_path = f"{audio_path}_temp.wav"
    sf.write(temp_path, y, sr)
    
    return temp_path, start_sec

def transcribe_and_translate(audio_path, model_name="base", task="transcribe", source_lang="zh", 
                           start_time=None, end_time=None):
    """转写音频并可选择翻译为英文
    
    参数：
        audio_path: 音频文件路径
        model_name: Whisper模型名称
        task: 任务类型（transcribe=转写/translate=翻译）
        source_lang: 源语言（zh=中文/ja=日文/en=英文）
        start_time: 开始时间（可选）
        end_time: 结束时间（可选）
    """
    # 屏蔽警告信息
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # 加载音频片段
    temp_audio, start_offset = load_audio(audio_path, start_time, end_time)
    
    try:
        # 显示使用的设备
        device = get_device_info()
        print(f"使用设备: {device}")
        
        # 加载模型
        print("加载模型...", end='', flush=True)
        start_time = time.time()
        model = whisper.load_model(model_name)
        model_load_time = time.time() - start_time
        print(f"\r模型加载完成，用时 {model_load_time:.1f}秒")
        
        # 转写
        print("开始转写...", end='', flush=True)
        transcribe_start = time.time()
        
        # 重定向stdout以抑制Whisper的输出
        import sys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            result = model.transcribe(
                temp_audio,
                language=source_lang,
                task=task,
                verbose=False
            )
        finally:
            sys.stdout = original_stdout
        
        # 打印总用时
        total_time = time.time() - start_time
        print(f"\r转写完成，总用时 {total_time:.1f}秒")
        
        # 如果指定了开始时间，调整时间戳
        if start_offset > 0:
            for segment in result['segments']:
                segment['start'] += start_offset
                segment['end'] += start_offset
        
        return result
    finally:
        # 清理临时文件
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

def save_as_srt(result, output_path):
    """将转写/翻译结果保存为SRT文件
    
    参数：
        result: Whisper转写结果
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], 1):
            # 写入字幕序号
            f.write(f"{i}\n")
            
            # 写入时间戳
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            f.write(f"{start_time} --> {end_time}\n")
            
            # 写入文本
            f.write(f"{segment['text'].strip()}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="将音视频转写为SRT字幕文件，支持翻译为英文")
    parser.add_argument("audio", help="音视频文件路径")
    parser.add_argument("--model", "-m", default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper模型大小")
    parser.add_argument("--english", "-e", action="store_true",
                       help="翻译为英文（默认保持原语言）")
    parser.add_argument("--language", "-l", default="zh",
                       choices=["zh", "ja", "en"],
                       help="源语言：zh（中文）、ja（日文）或en（英文）")
    parser.add_argument("--start", "-s", 
                       help="开始时间（格式：HH:MM:SS、MM:SS或秒数）。例如：01:30:00、5:30、90")
    parser.add_argument("--end", "-d", 
                       help="结束时间（格式：HH:MM:SS、MM:SS或秒数）。例如：02:30:00、10:30、150")
    args = parser.parse_args()
    
    # 生成输出文件名
    base_name = os.path.splitext(args.audio)[0]
    task = "translate" if args.english else "transcribe"
    out_lang = "en" if args.english else args.language
    
    # 添加时间范围到文件名
    time_suffix = ""
    if args.start or args.end:
        start_str = args.start.replace(":", "-") if args.start else "start"
        end_str = args.end.replace(":", "-") if args.end else "end"
        time_suffix = f"_{start_str}_to_{end_str}"
    
    output_path = f"{base_name}_{args.model}_{out_lang}{time_suffix}.srt"
    
    # 执行转写/翻译
    result = transcribe_and_translate(
        args.audio,
        model_name=args.model,
        task=task,
        source_lang=args.language,
        start_time=args.start,
        end_time=args.end
    )
    
    # 保存结果
    save_as_srt(result, output_path)
    print(f"字幕文件已保存至: {output_path}")

if __name__ == "__main__":
    main()
