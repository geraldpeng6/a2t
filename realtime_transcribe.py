import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import datetime
import torch
import time
import tkinter as tk
import logging
from tkinter import ttk
import os
import argparse

# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description="实时音频转字幕")
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="指定Whisper模型，默认是base"
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="指定音频语言的简写，例如'zh'表示中文，默认为None自动检测"
    )
    parser.add_argument(
        "-t", "--translate",
        action='store_true',
        help="如果设置此标志，将音频翻译成英文"
    )
    return parser.parse_args()

# 获取命令行参数
args = parse_arguments()

# 初始化whisper模型
model = whisper.load_model(args.model)

# 音频参数设置
SAMPLE_RATE = 16000
CHANNELS = 2  # 修改为2通道以支持立体声
DTYPE = np.float32
BLOCK_DURATION = 1  # 减小音频块时长为1秒
BUFFER_DURATION = 2  # 减小缓冲区大小为2秒

# 音频数据队列
audio_queue = queue.Queue()
# 转写文本队列
text_queue = queue.Queue()

# 设置日志记录
def setup_logging():
    """设置日志记录"""
    log_file = "transcription.log"
    
    # 只记录错误信息到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    
    # 记录所有信息到文件
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return log_file

def format_timestamp(seconds):
    """转换秒数为时间戳格式"""
    td = datetime.timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def find_blackhole_device():
    """查找BlackHole音频设备"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if 'BlackHole' in device['name']:
            return i
    return None

def audio_callback(indata, frames, time, status):
    """音频数据回调函数"""
    if status:
        logging.warning(f'音频回调状态: {status}')
    # 如果是立体声，将其转换为单声道
    if indata.shape[1] == 2:
        audio_data = np.mean(indata, axis=1, keepdims=True)
    else:
        audio_data = indata.copy()
    audio_queue.put(audio_data)

class SubtitleWindow:
    """字幕显示窗口类"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("实时字幕")
        
        # 设置窗口属性
        self.root.attributes('-topmost', True)  # 窗口置顶
        self.root.overrideredirect(True)        # 移除窗口边框
        
        # 获取屏幕尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 设置窗口大小和位置（底部居中）
        window_width = int(screen_width * 0.8)  # 使用80%的屏幕宽度
        window_height = 180
        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 100  # 距离底部100像素
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # 创建主框架
        self.main_frame = tk.Frame(self.root, bg='systemTransparent')
        self.main_frame.pack(fill='both', expand=True)
        
        # 创建字幕显示标签
        self.subtitle_label = tk.Label(
            self.main_frame,
            text="等待音频输入...",
            font=('SimHei', 24),  # 增大字体
            wraplength=window_width - 40,
            justify='center',
            fg='white',
            pady=20,
            highlightthickness=1,  # 添加描边效果
            highlightbackground='white'  # 设置描边颜色
        )
        self.subtitle_label.pack(expand=True)
        
        # 创建控制面板
        self.create_control_panel()
        
        # 绑定右键菜单用于退出
        self.create_context_menu()
        
        # 绑定拖动事件
        self.root.bind('<Button-1>', self.start_move)
        self.root.bind('<B1-Motion>', self.on_drag)
        
    def create_control_panel(self):
        """创建控制面板"""
        control_frame = tk.Frame(self.main_frame, bg='systemTransparent')
        control_frame.pack(side='bottom', fill='x')
        
        # 语言选择
        tk.Label(control_frame, text="语言:", bg='systemTransparent', fg='white').pack(side='left', padx=5)
        self.language_var = tk.StringVar(value='zh')
        language_menu = ttk.Combobox(control_frame, textvariable=self.language_var, values=[
            'zh', 'en', 'ja', 'fr', 'es', 'de', 'it', 'ko', 'ru', 'pt', 'nl', 'sv', 'fi'
        ])
        language_menu.pack(side='left')
        language_menu.bind('<<ComboboxSelected>>', self.update_language)
        
        # 翻译选项
        self.translate_var = tk.BooleanVar(value=args.translate)
        translate_check = tk.Checkbutton(control_frame, text="翻译", variable=self.translate_var, bg='systemTransparent', fg='white', command=self.update_translate)
        translate_check.pack(side='left', padx=5)
        
        # 透明度调整
        tk.Label(control_frame, text="透明度:", bg='systemTransparent', fg='white').pack(side='left', padx=5)
        self.alpha_scale = tk.Scale(control_frame, from_=0.0, to=1.0, resolution=0.1, orient='horizontal', bg='systemTransparent', fg='white', command=self.update_alpha)
        self.alpha_scale.set(0.0)
        self.alpha_scale.pack(side='left')
        
        # 字体颜色选择
        tk.Label(control_frame, text="颜色:", bg='systemTransparent', fg='white').pack(side='left', padx=5)
        self.color_var = tk.StringVar(value='white')
        color_menu = ttk.Combobox(control_frame, textvariable=self.color_var, values=['white', 'yellow', 'cyan', 'green', 'red'])
        color_menu.pack(side='left')
        color_menu.bind('<<ComboboxSelected>>', self.update_color)
        
    def update_language(self, event):
        """更新语言设置"""
        args.language = self.language_var.get()
        logging.info(f"语言设置为: {args.language}")
        
    def update_translate(self):
        """更新翻译设置"""
        args.translate = self.translate_var.get()
        logging.info(f"翻译设置为: {args.translate}")
        
    def update_alpha(self, value):
        """更新窗口透明度"""
        self.main_frame.configure(bg=f'#{int(float(value) * 255):02x}000000')
        
    def update_color(self, event):
        """更新字幕颜色"""
        color = self.color_var.get()
        self.subtitle_label.config(fg=color)
        logging.info(f"字幕颜色设置为: {color}")
        
    def create_context_menu(self):
        """创建右键菜单"""
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="退出", command=self.root.quit)
        self.root.bind('<Button-3>', self.show_context_menu)
        
    def show_context_menu(self, event):
        """显示右键菜单"""
        self.context_menu.post(event.x_root, event.y_root)
        
    def start_move(self, event):
        """开始拖动"""
        self.x = event.x
        self.y = event.y
        
    def on_drag(self, event):
        """处理拖动"""
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f'+{x}+{y}')
    
    def update_subtitle(self, text):
        """更新字幕文本"""
        self.subtitle_label.config(text=text)
        self.root.update()

def process_audio():
    """处理音频数据并进行转写"""
    buffer = np.array([], dtype=DTYPE)
    last_process_time = 0
    
    while True:
        try:
            # 获取音频数据
            audio_data = audio_queue.get(timeout=1.0)  # 设置超时，避免无限等待
            current_time = time.time()
            
            # 将新数据添加到缓冲区
            buffer = np.append(buffer, audio_data.flatten())
            
            # 如果缓冲区达到处理长度且距离上次处理已经过去了足够时间
            if (len(buffer) >= SAMPLE_RATE * BUFFER_DURATION and 
                current_time - last_process_time >= BLOCK_DURATION):
                
                # 归一化音频
                audio_data = buffer / (np.max(np.abs(buffer)) + 1e-10)
                
                # 设置任务类型
                task = "translate" if args.translate else "transcribe"
                
                # 转写或翻译音频
                result = model.transcribe(
                    audio_data, 
                    language=args.language,
                    task=task
                )
                
                if result["text"].strip():
                    text_data = {
                        'text': result["text"].strip()
                    }
                    text_queue.put(text_data)
                    
                    # 记录识别结果到日志
                    logging.info(f"识别结果: {result['text'].strip()}")
                
                # 更新处理时间
                last_process_time = current_time
                
                # 清空缓冲区，保留最后一小段以保持连续性
                overlap_samples = int(SAMPLE_RATE * 0.1)  # 保留0.1秒的重叠
                buffer = buffer[-overlap_samples:] if len(buffer) > overlap_samples else np.array([], dtype=DTYPE)
                
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"音频处理错误: {e}")
            time.sleep(0.1)  # 发生错误时短暂暂停

def update_subtitles(window):
    """更新字幕显示"""
    try:
        while True:
            try:
                subtitle = text_queue.get(timeout=0.5)  # 设置超时，避免无限等待
                window.update_subtitle(subtitle['text'])
            except queue.Empty:
                continue
    except Exception as e:
        logging.error(f"字幕更新错误: {e}")

def main():
    try:
        # 设置日志
        log_file = setup_logging()
        logging.info("开始实时转写...")
        
        # 查找BlackHole设备
        device_id = find_blackhole_device()
        if device_id is None:
            logging.error("未找到BlackHole虚拟音频设备。请确保已安装BlackHole并将系统音频输出设置为BlackHole。")
            print("\n请按照以下步骤设置：")
            print("1. 安装BlackHole: brew install blackhole-2ch")
            print("2. 在系统偏好设置 > 声音 > 输出 中选择'BlackHole 2ch'")
            print("3. 重新运行此程序")
            return
        
        # 创建字幕窗口
        subtitle_window = SubtitleWindow()
        
        # 启动音频处理线程
        audio_thread = threading.Thread(target=process_audio)
        audio_thread.daemon = True
        audio_thread.start()
        
        # 启动字幕更新线程
        subtitle_thread = threading.Thread(target=update_subtitles, args=(subtitle_window,))
        subtitle_thread.daemon = True
        subtitle_thread.start()
        
        # 启动音频流
        with sd.InputStream(
            device=device_id,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype=DTYPE,
            blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
            callback=audio_callback
        ):
            logging.info(f"实时转写已启动。日志文件保存在: {log_file}")
            print("实时字幕已启动。右键点击字幕窗口可以退出程序。")
            subtitle_window.root.mainloop()
                
    except KeyboardInterrupt:
        logging.info("停止转写...")
    except Exception as e:
        logging.error(f"主程序错误: {e}")

if __name__ == "__main__":
    main()
