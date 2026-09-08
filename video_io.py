# -*- coding: utf-8 -*-
"""
视频伴奏制作 —— 视频音频提取 / 音轨探测 / 伴奏合并。
业务逻辑参考《伴奏制作参考源码》(app.py)，仅复用其业务逻辑与预览逻辑，
依赖工具包自带的 ffmpeg，不使用参考源码所用的模型，保持工具包体积小巧。
"""
import os
import subprocess

import wav_io   # 复用其 _find_ffmpeg（优先工具包自带 runtime 的 ffmpeg）

_NW = getattr(subprocess, 'CREATE_NO_WINDOW', 0) or 0

VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv',
              '.webm', '.m4v', '.ts', '.m2ts', '.3gp', '.ogv'}


def is_video_file(path):
    """按扩展名判断是否为视频文件。"""
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def _ffmpeg():
    return wav_io._find_ffmpeg()


def _err(r):
    return r.stderr.decode('utf-8', 'ignore').strip()[-500:] if r.stderr else '未知错误'


def extract_audio(video_path, output_audio_path):
    """从视频提取 44100Hz / 立体声 / 16-bit WAV 音轨，供后续分离。"""
    cmd = [_ffmpeg(), '-y', '-i', video_path,
           '-vn', '-acodec', 'pcm_s16le',
           '-ar', '44100', '-ac', '2', output_audio_path]
    r = subprocess.run(cmd, capture_output=True, creationflags=_NW)
    if r.returncode != 0:
        raise RuntimeError('视频音频提取失败：%s' % _err(r))


def probe_audio(video_path):
    """检测视频第一音轨的编码与码率，返回 (codec, bitrate) 或 (None, None)。"""
    ffmpeg = _ffmpeg()
    r = subprocess.run([ffmpeg, '-i', video_path], capture_output=True, creationflags=_NW)
    output = (r.stderr or b'').decode('utf-8', 'ignore')
    codec = None
    bitrate = None
    for line in output.split('\n'):
        if 'Audio:' in line:
            audio_info = line.split('Audio:', 1)[1].strip()
            codec = audio_info.split(',')[0].strip().lower()
            for token in audio_info.split(','):
                tok = token.strip()
                if 'kb/s' in tok:
                    try:
                        bitrate = int(tok.replace('kb/s', '').strip())
                    except ValueError:
                        pass
            break
    return codec, bitrate


def merge_audio(video_path, audio_path, output_path, mode='add_track'):
    """把伴奏音轨合并进视频。
    mode='add_track'：保留原音轨，新增一条伴奏音轨；
    mode='replace'：仅保留伴奏音轨（覆盖原音轨）。
    视频流直接复制，伴奏按原音轨编码格式重新编码。"""
    orig_codec, orig_bitrate = probe_audio(video_path)

    if orig_codec == 'aac':
        audio_encoder = 'aac'
    elif orig_codec in ('mp3', 'mp3float'):
        audio_encoder = 'libmp3lame'
    else:
        audio_encoder = 'aac'

    if orig_bitrate and 96 <= orig_bitrate <= 192:
        audio_bitrate = '%dk' % orig_bitrate
    else:
        audio_bitrate = '128k'

    cmd = [_ffmpeg(), '-y', '-i', video_path, '-i', audio_path,
           '-map', '0:v:0', '-c:v', 'copy']
    if mode == 'replace':
        cmd += ['-map', '1:a:0', '-c:a', audio_encoder, '-b:a', audio_bitrate]
    else:
        cmd += ['-map', '0:a:0', '-map', '1:a:0',
                '-c:a:0', 'copy',
                '-c:a:1', audio_encoder, '-b:a:1', audio_bitrate]
    cmd += ['-shortest', output_path]

    r = subprocess.run(cmd, capture_output=True, creationflags=_NW)
    if r.returncode != 0:
        raise RuntimeError('音轨合并失败：%s' % _err(r))