# -*- coding: utf-8 -*-
"""
WAV 读写 + 采样率重采样（纯 numpy + 标准库，无额外依赖）。
- read_wav(path)          -> (float32 [2,T], sr)；支持 PCM(8/16/24/32) 与 IEEE float(32/64)
- decode_audio(path)      -> (float32 [2,T], 44100)；非 WAV 用系统 ffmpeg 转码，采样率归一到 44100
- write_pcm16(path,data,sr) 写 16-bit PCM
- write_float32(path,data,sr)写 32-bit IEEE float
"""
import os
import subprocess
import struct
import sys
import wave

import numpy as np

SR_TARGET = 44100

# 运行时创建子进程时抑制控制台窗口（ffmpeg 是控制台程序，不抑制会闪现黑框）
_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0) or 0


def _find_ffmpeg():
    """优先使用工具包自带 runtime 里的 ffmpeg，其次系统 PATH 中的 ffmpeg。
    兼容 PyInstaller 打包：bundle 内 ffmpeg 位于 _MEIPASS/runtime/ffmpeg.exe。"""
    here = os.path.dirname(os.path.abspath(__file__))
    meipass = getattr(sys, '_MEIPASS', None)
    for root in (meipass, here):
        if not root:
            continue
        cand = os.path.join(root, 'runtime', 'ffmpeg.exe')
        if os.path.exists(cand):
            return cand
    return 'ffmpeg'


# ---------------- WAV 读取 ----------------
def _read_wav_header(fp):
    """返回 (fmt_kwargs, data_bytes)。全部手动解析以支持 float 与 extensible。"""
    def rd(n):
        b = fp.read(n)
        if len(b) != n:
            raise IOError("WAV 文件不完整")
        return b

    assert rd(4) == b'RIFF'
    rd(4)  # size
    assert rd(4) == b'WAVE'
    fmt = None
    data = None
    while True:
        chunk = rd(4)
        if len(chunk) < 4:
            break
        size = struct.unpack('<I', rd(4))[0]
        body = rd(size) if chunk != b'data' or size % 2 == 0 else rd(size + 1)[:-1]
        if chunk == b'fmt ':
            fmt = body
        elif chunk == b'data':
            data = body
            break
    if fmt is None or data is None:
        raise IOError("WAV 缺少 fmt/data 块")
    return fmt, data


def _decode_fmt(fmt):
    code = struct.unpack('<H', fmt[0:2])[0]
    ch = struct.unpack('<H', fmt[2:4])[0]
    sr = struct.unpack('<I', fmt[4:8])[0]
    bits = struct.unpack('<H', fmt[14:16])[0]
    if code == 0xFFFE and len(fmt) >= 40:  # extensible
        sub = fmt[24:28]
        if sub == b'\x01\x00\x00\x00':
            code = 1
        elif sub == b'\x03\x00\x00\x00':
            code = 3
    return code, ch, sr, bits


def read_wav(path):
    with open(path, 'rb') as f:
        fmt, data = _read_wav_header(f)
    code, ch, sr, bits = _decode_fmt(fmt)
    raw = np.frombuffer(data, dtype=np.uint8)

    if code == 3:  # IEEE float
        if bits == 32:
            a = raw.view(np.float32)
        elif bits == 64:
            a = raw.view(np.float64).astype(np.float32)
        else:
            raise IOError("不支持该 float 位深")
    elif code == 1:  # PCM int
        if bits == 8:
            a = raw.astype(np.float32) / 128.0 - 1.0
        elif bits == 16:
            a = raw.view(np.int16).astype(np.float32) / 32768.0
        elif bits == 24:
            # 小端 24-bit：三字节拼成 int，符号扩展
            a = (raw[0::3].astype(np.int32) * 65536
                 + raw[1::3].astype(np.int32) * 256
                 + raw[2::3].astype(np.int32))
            a = a.astype(np.float32)
            a = np.where(a > 8388607, a - 16777216, a).astype(np.float32) / 8388608.0
        elif bits == 32:
            a = raw.view(np.int32).astype(np.float32) / 2147483648.0
        else:
            raise IOError("不支持该 PCM 位深")
    else:
        raise IOError("不支持的 WAV 格式 code=%r" % code)

    if ch == 1:
        a = np.stack([a, a])
    else:
        a = a.reshape(-1, ch).T  # 交错 -> (ch, frames)
        a = a[:2]                # 只保留前两声道（若为多声道）
    # 确保是 (2,T)
    a = a.reshape(2, -1)
    return a.astype(np.float32), sr


# ---------------- 重采样（FFT 法） ----------------
def resample(x, sr_in, sr_out=SR_TARGET):
    """x:(2,T) float32 -> (2,T') 线性插值+FFT 抗混叠，直接 numpy 实现。"""
    if sr_in == sr_out:
        return x
    n_out = int(round(x.shape[1] * sr_out / sr_in))
    # FFT 频域重采样（窗口化 sinc 近似），对短文件稳定
    n_in = x.shape[1]
    n = max(n_in, n_out)
    f = np.fft.rfft(x, n=n, axis=1)
    if n_out < n_in:
        f = f[:, :n_out // 2 + 1] * 2
    else:
        pad = np.zeros((f.shape[0], n_out // 2 + 1 - f.shape[1]), dtype=f.dtype)
        f = np.concatenate([f, pad], axis=1)
    y = np.fft.irfft(f, n=n_out, axis=1)
    scale = n_out / float(n_in)
    y = y * scale
    return y.astype(np.float32)


# ---------------- 解码入口 ----------------
def decode_audio(path, sr_out=SR_TARGET):
    """任意输入 -> (float32 [2,T], sr_out)。WAV 原生读；其它格式调用系统 ffmpeg。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        x, sr = read_wav(path)
        return resample(x, sr, sr_out), sr_out
    # 其它格式：用 ffmpeg 转 44100/立体声/float
    tmp = path + '.tmp.wav'
    ffmpeg = _find_ffmpeg()
    try:
        r = subprocess.run(
            [ffmpeg, '-y', '-i', path,
             '-vn', '-ac', '2', '-ar', str(SR_TARGET),
             '-c:a', 'pcm_f32le', tmp],
            capture_output=True, timeout=1200, creationflags=_NO_WINDOW)
        if r.returncode != 0:
            raise IOError("ffmpeg 转码失败（要不要先转成 .wav 再上传？）")
        return read_wav(tmp)[0], SR_TARGET
    except FileNotFoundError:
        raise IOError("当前文件不是 WAV，且未找到 ffmpeg。请先用 ffmpeg 转成 WAV，"
                       "或安装 ffmpeg 后重试。")
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------- 原格式探测与转码 ----------------
_CODEC_ENCODER = {
    'mp3': 'libmp3lame', 'mp3float': 'libmp3lame',
    'flac': 'flac', 'aac': 'aac', 'alac': 'alac',
    'vorbis': 'libvorbis', 'ogg': 'libvorbis', 'opus': 'libopus',
    'wma': 'wmav2', 'wmav2': 'wmav2', 'ac3': 'ac3', 'eac3': 'eac3',
}


def probe_audio_format(path):
    """检测音频编码 / 码率 / 采样率，返回 (codec, bitrate_kbps, sr) 或 (None,None,None)。"""
    ffmpeg = _find_ffmpeg()
    r = subprocess.run([ffmpeg, '-i', path], capture_output=True,
                       creationflags=_NO_WINDOW)
    output = (r.stderr or b'').decode('utf-8', 'ignore')
    codec = bitrate = sr = None
    for line in output.split('\n'):
        if 'Audio:' not in line:
            continue
        info = line.split('Audio:', 1)[1].strip()
        codec = info.split(',')[0].strip().lower()
        for token in info.split(','):
            tok = token.strip()
            if 'kb/s' in tok:
                try:
                    bitrate = int(tok.replace('kb/s', '').strip())
                except ValueError:
                    pass
            m = re_search_rate(tok)
            if m:
                sr = int(m.group(1)) * 1000 if 'k' in m.group(0).lower() else int(m.group(1))
        break
    return codec, bitrate, sr


def re_search_rate(tok):
    import re
    return re.search(r'(\d+)\s*(?:k)?[hH][zZ]', tok)


def convert_same_format(wav_path, orig_path, out_path):
    """把分离后的 wav_path 按原文件(orig_path)的编码/码率/采样率转码输出到 out_path。"""
    codec, bitrate, sr = probe_audio_format(orig_path)
    enc = _CODEC_ENCODER.get(codec)
    if not enc:
        raise RuntimeError('不支持的音频编码: %s' % codec)
    ffmpeg = _find_ffmpeg()
    cmd = [ffmpeg, '-y', '-i', wav_path, '-map', '0:a:0', '-c:a', enc]
    if sr and codec not in ('alac', 'flac'):
        cmd += ['-ar', str(sr)]
    if bitrate and codec not in ('alac', 'flac', 'opus'):
        cmd += ['-b:a', '%dk' % bitrate]
    cmd += [out_path]
    r = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError('格式转换失败：%s' % (r.stderr or b'').decode('utf-8', 'ignore')[-500:])


# ---------------- WAV 写出 ----------------
def _write_header(path, sr, ch, bits, is_float, nframes):
    byte_rate = sr * ch * bits // 8
    with open(path, 'wb') as w:
        w.write(b'RIFF')
        w.write(struct.pack('<I', 36 + nframes * ch * bits // 8))
        w.write(b'WAVE')
        w.write(b'fmt ')
        code = 3 if is_float else 1
        w.write(struct.pack('<IHHIIHH',
                            16, code, ch, sr, byte_rate, ch * bits // 8, bits))
        w.write(b'data')
        w.write(struct.pack('<I', nframes * ch * bits // 8))


def write_wav(path, data, sr, bitdepth=32, is_float=True):
    """data: (2,T) float32。bitdepth: 16/24/32；is_float 决定 float 或 PCM。"""
    x = np.clip(data, -1.0, 1.0).astype(np.float32)
    n = x.shape[1]
    _write_header(path, sr, 2, bitdepth, is_float, n)
    with open(path, 'ab') as w:
        if is_float and bitdepth == 32:
            w.write(np.ascontiguousarray(x.T).tobytes())
        elif is_float and bitdepth == 64:
            w.write(np.ascontiguousarray(x.T).astype(np.float64).tobytes())
        elif not is_float and bitdepth == 16:
            pcm = (x.T * 32767.0).astype(np.int16)
            w.write(pcm.tobytes())
        elif not is_float and bitdepth == 24:
            pcm = np.clip(np.round(x.T * 8388607.0), -8388608, 8388607).astype(np.int32)
            out = np.zeros((pcm.shape[0], 3), dtype=np.uint8)
            for i in range(3):
                out[:, i] = ((pcm >> (8 * i)) & 0xFF).astype(np.uint8)
            w.write(out.tobytes())
        else:
            pcm = (x.T * 2147483647.0).astype(np.int32)
            w.write(pcm.tobytes())