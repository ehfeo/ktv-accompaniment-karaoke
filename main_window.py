# -*- coding: utf-8 -*-
"""
UVR-MDX-NET-Inst_HQ_3 分轨工具 —— 原生窗口版（wxPython）。

不依赖浏览器：直接在原生窗口中操作，文件/文件夹用系统对话框选择，
支持拖入文件排队处理。预览（音频/视频）调用系统默认播放器打开。

复用浏览器版同一套处理管线：separate_mdx(分离) / wav_io(解码与转码) /
video_io(视频音轨提取与合并)。
"""
import os
import shutil
import sys
import threading
import time
import uuid
import webbrowser

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import wx

from separate_mdx import MDXSeparator, MODEL_CONFIGS, DEFAULT_MODEL
import wav_io
import video_io

APP_NAME = '伴奏分离·和声版 52pojie出品'
APP_VERSION = 'v1.3.1'
GITHUB_URL = 'https://github.com/ehfeo/ktv-accompaniment-karaoke'
MODEL_NAME = 'UVR-MDX-NET-Inst_HQ_3'
AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus',
              '.wma', '.aiff', '.wv', '.ape'}

FROZEN = getattr(sys, 'frozen', False)
if FROZEN:
    # PyInstaller 打包后：可执行文件所在目录用于写临时产物；bundle 解包目录为 _MEIPASS
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
    _BUNDLE = getattr(sys, '_MEIPASS', APP_DIR)
else:
    APP_DIR = _HERE
    _BUNDLE = _HERE

# 模型路径改由 get_separator 按所选模型从 MODELS 目录解析

# 产物暂存目录：非打包用本目录 jobs/；打包后 exe 目录不一定可写，用系统临时目录
if FROZEN:
    import tempfile
    WORKDIR = os.path.join(tempfile.gettempdir(), 'uvr_window_%s' % os.getpid())
else:
    WORKDIR = os.path.join(_HERE, 'jobs')
os.makedirs(WORKDIR, exist_ok=True)


def _icon_path():
    """定位应用图标（窗口左上角用）。兼容源码运行与 PyInstaller 打包。"""
    for d in (getattr(sys, '_MEIPASS', None), _HERE, APP_DIR):
        if not d:
            continue
        p = os.path.join(d, 'app_icon.ico')
        if os.path.exists(p):
            return p
    return None


_JOBS_LOCK = threading.Lock()   # 串行化模型推理（CPU 单例）
_separator = None
_sep_model = None
_singleton_lock = threading.Lock()


def get_separator(model_name=DEFAULT_MODEL):
    """按模型名加载单例分离器；切换模型时重建，同模型复用。"""
    global _separator, _sep_model
    with _singleton_lock:
        if _separator is None or _sep_model != model_name:
            cfg = MODEL_CONFIGS[model_name]
            p = os.path.join(_BUNDLE, 'models', cfg['file'])
            if not os.path.exists(p):
                p = os.path.join(APP_DIR, 'models', cfg['file'])
            if not os.path.exists(p):
                raise RuntimeError('未找到模型文件：%s' % cfg['file'])
            _separator = MDXSeparator(p, cfg)
            _sep_model = model_name
    return _separator


def _is_media(fp):
    return video_io.is_video_file(fp) or os.path.splitext(fp)[1].lower() in AUDIO_EXTS


# ---------------- 任务模型 ----------------
class Task:
    def __init__(self, src_path):
        self.id = uuid.uuid4().hex[:12]
        self.src_path = os.path.abspath(src_path)
        self.name = os.path.basename(src_path)
        self.is_video = video_io.is_video_file(src_path)
        self.origin = os.path.splitext(self.name)[0]
        self.ext = os.path.splitext(self.src_path)[1].lower()
        self.src_dir = os.path.dirname(self.src_path)
        self.status = 'queued'        # queued/processing/done/error
        self.phase = '排队中'
        self.pct = 0
        self.error = None
        # 产物
        self.inst = None              # 伴奏 wav
        self.voc = None               # 人声 wav
        self.inst_orig = None         # 伴奏 原格式
        self.video_out = None         # 加伴奏/替换后的视频
        self.video_preview = None     # 纯伴奏视频


def _copy_to_src(src_dir, src_file, dst_name):
    """复制产物到源目录（源文件目录）。"""
    if not (src_dir and os.path.isdir(src_dir)):
        return
    try:
        shutil.copyfile(src_file, os.path.join(src_dir, dst_name))
    except OSError:
        pass


# ---------------- 处理管线（等价于浏览器版 _run_task） ----------------
def _process(task, video_mode, report, model_name=DEFAULT_MODEL):
    """串行处理单个任务。report(phase, pct) 通知进度（phase 为人类可读阶段描述）。"""
    try:
        if task.is_video:
            report('① 正在从视频提取音轨…', 2)
            src_wav = os.path.join(WORKDIR, task.id + '_src.wav')
            video_io.extract_audio(task.src_path, src_wav)
            mix_path = src_wav
        else:
            src_wav = None
            mix_path = task.src_path

        report('② 正在解码音频…', 8)
        mix, sr = wav_io.decode_audio(mix_path)

        _disp = 'KARA_2' if 'KARA' in model_name else 'Inst_HQ_3'
        report('③ 加载并分离伴奏 / 人声（%s，CPU 推理，请耐心等待）…' % _disp, 10)
        sep = get_separator(model_name)
        with _JOBS_LOCK:
            inst = sep.demix(mix, progress=lambda i, t: report(
                '③ 分离伴奏 / 人声（%s）…' % _disp, 10 + (0 if t <= 0 else int(i * 90 / t))))

        report('④ 正在写入音频文件…', 100)
        voc = mix - inst
        base = os.path.join(WORKDIR, task.id)
        wav_io.write_wav(base + '_inst.wav', inst, sr, bitdepth=16, is_float=False)
        wav_io.write_wav(base + '_voc.wav', voc, sr, bitdepth=16, is_float=False)
        task.inst = base + '_inst.wav'
        task.voc = base + '_voc.wav'

        if not task.is_video and task.ext != '.wav':
            # 音频：若原格式不是 WAV，额外生成一份同格式/码率/采样率伴奏
            try:
                report('⑤ 按原格式编码伴奏（转码中）…', 100)
                p = base + '_inst' + task.ext
                wav_io.convert_same_format(task.inst, task.src_path, p)
                task.inst_orig = p
            except Exception:
                task.inst_orig = None

        if task.is_video:
            # 视频：合并出“加伴奏/替换”成品 + 纯伴奏预览
            out_ext = os.path.splitext(task.src_path)[1] or '.mp4'
            vid_out = base + '_video' + out_ext
            preview = base + '_preview' + out_ext
            report('⑤ 正在把伴奏音轨合并进视频（成品）…', 100)
            video_io.merge_audio(task.src_path, task.inst, vid_out,
                                 mode=video_mode)
            report('⑥ 正在生成纯伴奏预览视频…', 100)
            video_io.merge_audio(task.src_path, task.inst, preview, mode='replace')
            task.video_out = vid_out
            task.video_preview = preview

        # 回写产物到源目录
        report('⑦ 正在回写产物到源目录…', 100)
        if task.is_video and task.video_out and os.path.exists(task.video_out):
            _copy_to_src(task.src_dir, task.video_out,
                         task.origin + '_加伴奏音轨' + os.path.splitext(task.video_out)[1])
        elif not task.is_video:
            _copy_to_src(task.src_dir, task.inst, task.origin + '_伴奏.wav')
            _copy_to_src(task.src_dir, task.voc, task.origin + '_人声.wav')
            if task.inst_orig and os.path.exists(task.inst_orig):
                _copy_to_src(task.src_dir, task.inst_orig, task.origin + '_伴奏' + task.ext)

        # 清理中间产物
        try:
            if src_wav and os.path.exists(src_wav):
                os.remove(src_wav)
        except OSError:
            pass
    except Exception as e:
        task.error = str(e)
        # 失败清理残留
        try:
            base = os.path.join(WORKDIR, task.id)
            for p in (base + '_inst.wav', base + '_voc.wav',
                      base + '_src.wav', task.video_out, task.video_preview):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
        except OSError:
            pass


# ---------------- 播放相关 ----------------
def _open_default(path):
    """调用系统默认程序打开（音频/视频预览、打开文件夹）。"""
    if path and os.path.exists(path):
        os.startfile(path)


def _run_hidden(cmd):
    """后台执行系统命令，不弹出控制台黑框（shutdown 关机/取消）。"""
    import subprocess
    try:
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        subprocess.Popen(cmd, shell=False,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=flags)
    except Exception:
        pass


class _ShutdownConfirmDialog(wx.Frame):
    """处理完成后的自动关机确认框（用 Frame 而非 Dialog，规避部分系统上
    Dialog 内按钮中文渲染异常；主窗口 Frame 内按钮已验证渲染正常）：
    60 秒倒计时，可取消。"""
    def __init__(self, parent, seconds=60):
        super().__init__(parent, title='即将自动关机', size=(430, 176),
                         style=wx.CAPTION | wx.CLOSE_BOX | wx.SYSTEM_MENU)
        self.remaining = seconds
        self._seconds = seconds

        panel = wx.Panel(self)
        sz = wx.BoxSizer(wx.VERTICAL)

        self.lbl = wx.StaticText(panel, label='', style=wx.ALIGN_CENTRE_HORIZONTAL)
        f = wx.Font(wx.NORMAL_FONT.GetPointSize() + 8, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        self.lbl.SetFont(f)
        self.lbl.SetForegroundColour(wx.Colour(51, 122, 183))
        self._update_label()
        sz.Add(self.lbl, 0, wx.LEFT | wx.RIGHT | wx.TOP, 18)

        hint = wx.StaticText(panel, label='所有任务已处理完成，倒计时结束将自动关闭电脑。\n如需取消，请在倒计时结束前点击“取消关机”。',
                             style=wx.ALIGN_CENTRE_HORIZONTAL)
        sz.Add(hint, 0, wx.LEFT | wx.RIGHT | wx.TOP, 6)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_go = wx.Button(panel, label='立即关机')
        self.btn_cancel = wx.Button(panel, label='取消关机')
        for b in (self.btn_go, self.btn_cancel):
            b.SetMinSize((150, 38))
        self.btn_cancel.SetDefault()
        btns.Add(self.btn_go, 1, wx.EXPAND, 0)
        btns.AddSpacer(12)
        btns.Add(self.btn_cancel, 1, wx.EXPAND, 0)
        sz.Add(btns, 0, wx.ALIGN_CENTRE_HORIZONTAL | wx.ALL, 14)

        panel.SetSizer(sz)

        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_tick, self._timer)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
        self.btn_go.Bind(wx.EVT_BUTTON, self._on_go)
        self.Bind(wx.EVT_CLOSE, self._on_cancel)
        self._timer.Start(1000)
        self.Centre()

    def _update_label(self):
        self.lbl.SetLabel('%d 秒' % self.remaining)

    def _shutdown(self):
        _run_hidden(['shutdown', '/s', '/t', '0'])

    def _cancel_guard(self):
        _run_hidden(['shutdown', '/a'])

    def _finish(self):
        try:
            self._timer.Stop()
        except Exception:
            pass
        self.Destroy()

    def _on_tick(self, e):
        self.remaining -= 1
        self._update_label()
        if self.remaining <= 0:
            self._shutdown()
            self._finish()

    def _on_cancel(self, e=None):
        self._cancel_guard()
        self._finish()

    def _on_go(self, e=None):
        self._shutdown()
        self._finish()


# ---------------- 主窗口 ----------------
class _MediaDropTarget(wx.FileDropTarget):
    """wxPython 中 FileDropTarget 是 C++ 抽象类，需在 Python 侧子类化。"""
    def __init__(self, frame):
        super().__init__()
        self._frame = frame

    def OnDropFiles(self, x, y, filenames):
        self._frame._on_drop(filenames)
        return True


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title=APP_NAME, size=(860, 620),
                         style=wx.DEFAULT_FRAME_STYLE)
        self.tasks = []               # 顺序：先入队在上（自顶向下=处理顺序）
        self._worker = None
        self._stop = False
        self._current = None
        self._lock = threading.Lock()

        self._build_ui()
        ico = _icon_path()
        if ico:
            try:
                self.SetIcon(wx.Icon(ico, wx.BITMAP_TYPE_ICO))
            except Exception:
                pass
        self._set_drop_target()

        self.Centre()
        self.Show()

    # ---------- UI（参照 web 页面布局） ----------
    def _build_ui(self):
        self.panel = wx.Panel(self)
        root = wx.BoxSizer(wx.VERTICAL)

        # -- 标题区（主标题 + 当前模型标签 + 版本 + 自动关机） --
        title = wx.BoxSizer(wx.HORIZONTAL)
        title.Add(wx.StaticText(self.panel, label='伴奏分离·和声版'),
                  0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        badge_inst = wx.StaticText(self.panel, label='Inst_HQ_3',
                                    style=wx.ALIGN_CENTRE_HORIZONTAL)
        badge_inst.SetForegroundColour(wx.Colour(255, 255, 255))
        badge_inst.SetBackgroundColour(wx.Colour(91, 95, 199))
        badge_inst.SetMinSize((86, 24))
        title.Add(badge_inst, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        badge_kara = wx.StaticText(self.panel, label='KARA_2',
                                   style=wx.ALIGN_CENTRE_HORIZONTAL)
        badge_kara.SetForegroundColour(wx.Colour(255, 255, 255))
        badge_kara.SetBackgroundColour(wx.Colour(242, 101, 34))
        badge_kara.SetMinSize((86, 24))
        title.Add(badge_kara, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        ver = wx.StaticText(self.panel, label=APP_VERSION, style=wx.ALIGN_CENTRE_HORIZONTAL)
        ver.SetForegroundColour(wx.Colour(255, 255, 255))
        ver.SetBackgroundColour(wx.Colour(92, 138, 63))
        ver.SetMinSize((58, 24))
        title.Add(ver, 0, wx.ALIGN_CENTER_VERTICAL)
        title.AddStretchSpacer(1)
        self.chk_shutdown = wx.CheckBox(self.panel, label='完成后自动关机')
        title.Add(self.chk_shutdown, 0, wx.ALIGN_CENTER_VERTICAL)
        root.Add(title, 0, wx.ALL | wx.EXPAND, 10)

        # -- 副标题 --
        sub = wx.StaticText(self.panel,
                            label='基于 UVR-MDX-NET 分离模型，本地 CPU 处理。可切换 Inst_HQ_3（干净伴奏）/ KARA_2（保留和声），一次拖入多个音视频自动排队处理。',
                            style=wx.ST_NO_AUTORESIZE)
        sub.Wrap(760)
        sub.SetForegroundColour(wx.Colour(74, 79, 87))
        root.Add(sub, 0, wx.LEFT | wx.RIGHT, 10)

        # -- 顶部操作区（对应网页里的“选择文件 / 选择文件夹”）--
        top = wx.BoxSizer(wx.HORIZONTAL)
        btn_add_files = wx.Button(self.panel, label='选择文件…')
        btn_add_dir = wx.Button(self.panel, label='选择文件夹…')
        self.btn_start = wx.Button(self.panel, label='开始处理')
        self.btn_clear = wx.Button(self.panel, label='清空列表')
        self.btn_stop = wx.Button(self.panel, label='停止')
        self.lbl_count = wx.StaticText(self.panel, label=' 0 项', style=wx.ST_NO_AUTORESIZE)
        for b in (btn_add_files, btn_add_dir, self.btn_start, self.btn_clear, self.btn_stop):
            top.Add(b, 0, wx.RIGHT, 8)
        top.Add(self.lbl_count, 0, wx.ALIGN_CENTER_VERTICAL)
        root.Add(top, 0, wx.ALL & ~wx.BOTTOM, 10)

        # -- 拖放区（点击可打开文件，也可拖入）--
        drop_pnl = wx.Panel(self.panel, style=wx.BORDER_SUNKEN)
        drop_pnl.SetBackgroundColour(wx.Colour(255, 255, 255))
        drop_sz = wx.BoxSizer(wx.VERTICAL)
        d1 = wx.StaticText(drop_pnl, label='拖入一个或多个音频 / 视频文件，或点击选择',
                           style=wx.ALIGN_CENTRE_HORIZONTAL)
        d1.SetForegroundColour(wx.Colour(21, 23, 26))
        d2 = wx.StaticText(drop_pnl,
                           label='音频：WAV / MP3 / FLAC / M4A 等；视频：MP4 / MKV / AVI / MOV / FLV / WEBM 等（自带 ffmpeg，无需另外安装）',
                           style=wx.ALIGN_CENTRE_HORIZONTAL | wx.ST_NO_AUTORESIZE)
        d2.SetForegroundColour(wx.Colour(106, 112, 120))
        drop_sz.Add(d1, 0, wx.EXPAND | wx.ALL, 8)
        drop_sz.Add(d2, 0, wx.EXPAND | wx.BOTTOM | wx.LEFT | wx.RIGHT, 8)
        drop_pnl.SetSizer(drop_sz)
        for w in (drop_pnl, d1, d2):
            w.Bind(wx.EVT_LEFT_UP, lambda e: self._on_add_files(None))
        self.drop_pnl = drop_pnl
        root.Add(drop_pnl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        # -- 分轨模型选择 --
        modelbox = wx.BoxSizer(wx.HORIZONTAL)
        modelbox.Add(wx.StaticText(self.panel, label='分轨模型：'),
                     0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.radio_clean = wx.RadioButton(self.panel, label='干净伴奏（Inst_HQ_3）', style=wx.RB_GROUP)
        self.radio_kara = wx.RadioButton(self.panel, label='保留和声（KARA_2）')
        modelbox.Add(self.radio_clean, 0, wx.RIGHT, 16)
        modelbox.Add(self.radio_kara, 0)
        self.radio_clean.SetValue(True)
        root.Add(modelbox, 0, wx.LEFT | wx.TOP, 10)

        # -- 视频音轨模式 --
        modebox = wx.BoxSizer(wx.HORIZONTAL)
        modebox.Add(wx.StaticText(self.panel, label='视频音轨模式：'),
                    0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.radio_add = wx.RadioButton(self.panel, label='添加新音轨（保留原唱，另加伴奏音轨）', style=wx.RB_GROUP)
        self.radio_replace = wx.RadioButton(self.panel, label='替换原音轨（仅保留伴奏）')
        modebox.Add(self.radio_add, 0, wx.RIGHT, 16)
        modebox.Add(self.radio_replace, 0)
        self.radio_add.SetValue(True)
        root.Add(modebox, 0, wx.LEFT | wx.TOP, 10)

        # 自动关机复选框已移到标题区右侧（版本号之后），不再单独占一行

        # -- 任务队列 --
        qbar = wx.BoxSizer(wx.HORIZONTAL)
        qlbl = wx.StaticText(self.panel, label='任务队列', style=wx.ST_NO_AUTORESIZE)
        qlbl.SetFont(wx.Font(wx.NORMAL_FONT.GetPointSize(), wx.DEFAULT, wx.NORMAL, wx.BOLD))
        qbar.Add(qlbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.lbl_qnote = wx.StaticText(self.panel, label='（自顶向下=处理顺序）',
                                       style=wx.ST_NO_AUTORESIZE)
        self.lbl_qnote.SetForegroundColour(wx.Colour(106, 112, 120))
        qbar.Add(self.lbl_qnote, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 4)
        root.Add(qbar, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        # 任务列表（含进度条列的表格）
        self.grid = wx.ListCtrl(self.panel, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        self.grid.InsertColumn(0, '#', width=44)
        self.grid.InsertColumn(1, '文件名', width=320)
        self.grid.InsertColumn(2, '类型', width=70)
        self.grid.InsertColumn(3, '状态', width=200)
        self.grid.InsertColumn(4, '进度', width=120)
        self.grid.InsertColumn(5, '源目录', width=240)
        self.grid.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_select)
        self.grid.Bind(wx.EVT_LIST_ITEM_DESELECTED, self._on_select)
        root.Add(self.grid, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 6)

        # -- 底部预览操作区（针对选中任务） --
        bottom = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_preview_voc = wx.Button(self.panel, label='预览人声')
        self.btn_preview_inst = wx.Button(self.panel, label='预览伴奏')
        self.btn_preview_origfmt = wx.Button(self.panel, label='预览伴奏(原格式)')
        self.btn_preview_video = wx.Button(self.panel, label='预览伴奏视频')
        self.btn_open_folder = wx.Button(self.panel, label='打开所在文件夹')
        self.lbl_note = wx.StaticText(self.panel, label=' 提示：拖入文件后点“开始处理“')
        for b in (self.btn_preview_voc, self.btn_preview_inst, self.btn_preview_origfmt,
                  self.btn_preview_video, self.btn_open_folder):
            b.Disable()
            bottom.Add(b, 0, wx.RIGHT, 8)
        bottom.Add(self.lbl_note, 1, wx.ALIGN_CENTER_VERTICAL)
        root.Add(bottom, 0, wx.ALL, 10)

        # -- 底部右侧 GitHub 开源链接 --
        footer = wx.BoxSizer(wx.HORIZONTAL)
        footer.AddStretchSpacer(1)
        self.lbl_gh_link = wx.StaticText(self.panel, label='GitHub 开源项目（52pojie 出品）')
        self.lbl_gh_link.SetForegroundColour(wx.Colour(51, 122, 183))
        self.lbl_gh_link.SetCursor(wx.Cursor(wx.CURSOR_HAND))
        footer.Add(self.lbl_gh_link, 0, wx.ALIGN_CENTER_VERTICAL)
        self.lbl_gh_link.Bind(wx.EVT_LEFT_DOWN, self._on_open_github)
        root.Add(footer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # -- 状态栏 --（参照网页 tip，放在底部）
        self.status = self.CreateStatusBar()
        self.status.SetStatusText('就绪 — 所有文件仅在本地处理，不会上传任何服务器')

        self.panel.SetSizer(root)
        # 关键：把 panel 挂到 frame 并撑满，否则内容不会显示
        frame_sz = wx.BoxSizer(wx.VERTICAL)
        frame_sz.Add(self.panel, 1, wx.EXPAND)
        self.SetSizer(frame_sz)

        # 事件
        btn_add_files.Bind(wx.EVT_BUTTON, self._on_add_files)
        btn_add_dir.Bind(wx.EVT_BUTTON, self._on_add_dir)
        self.btn_start.Bind(wx.EVT_BUTTON, self._on_start)
        self.btn_clear.Bind(wx.EVT_BUTTON, self._on_clear)
        self.btn_stop.Bind(wx.EVT_BUTTON, self._on_stop)
        self.btn_preview_voc.Bind(wx.EVT_BUTTON, lambda e: self._preview('voc'))
        self.btn_preview_inst.Bind(wx.EVT_BUTTON, lambda e: self._preview('inst'))
        self.btn_preview_origfmt.Bind(wx.EVT_BUTTON, lambda e: self._preview('origfmt'))
        self.btn_preview_video.Bind(wx.EVT_BUTTON, lambda e: self._preview('video'))
        self.btn_open_folder.Bind(wx.EVT_BUTTON, lambda e: self._preview('folder'))

    def _set_drop_target(self):
        self.SetDropTarget(_MediaDropTarget(self))

    # ---------- 事件处理 ----------
    def _on_drop(self, filenames):
        files = [f for f in filenames if _is_media(f)]
        files = sorted(files, key=str.lower)   # 与选文件夹一致：按文件名升序
        if files:
            self._append_tasks(files)
        return True

    def _on_add_files(self, e):
        dlg = wx.FileDialog(self, '选择音视频文件', style=wx.FD_MULTIPLE)
        if dlg.ShowModal() == wx.ID_OK:
            files = list(dlg.GetPaths())
            files = [f for f in files if _is_media(f)]
            files = sorted(files, key=str.lower)
            if files:
                self._append_tasks(files)
        dlg.Destroy()

    def _on_add_dir(self, e):
        dlg = wx.DirDialog(self, '选择文件夹', style=wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            folder = dlg.GetPath()
            files = []
            for n in sorted(os.listdir(folder), key=str.lower):
                fp = os.path.join(folder, n)
                if os.path.isfile(fp) and _is_media(fp):
                    files.append(fp)
            if files:
                self._append_tasks(files)
            else:
                wx.MessageBox('该文件夹中没有可处理的音视频文件', '提示',
                              wx.OK | wx.ICON_INFORMATION)
        dlg.Destroy()

    def _append_tasks(self, files):
        with self._lock:
            for f in files:
                self.tasks.append(Task(f))
        self._refresh_grid()

    def _selected_model(self):
        return 'UVR_MDXNET_KARA_2' if self.radio_kara.GetValue() else 'UVR-MDX-NET-Inst_HQ_3'

    def _on_open_github(self, e):
        webbrowser.open(GITHUB_URL)


    def _on_start(self, e):
        with self._lock:
            pending = [t for t in self.tasks if t.status in ('queued', 'error')]
        if not pending:
            self.lbl_note.SetLabel(' 没有待处理的任务；请先选择文件/文件夹或拖入文件')
            return
        self.btn_start.Disable()
        self.btn_clear.Disable()
        self._stop = False
        self.lbl_note.SetLabel(' 开始处理……')
        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()

    def _on_stop(self, e):
        self._stop = True
        self._clear_queue_not_started()
        self.lbl_note.SetLabel(' 正在停止……（当前任务完成后结束）')

    def _on_clear(self, e):
        with self._lock:
            if any(t.status in ('queued', 'processing') for t in self.tasks):
                wx.MessageBox('有任务在处理中，请先停止后再清空', '提示',
                              wx.OK | wx.ICON_INFORMATION)
                return
            self.tasks = []
        self._refresh_grid()
        self._update_preview_buttons()

    # ---------- 工作线程 ----------
    def _work_loop(self):
        while not self._stop:
            with self._lock:
                task = next((t for t in self.tasks if t.status == 'queued'), None)
            if task is None:
                break
            task.status = 'processing'
            self._current = task
            wx.CallAfter(self._refresh_grid)
            video_mode = 'replace' if self.radio_replace.GetValue() else 'add_track'
            _process(task, video_mode, self._make_reporter(task), self._selected_model())
            task.status = 'error' if task.error else 'done'
            if self._stop and task.status == 'done':
                pass
            self._current = None
            wx.CallAfter(self._refresh_grid)
            wx.CallAfter(self._on_task_done, task)
        wx.CallAfter(self._on_all_done)

    def _make_reporter(self, task):
        last = [0.0]
        started = [time.time()]
        def _rep(phase, pct):
            now = time.time()
            if now - last[0] >= 0.15:
                last[0] = now
                task.phase = phase
                task.pct = pct
                task.elapsed = int(now - started[0])
                wx.CallAfter(self._refresh_grid)
                # 状态栏与提示区同步当前操作，充分展示进展
                wx.CallAfter(self.status.SetStatusText,
                             '正在处理：%s　%s（%d%%）　已用 %d 秒' % (task.name, phase, pct, task.elapsed))
        return _rep

    def _clear_queue_not_started(self):
        """停止时：把仍未开始的任务标记为取消。"""
        with self._lock:
            for t in self.tasks:
                if t.status == 'queued':
                    t.status = 'error'
                    t.error = '已取消'
        wx.CallAfter(self._refresh_grid)

    def _on_task_done(self, task):
        self.lbl_note.SetLabel(' 完成：%s' % task.name)
        if self._selected_task() is task:
            self._update_preview_buttons()

    def _on_all_done(self):
        self.btn_start.Enable()
        self.btn_clear.Enable()
        self.lbl_note.SetLabel(' 全部任务已结束。可以预览或再次选择新文件。')
        self.status.SetStatusText('完成')
        # 自动关机：仅当勾选且用户未主动点击“停止”时才触发
        if not self._stop and self.chk_shutdown.GetValue():
            self.status.SetStatusText('所有任务处理完成，正在等待确认关机……')
            dlg = _ShutdownConfirmDialog(self, seconds=60)
            dlg.Show()

    # ---------- 展示 ----------
    def _refresh_grid(self):
        sel_id = None
        cur = self._selected_task()
        if cur:
            sel_id = cur.id
        self.grid.DeleteAllItems()
        with self._lock:
            items = list(self.tasks)
        self.lbl_count.SetLabel(' %d 项' % len(items))
        self.lbl_qnote.SetLabel('（自顶向下=处理顺序）')
        for i, t in enumerate(items):
            idx = self.grid.InsertItem(self.grid.GetItemCount(), str(i + 1))
            self.grid.SetItem(idx, 1, t.name)
            self.grid.SetItem(idx, 2, '视频' if t.is_video else '音频')
            if t.status == 'queued':
                st = '排队中'
            elif t.status == 'processing':
                st = '%s %d%%（已用 %d 秒）' % (t.phase, t.pct, getattr(t, 'elapsed', 0))
            elif t.status == 'done':
                st = '完成'
            else:
                st = '失败：' + (t.error or '')
            self.grid.SetItem(idx, 3, st)
            if t.status == 'processing':
                prog = '%d%%' % t.pct
            elif t.status == 'done':
                prog = '100%'
            else:
                prog = '—'
            self.grid.SetItem(idx, 4, prog)
            self.grid.SetItem(idx, 5, t.src_dir)
            if sel_id is not None and t.id == sel_id:
                self.grid.Select(idx, True)
                self.grid.Focus(idx)
        # 每次刷新后同步：没有选中条目时预览按钮一律禁用，避免“点了没反应”的歧义
        self._update_preview_buttons()

    def _selected_task(self):
        sel = self.grid.GetFirstSelected()
        if sel < 0:
            return None
        with self._lock:
            return self.tasks[sel] if sel < len(self.tasks) else None

    def _on_select(self, e):
        self._update_preview_buttons()

    def _update_preview_buttons(self):
        t = self._selected_task()
        for b in (self.btn_preview_voc, self.btn_preview_inst, self.btn_preview_origfmt,
                  self.btn_preview_video, self.btn_open_folder):
            b.Disable()
        if not t or t.status != 'done':
            return
        if t.is_video:
            if t.video_out:
                self.btn_preview_video.Enable()
            if t.voc:
                self.btn_preview_voc.Enable()
            if t.inst:
                self.btn_preview_inst.Enable()
        else:
            if t.voc:
                self.btn_preview_voc.Enable()
            if t.inst:
                self.btn_preview_inst.Enable()
            if t.inst_orig:
                self.btn_preview_origfmt.Enable()
        self.btn_open_folder.Enable()

    def _preview(self, which):
        t = self._selected_task()
        if not t:
            return
        path = None
        if which == 'voc':
            path = t.voc
        elif which == 'inst':
            path = t.inst
        elif which == 'origfmt':
            path = t.inst_orig
        elif which == 'video':
            path = t.video_out
        elif which == 'folder':
            path = t.src_dir if os.path.isdir(t.src_dir) else WORKDIR
        if path and os.path.exists(path):
            _open_default(path)
        else:
            self.lbl_note.SetLabel(' 预览文件不存在：%s' % path)

    def _on_close(self, e):
        self._stop = True
        try:
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=2)
        except Exception:
            pass
        self.Destroy()


class App(wx.App):
    def OnInit(self):
        frame = MainFrame()
        self.SetTopWindow(frame)
        return True


def main():
    print('加载模型 %s ...' % MODEL_NAME)
    get_separator()
    print('模型就绪，启动原生窗口。')
    app = App()
    app.MainLoop()


if __name__ == '__main__':
    main()