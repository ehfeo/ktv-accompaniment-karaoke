# -*- coding: utf-8 -*-
"""一键打包脚本：用 runtime 内 python + pyinstaller 打包 onedir，输出到 dist_onedir。"""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
NAME = '伴奏分离窗口版'
ICON = 'app_icon.ico'
DIST = 'dist_onedir'
WORK = 'build_onedir'
DATA = [
    (os.path.join('models', 'UVR-MDX-NET-Inst_HQ_3.onnx'), 'models'),
    (os.path.join('models', 'UVR_MDXNET_KARA_2.onnx'), 'models'),
    (os.path.join('runtime', 'ffmpeg.exe'), 'runtime'),
    ('app_icon.ico', '.'),
]

cmd = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onedir', '--windowed',
       '--name', NAME, '--icon', ICON,
       '--distpath', DIST, '--workpath', WORK]
for src, dest in DATA:
    cmd += ['--add-data', '%s;%s' % (src, dest)]
cmd.append('main_window.py')

os.chdir(BASE)
print('开始打包 onedir ...')
rc = subprocess.call(cmd)
if rc == 0:
    print('完成，产物在 %s' % os.path.join(BASE, DIST))
else:
    print('打包失败，退出码 %d' % rc)
sys.exit(rc)