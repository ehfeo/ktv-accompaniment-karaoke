# 伴奏分离 · 和声版

基于 **UVR-MDX-NET** 的本地伴奏/人声分轨工具（**原生 Windows 窗口版**），支持**双模型一键切换**。**52pojie 出品**。

把歌曲拖进窗口，即可把音乐拆成 **伴奏** 与 **人声** 两条音轨；视频可**新增伴奏音轨**或**替换原音轨**。产物自动回写到**源文件所在目录**。

## 功能

- **双模型一键切换**：
  - `Inst_HQ_3`（干净伴奏）—— 乐器伴奏更纯净
  - `KARA_2`（保留和声）—— 伴奏中尽量保留背景和声
- **音频分轨**：wav / mp3 / flac / m4a / aac / ogg / opus / wma / aiff 等，CPU 推理、纯本地运行
- **视频伴奏制作**：提取音轨 → 分离 → 加伴奏/替换原音轨合成新视频，并生成纯伴奏预览视频
- **源目录回写**：产物默认生成在源文件同目录，文件名追加 `_伴奏` / `_人声` / `_加伴奏音轨`
- **任务队列**：可拖入或选择多个文件，自上而下依次排队处理，进度分阶段显示
- **预览**：选中已完成任务，可打开人声 / 伴奏 / 伴奏视频
- **处理完成自动关机**：可勾选，全部任务完成后弹出 60 秒倒计时确认框，可取消
- **原生对话框**：文件/文件夹选择使用系统对话框且自动置顶

## 使用

1. 从 **Releases** 下载 `伴奏分离窗口版-v1.3.1.zip`（含 exe 与运行库，免安装）
2. 解压后运行 `伴奏分离窗口版.exe`
3. 拖入或选择要处理的文件 → 选分轨模型 → 点 **开始处理**
4. 在提示的目录中查看产物

> 首次运行会加载模型（约几秒），属正常现象。

## 从源码运行

- 需自行准备 `runtime/`（Python 3.9 + wxPython + onnxruntime）与 `models/`（`UVR-MDX-NET-Inst_HQ_3.onnx`、`UVR_MDXNET_KARA_2.onnx`）
- 启动：`runtime\python.exe main_window.py`（或 `start.bat` / `启动.vbs`）

## 打包

仓库内提供 `打包.bat` 一键打包为 onedir 程序，产物在 `dist_onedir`：

```bat
打包.bat
```

或手动：

```bash
runtime\python.exe -m PyInstaller --noconfirm --onedir --windowed ^
  --name "伴奏分离窗口版" --icon app_icon.ico ^
  --add-data "models\UVR-MDX-NET-Inst_HQ_3.onnx;models" ^
  --add-data "models\UVR_MDXNET_KARA_2.onnx;models" ^
  --add-data "runtime\ffmpeg.exe;runtime" ^
  --add-data "app_icon.ico;." ^
  --distpath dist_onedir --workpath build_onedir main_window.py
```

## 仓库说明

本仓库只含**源码与文档**。模型（约 100MB）与打包好的 exe 请从 **Releases** 获取，不纳入 git 以保持仓库轻量。

## 相关项目

- [ehfeo/ktv-accompaniment-desktop](https://github.com/ehfeo/ktv-accompaniment-desktop)：本系列上一版（仅 Inst_HQ_3）
- [ehfeo/ktv-accompaniment-tool](https://github.com/ehfeo/ktv-accompaniment-tool)：浏览器 WebUI 版（服务端 + 网页界面）
