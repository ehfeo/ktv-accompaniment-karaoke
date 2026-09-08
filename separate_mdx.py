# -*- coding: utf-8 -*-
"""
UVR-MDX-NET-Inst_HQ_3 单模型伴奏分离推理
复刻 Ultimate Vocal Remover 的 MDX-Net demix 算法（纯 numpy + onnxruntime，无 torch）。
"""
import os
import numpy as np
import onnxruntime as ort


# ---------- MDX 模型配置（文件, n_fft, hop, dim_f, dim_t, compensate, overlap）----------
# dim_t = 2 ** dim_t_set，各 MDX-Net 模型一致为 256；n_fft = dim_f * 2。
# dim_f/n_fft 由 onnx 输入形状决定（self.session.get_inputs()[0].shape[2]）。
MODEL_CONFIGS = {
    'UVR-MDX-NET-Inst_HQ_3': dict(
        file='UVR-MDX-NET-Inst_HQ_3.onnx', n_fft=6144, hop=1024,
        dim_f=3072, dim_t=256, compensate=1.022, overlap=0.25),
    'UVR_MDXNET_KARA_2': dict(
        file='UVR_MDXNET_KARA_2.onnx', n_fft=4096, hop=1024,
        dim_f=2048, dim_t=256, compensate=1.0, overlap=0.25),
}
DEFAULT_MODEL = 'UVR-MDX-NET-Inst_HQ_3'
SAMPLE_RATE = 44100


def _hann_periodic(n):
    """复刻 torch.hann_window(periodic=True)"""
    if n % 2 == 0:
        return np.hanning(n + 1)[:-1].astype(np.float32)
    return np.hanning(n).astype(np.float32)


class STFT:
    """复刻 UVR lib_v5/tfc_tdf_v3.py 的 STFT（基于 torch.stft/istft 语义）。"""

    def __init__(self, n_fft, hop_length, dim_f):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim_f = dim_f
        self.window = _hann_periodic(n_fft)

    def forward(self, x):
        """x: (2, T)  float32 -> (1, 4, dim_f, n_frames)"""
        c, t = x.shape
        pad = self.n_fft // 2
        # torch.stft center=True, pad_mode='reflect'
        xp = np.pad(x, ((0, 0), (pad, pad)), mode='reflect')
        frames = _enframe(xp, self.n_fft, self.hop_length)     # (c, n, n_fft)
        frames = frames * self.window[None, None, :]
        spec = np.fft.rfft(frames, n=self.n_fft, axis=-1)      # (c, n, n_fft//2+1) complex
        # 拆成 (实部, 虚部) 两通道并换成 (频率, 时间) 轴: (c,2,freq,n)
        spec = np.transpose(spec, (0, 2, 1))                   # (c, freq, n)
        spec = np.stack([spec.real, spec.imag], axis=1)        # (c,2,freq,n)
        out = spec.reshape(1, c * 2, spec.shape[2], spec.shape[3])  # (1,4,freq,n)
        return out[..., :self.dim_f, :].astype(np.float32)

    def inverse(self, x):
        """x: (1, 4, dim_f, t) float32 -> (2, T)"""
        x = np.asarray(x, dtype=np.float64)
        pad = self.n_fft // 2
        n = self.n_fft // 2 + 1
        _, c, f, t = x.shape
        # 补回被截断的最高频
        f_pad = np.zeros((1, c, n - f, t), dtype=x.dtype)
        x = np.concatenate([x, f_pad], axis=-2)                # (1,4,n,t)
        x = x.reshape(1, c // 2, 2, n, t)                      # (1,2ch,2,f,t)
        comp = x[..., 0, :, :] + 1j * x[..., 1, :, :]          # (1,2,f,t)
        tframes = np.fft.irfft(comp, n=self.n_fft, axis=-2)    # (1,2,n_fft,t)
        tframes = np.transpose(tframes, (0, 1, 3, 2))          # (1,2,t,n_fft)
        tframes *= self.window[None, None, None, :]
        # overlap-add
        out_len = (t - 1) * self.hop_length + self.n_fft
        y = np.zeros((1, 2, out_len), dtype=np.float64)
        wsq = np.zeros((1, 2, out_len), dtype=np.float64)
        for i in range(t):
            s = i * self.hop_length
            y[:, :, s:s + self.n_fft] += tframes[:, :, i, :]
            wsq[:, :, s:s + self.n_fft] += self.window * self.window
        y = y / np.maximum(wsq, 1e-12)
        # torch.istft center=True, length=None: 返回 (n_frames-1)*hop，
        # 即去掉前后各 n_fft//2 的 padding
        return y[:, :, pad:pad + (t - 1) * self.hop_length][0].astype(np.float32)  # (2, T)


def _enframe(x, frame_len, hop):
    """按 hop 滑窗切帧: (c, L) -> (c, n, frame_len)"""
    c, l = x.shape
    n = 1 + (l - frame_len) // hop
    strides = (x.strides[0], x.strides[1] * hop, x.strides[1])
    return np.lib.stride_tricks.as_strided(x, shape=(c, n, frame_len), strides=strides)


class MDXSeparator:
    def __init__(self, model_path, cfg=None):
        cfg = cfg or MODEL_CONFIGS[DEFAULT_MODEL]
        self.session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        self.n_fft = cfg['n_fft']
        self.hop = cfg['hop']
        self.dim_f = cfg['dim_f']
        self.dim_t = cfg['dim_t']
        self.compensate = cfg['compensate']
        self.overlap = cfg['overlap']
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop * (self.dim_t - 1)
        self.gen_size = self.chunk_size - 2 * self.trim
        self._clear_bands = cfg.get('clear_bands', 3)
        self.stft = STFT(self.n_fft, self.hop, self.dim_f)
        # 确认输入/输出名
        self.input_name = self.session.get_inputs()[0].name

    def _model_run(self, spek):
        """spek: (1,4,dim_f,t) -> 同形状"""
        return self.session.run(None, {self.input_name: spek})[0]

    def run_model(self, mix_part):
        """mix_part: (2, chunk_size) -> (2, chunk_size) 模型输出波形"""
        spek = self.stft.forward(mix_part)
        spek[:, :, :self._clear_bands, :] *= 0  # 复刻 UVR: 清零最低频点
        # 经典 MDX-Net 反演(is_denoise): spec_pred = -model(-spek)*0.5 + model(spek)*0.5
        spec_pred = -self._model_run(-spek) * 0.5 + self._model_run(spek) * 0.5
        return self.stft.inverse(spec_pred)

    def demix(self, mix, progress=None):
        """
        mix: (2, T) float32; progress(i, total) 可选进度回调
        return: (2, T) 伴奏波形（含 compensate）
        """
        trim = self.trim
        chunk_size = self.chunk_size
        gen_size = self.gen_size
        overlap = self.overlap

        pad = gen_size + trim - (mix.shape[-1] % gen_size)
        mixture = np.concatenate(
            (np.zeros((2, trim), dtype='float32'),
             mix,
             np.zeros((2, pad), dtype='float32')), 1)

        step = int((1 - overlap) * chunk_size)
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        total_chunks = (mixture.shape[-1] + step - 1) // step
        chunk_index = 0

        for i in range(0, mixture.shape[-1], step):
            chunk_index += 1
            if progress:
                progress(chunk_index, total_chunks)
            start = i
            end = min(i + chunk_size, mixture.shape[-1])
            chunk_size_actual = end - start

            window = np.hanning(chunk_size_actual)
            window = np.tile(window[None, None, :], (1, 2, 1))

            mix_part_ = mixture[:, start:end]
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype='float32')), axis=-1)

            tar_waves = self.run_model(mix_part_)[None]   # (1,2,chunk)

            tar_waves[..., :chunk_size_actual] *= window
            divider[..., start:end] += window
            result[..., start:end] += tar_waves[..., :end - start]

        with np.errstate(invalid='ignore', divide='ignore'):
            tar_waves = result / np.maximum(divider, 1e-8)
        tar_waves = np.nan_to_num(tar_waves, nan=0.0, posinf=0.0, neginf=0.0)
        tar_waves = tar_waves[:, :, trim:-trim][:, :, :mix.shape[-1]]

        return (tar_waves[0] * self.compensate).astype(np.float32)


def separate_into_instrumental_and_vocals(mix):
    """
    顶层接口：输入全音频 (2, T)，返回 (伴奏, 人声)，均为 (2, T)。
    """
    raise NotImplementedError("该模块不直接加载模型，请通过 MDXSeparator 使用。")


if __name__ == '__main__':
    # 简单自检：验证 STFT 往返重建（用 hop 倍数长度，避开边缘）
    _c = MODEL_CONFIGS[DEFAULT_MODEL]
    rng = np.random.RandomState(0)
    st = STFT(_c['n_fft'], _c['hop'], _c['dim_f'])
    L = 128 * HOP                                 # 131072，hop 对齐
    sig = (rng.randn(2, L) * 0.1).astype(np.float32)
    spec = st.forward(sig)
    back = st.inverse(spec)
    print("length match:", back.shape[-1] == L)
    seg = slice(N_FFT, -N_FFT)
    err = np.max(np.abs(back[:, seg] - sig[:, seg]))
    print("STFT roundtrip max-abs-err (center):", err)
    # 复刻 demix：截断到 dim_f 后再重建
    spec_cut = spec[..., :_c['dim_f'], :]
    back2 = st.inverse(spec_cut)
    err2 = np.max(np.abs(back2[:, seg] - sig[:, seg]))
    print("STFT roundtrip max-abs-err with cutoff:", err2)