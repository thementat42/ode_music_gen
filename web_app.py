from __future__ import annotations
import io
import os
import contextlib
import numpy as np
import streamlit as st
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display


st.set_page_config(page_title="ODE Music Analyzer", layout="wide")
st.title("Audio Harmonic Analyzer and Resynthesizer")
st.write("Upload an audio file, view its spectrum A(f), detect peaks, and resynthesize for A/B comparison.")


def load_audio(file_bytes: bytes, target_sr: int = 44100):
    # Try soundfile first (fast path), fallback to librosa
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return data.astype(np.float32), sr
    except Exception:
        # Suppress mpg123/audioread ID3 warnings on stderr
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
            y, sr = librosa.load(io.BytesIO(file_bytes), sr=target_sr, mono=True)
        return y.astype(np.float32), sr


def compute_spectrum(y: np.ndarray, sr: int):
    # Window and FFT
    n = len(y)
    # Use power-of-two FFT size >= n up to a limit
    nfft = int(2 ** np.ceil(np.log2(max(1024, n))))
    win = np.hanning(min(n, nfft))
    x = np.zeros(nfft, dtype=np.float32)
    x[:len(win)] = y[:len(win)] * win
    Y = np.fft.rfft(x)
    mag = np.abs(Y)
    freq = np.fft.rfftfreq(nfft, d=1.0/sr)
    phase = np.angle(Y)
    return freq, mag, phase, nfft


def pick_peaks(freq: np.ndarray, mag: np.ndarray, phase: np.ndarray,
               min_prominence: float, max_peaks: int, fmin: float, fmax: float):
    # Simple local-max peak picking with prominence
    mask = (freq >= fmin) & (freq <= fmax)
    idxs = np.flatnonzero(mask)
    f = freq[mask]
    m = mag[mask]
    if len(m) < 3:
        return np.array([]), np.array([]), np.array([])
    # Normalize magnitude for stable thresholding
    m_norm = m / (np.max(m) + 1e-9)
    peaks = []
    for i in range(1, len(m_norm) - 1):
        if m_norm[i] > m_norm[i-1] and m_norm[i] > m_norm[i+1] and m_norm[i] >= min_prominence:
            # Use absolute FFT bin index via masked indices
            abs_idx = int(idxs[i])
            peaks.append((float(f[i]), float(m_norm[i]), abs_idx))
    # Sort by magnitude desc and keep top K
    peaks.sort(key=lambda t: t[1], reverse=True)
    peaks = peaks[:max_peaks]
    if not peaks:
        return np.array([]), np.array([]), np.array([])
    pf = np.array([p[0] for p in peaks], dtype=np.float32)
    pa = np.array([p[1] for p in peaks], dtype=np.float32)
    pph = np.array([phase[p[2]] for p in peaks], dtype=np.float32)
    return pf, pa, pph


def pick_peaks_nms(
    freq: np.ndarray,
    mag: np.ndarray,
    phase: np.ndarray,
    max_peaks: int,
    fmin: float,
    fmax: float,
    min_distance_hz: float = 20.0,
    min_prominence_norm: float = 0.0,
):
    """Greedy non-maximum suppression (NMS) peak picking.

    Steps:
    - Find local maxima within [fmin, fmax].
    - Optionally filter by normalized prominence threshold.
    - Sort candidates by magnitude descending.
    - Iteratively accept a candidate only if it is not within min_distance_hz of any already selected peak.
    - Stop when max_peaks selected or no candidates remain.
    """
    mask = (freq >= fmin) & (freq <= fmax)
    if not np.any(mask) or max_peaks <= 0:
        return np.array([]), np.array([]), np.array([])
    idxs = np.flatnonzero(mask)
    f = freq[mask]
    m = mag[mask]
    if len(m) < 3:
        return np.array([]), np.array([]), np.array([])
    m_norm = m / (np.max(m) + 1e-9)

    # Collect local maxima candidates (relative indices)
    cand_rel = []
    for i in range(1, len(m) - 1):
        if m[i] > m[i - 1] and m[i] > m[i + 1]:
            if m_norm[i] >= float(min_prominence_norm):
                cand_rel.append(i)
    # Fallback to top bins if no local maxima pass the threshold
    if not cand_rel:
        order = np.argsort(m)[::-1]
        cand_rel = order.tolist()

    # Build candidate list with absolute indices and values
    cands = []
    for i in cand_rel:
        abs_idx = int(idxs[i])
        cands.append((float(freq[abs_idx]), float(mag[abs_idx]), float(phase[abs_idx]), abs_idx))

    # Sort by amplitude descending
    cands.sort(key=lambda t: t[1], reverse=True)

    sel_f, sel_a, sel_p = [], [], []
    for cf, ca, cp, _ in cands:
        if len(sel_f) >= int(max_peaks):
            break
        # Suppress if within min_distance_hz of any stronger selected peak
        if any(abs(cf - sf) < float(min_distance_hz) for sf in sel_f):
            continue
        sel_f.append(cf)
        sel_a.append(ca)
        sel_p.append(cp)

    if not sel_f:
        return np.array([]), np.array([]), np.array([])
    return np.array(sel_f, dtype=np.float32), np.array(sel_a, dtype=np.float32), np.array(sel_p, dtype=np.float32)


def resynthesize_from_peaks(
    duration: float,
    sr: int,
    peak_freqs: np.ndarray,
    peak_amps: np.ndarray,
    peak_phases: np.ndarray | None = None,
    apply_scaling: bool = True,
    scale_multiplier: float = 1.0,
):
    """Resynthesize a signal from peak frequencies, amplitudes, and phases.

    Includes the requested normalization factor 1/(Δt·n) = sr/num_peaks, where:
    - Δt = 1/sr (sample period)
    - n = number of included frequency components (len(peak_freqs))
    A safety normalization is applied only if the signal would clip.
    """
    if len(peak_freqs) == 0:
        return np.zeros(int(duration * sr), dtype=np.float32)
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    # Normalize peak amplitudes to a stable range before applying scaling
    a = peak_amps / (np.max(peak_amps) + 1e-9)
    y = np.zeros_like(t)
    if peak_phases is None or len(peak_phases) != len(peak_freqs):
        peak_phases = np.zeros_like(peak_freqs)
    for f, amp, ph in zip(peak_freqs, a, peak_phases):
        y += amp * np.sin(2 * np.pi * f * t + ph)
    # Optional 1/(Δt·n) scaling: Δt = 1/sr => sr/n
    if apply_scaling:
        n_comp = max(1, int(len(peak_freqs)))
        y *= (float(sr) / float(n_comp))
    # Optional user multiplier to compare differences
    y *= float(scale_multiplier)
    # Safety normalization: only if clipping would occur
    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


uploaded = st.file_uploader("Upload audio (wav, mp3, flac)", type=["wav", "mp3", "flac", "ogg", "m4a"])

col1, col2 = st.columns(2)
with col1:
    sr = st.number_input("Sample rate", min_value=8000, max_value=96000, value=44100, step=1000)
    max_peaks = st.slider("Max peaks", 1, 50, 5)
    min_prom = st.slider("Min prominence (normalized)", 0.0, 1.0, 0.0, 0.01)
with col2:
    fmin = st.number_input("Min freq (Hz)", min_value=0.0, value=20.0, step=1.0)
    fmax_default = float(min(int(sr)//2, 10000))
    fmax = st.number_input("Max freq (Hz)", min_value=100.0, value=fmax_default, step=10.0)
    resyn_dur = st.number_input("Resynthesis duration (s)", min_value=0.1, value=2.0, step=0.1)
    scale_mult = st.number_input("Scaling multiplier", min_value=0.01, max_value=10.0, value=1.0, step=0.01)

if uploaded is not None:
    file_bytes = uploaded.read()
    y, sr = load_audio(file_bytes, target_sr=int(sr))
    st.audio(y, sample_rate=int(sr), format="audio/wav")

    # Spectrum
    freq, mag, phase, nfft = compute_spectrum(y, int(sr))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freq, mag, lw=0.8)
    ax.set_xlim(0.0, float(min(fmax, float(freq[-1]))))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (|A(f)|)")
    ax.set_title("Magnitude Spectrum")
    st.pyplot(fig, clear_figure=True)

    # Peaks with non-maximum suppression (20 Hz gap)
    pf, pa, pph = pick_peaks_nms(
        freq, mag, phase,
        max_peaks=int(max_peaks),
        fmin=float(fmin),
        fmax=float(fmax),
        min_distance_hz=100.0,
        min_prominence_norm=float(min_prom),
    )
    if len(pf) > 0:
        st.write(f"Detected {len(pf)} peaks (min gap 20 Hz)")
        # Overlay peaks
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(freq, mag, lw=0.6, alpha=0.6)
        ax2.scatter(pf, pa, color='r', s=15, label='peaks')
        ax2.set_xlim(0.0, float(min(fmax, float(freq[-1]))))
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude (|A(f)|)")
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)

        # Show selected peaks table
        st.write("Selected peaks:")
        st.dataframe({
            "Frequency (Hz)": np.round(pf.astype(float), 2),
            "Amplitude": np.round(pa.astype(float), 6),
        }, use_container_width=True)

        # Resynthesize
        y_resyn = resynthesize_from_peaks(
            float(resyn_dur),
            int(sr),
            pf,
            pa,
            pph,
            apply_scaling=True,
            scale_multiplier=float(scale_mult),
        )
        st.subheader("Original vs Resynthesized")
        st.write("Original")
        st.audio(y, sample_rate=int(sr), format="audio/wav")
        st.write("Resynthesized (sinusoidal, peak amplitudes normalized)")
        st.audio(y_resyn, sample_rate=int(sr), format="audio/wav")
    else:
        st.info("No peaks detected with current settings; try lowering min prominence or widening the frequency range.")
