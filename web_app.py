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

# tabs for different features: first tab is single-file analysis, second tab is two-file comparison
tab1, tab2, tab3 = st.tabs(["Single Audio Analysis", "Two Recording Comparison", "Playground"])

def load_audio(file_bytes: bytes, target_sr: int = 44100):
    # attempt decoding with soundfile first for efficiency; fall back to librosa on failure
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return data.astype(np.float32), sr
    except Exception:
        # suppress mpg123 and id3 related stderr warnings during fallback load
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
            y, sr = librosa.load(io.BytesIO(file_bytes), sr=target_sr, mono=True)
        return y.astype(np.float32), sr


def compute_spectrum(y: np.ndarray, sr: int):
    # create analysis window and perform fft
    n = len(y)
    # choose the next power of two greater than or equal to n (minimum 1024) for fft efficiency
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
    # perform simple local maximum peak selection with a prominence threshold
    mask = (freq >= fmin) & (freq <= fmax)
    idxs = np.flatnonzero(mask)
    f = freq[mask]
    m = mag[mask]
    if len(m) < 3:
        return np.array([]), np.array([]), np.array([])
    # normalize magnitudes to stabilize thresholding
    m_norm = m / (np.max(m) + 1e-9)
    peaks = []
    for i in range(1, len(m_norm) - 1):
        if m_norm[i] > m_norm[i-1] and m_norm[i] > m_norm[i+1] and m_norm[i] >= min_prominence:
            # record absolute fft bin index for phase retrieval
            abs_idx = int(idxs[i])
            peaks.append((float(f[i]), float(m_norm[i]), abs_idx))
    # sort peaks by descending magnitude and retain the top k
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
    """non-maximum suppression peak picking.

    steps:
    - identify local maxima within the specified frequency range.
    - filter by normalized prominence threshold if provided.
    - sort candidates by magnitude in descending order.
    - iteratively retain a candidate only if it is not within min_distance_hz of any previously retained peak.
    - stop when max_peaks is reached or no candidates remain.
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

    # collect indices of local maxima that pass prominence threshold
    cand_rel = []
    for i in range(1, len(m) - 1):
        if m[i] > m[i - 1] and m[i] > m[i + 1]:
            if m_norm[i] >= float(min_prominence_norm):
                cand_rel.append(i)
    # fallback: if no local maxima pass, use strongest magnitude bins
    if not cand_rel:
        order = np.argsort(m)[::-1]
        cand_rel = order.tolist()

    # build candidate tuples (frequency, magnitude, phase, absolute index)
    cands = []
    for i in cand_rel:
        abs_idx = int(idxs[i])
        cands.append((float(freq[abs_idx]), float(mag[abs_idx]), float(phase[abs_idx]), abs_idx))

    # sort candidates by descending magnitude
    cands.sort(key=lambda t: t[1], reverse=True)

    sel_f, sel_a, sel_p = [], [], []
    for cf, ca, cp, _ in cands:
        if len(sel_f) >= int(max_peaks):
            break
        # skip candidate if it is within min_distance_hz of a retained peak
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
    peak_phases: np.ndarray = None,
    apply_scaling: bool = True,
    scale_multiplier: float = 1.0,
):
    """resynthesize a sinusoidal mixture from selected peak frequencies.

    incorporates optional scaling by sr / n (normalization factor) and applies peak normalization only if clipping would occur.
    if phases are absent or mismatched in length, zeros are used.
    """
    if len(peak_freqs) == 0:
        return np.zeros(int(duration * sr), dtype=np.float32)
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    # normalize amplitudes to avoid scaling instabilities
    a = peak_amps / (np.max(peak_amps) + 1e-9)
    y = np.zeros_like(t)
    if peak_phases is None or len(peak_phases) != len(peak_freqs):
        peak_phases = np.zeros_like(peak_freqs)
    for f, amp, ph in zip(peak_freqs, a, peak_phases):
        y += amp * np.sin(2 * np.pi * f * t + ph)
    # optionally scale by sr / n (requested normalization factor)
    if apply_scaling:
        n_comp = max(1, int(len(peak_freqs)))
        y *= (float(sr) / float(n_comp))
    # apply user-provided overall gain multiplier
    y *= float(scale_multiplier)
    # apply peak normalization only if clipping would otherwise occur
    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def find_harmonic_amplitudes(freq: np.ndarray, mag: np.ndarray, fundamental_freq: float, max_harmonics: int, tolerance_hz: float = 5.0):
    """locate each harmonic n * f0 within a +/- tolerance window and select the loudest bin.

    parameters:
      freq: fft frequency bins.
      mag: raw magnitudes (normalized internally).
      fundamental_freq: detected fundamental frequency f0.
      max_harmonics: number of harmonic multiples to evaluate.
      tolerance_hz: half-width of the search window around each target frequency.

    returns:
      harmonic_amps: normalized amplitudes per harmonic.
      harmonic_freqs: actual bin frequencies selected.
      target_freqs: ideal harmonic target frequencies n * f0.
    """
    # normalize magnitudes for consistent scaling
    mag_normalized = mag / (np.max(mag) + 1e-9)

    harmonic_amps: list[float] = []
    harmonic_freqs: list[float] = []
    target_freqs: list[float] = []

    for n in range(1, max_harmonics + 1):
        target_freq = n * fundamental_freq
        target_freqs.append(target_freq)

        # define search window bounds around the target frequency
        freq_min = target_freq - tolerance_hz
        freq_max = target_freq + tolerance_hz

        # determine which bins lie inside the window
        window_mask = (freq >= freq_min) & (freq <= freq_max)

        if np.any(window_mask):
            # select the loudest bin within the window
            window_indices = np.where(window_mask)[0]
            window_mags = mag_normalized[window_indices]
            window_freqs = freq[window_indices]

            max_idx_in_window = np.argmax(window_mags)
            actual_idx = window_indices[max_idx_in_window]

            harmonic_amps.append(float(mag_normalized[actual_idx]))
            harmonic_freqs.append(float(freq[actual_idx]))
        else:
            # if no bins match, assign zero amplitude and retain the target frequency
            harmonic_amps.append(0.0)
            harmonic_freqs.append(float(target_freq))  # keep target freq as placeholder

    return (
        np.array(harmonic_amps, dtype=np.float32),
        np.array(harmonic_freqs, dtype=np.float32),
        np.array(target_freqs, dtype=np.float32),
    )


def calculate_harmonic_mse(actual_harmonics: np.ndarray, theoretical_harmonics: np.ndarray):
    """compute mean squared error per harmonic and the overall average mse.

    arrays are truncated to a common length if they differ.
    """
    # ensure arrays share the same length by truncation if necessary
    min_len = min(len(actual_harmonics), len(theoretical_harmonics))
    actual = actual_harmonics[:min_len]
    theoretical = theoretical_harmonics[:min_len]
    
    # compute squared error per harmonic
    mse_per_harmonic = (actual - theoretical) ** 2
    mse_total = np.mean(mse_per_harmonic)
    
    return mse_total, mse_per_harmonic


def detect_fundamental_frequency(freq: np.ndarray, mag: np.ndarray, min_freq: float = 50.0, max_freq: float = 800.0):
    """estimate the fundamental frequency by selecting the strongest peak within a candidate band.

    suitable for steady tonal signals.
    """
    # restrict frequency range to plausible fundamental bounds
    mask = (freq >= min_freq) & (freq <= max_freq)
    if not np.any(mask):
        return 220.0  # Default fallback frequency
    
    freq_filtered = freq[mask]
    mag_filtered = mag[mask]
    
    # select frequency with maximum magnitude within filtered band
    max_idx = np.argmax(mag_filtered)
    fundamental_freq = freq_filtered[max_idx]
    
    return float(fundamental_freq)


with tab1:
    st.write("Upload an audio file, view its spectrum A(f), detect peaks, and resynthesize for A/B comparison.")  # single-file analysis workflow
    
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

        # plot spectrum for the uploaded audio
        freq, mag, phase, nfft = compute_spectrum(y, int(sr))
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(freq, mag, lw=0.8)
        ax.set_xlim(0.0, float(min(fmax, float(freq[-1]))))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (|A(f)|)")
        ax.set_title("Magnitude Spectrum")
        st.pyplot(fig, clear_figure=True)

        # detect peaks using nms approach to avoid closely spaced duplicates
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
            # overlay detected peaks on the magnitude spectrum
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.plot(freq, mag, lw=0.6, alpha=0.6)
            ax2.scatter(pf, pa, color='r', s=15, label='peaks')
            ax2.set_xlim(0.0, float(min(fmax, float(freq[-1]))))
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Amplitude (|A(f)|)")
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)

            # display table of detected peak frequencies and amplitudes
            st.write("Selected peaks:")
            st.dataframe({
                "Frequency (Hz)": np.round(pf.astype(float), 2),
                "Amplitude": np.round(pa.astype(float), 6),
            }, use_container_width=True)

            # resynthesize signal from detected sinusoidal components
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

with tab2:
    st.write("Compare two recordings: actual vs theoretical. Analyze harmonic frequencies and calculate MSE error.")  # two-file comparative analysis workflow
    
    # upload widgets for actual and theoretical audio recordings
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        st.subheader("Actual Recording")
        actual_file = st.file_uploader("Upload actual recording", type=["wav", "mp3", "flac", "ogg", "m4a"], key="actual")
    
    with col_upload2:
        st.subheader("Theoretical Recording") 
        theoretical_file = st.file_uploader("Upload theoretical recording", type=["wav", "mp3", "flac", "ogg", "m4a"], key="theoretical")
    
    # user-adjustable parameters for comparison
    col_param1, col_param2 = st.columns(2)
    
    with col_param1:
        comp_sr = st.number_input("Sample rate for comparison", min_value=8000, max_value=96000, value=44100, step=1000, key="comp_sr")
        min_fund_freq = st.number_input("Min fundamental freq (Hz)", min_value=20.0, value=50.0, step=1.0, key="min_fund")
        max_display_freq = st.number_input("Max display frequency (Hz)", min_value=500.0, value=2000.0, step=100.0, key="max_display_freq")
        
    with col_param2:
        max_fund_freq = st.number_input("Max fundamental freq (Hz)", min_value=100.0, value=800.0, step=1.0, key="max_fund")
        max_harmonics = st.slider("Max harmonics (n)", 1, 20, 10, key="max_harmonics")
        window_size = st.number_input("Search window size (Hz)", min_value=0.1, value=5.0, step=0.1, key="window_size")
    
    if actual_file is not None and theoretical_file is not None:
        # load and resample both recordings to a common sample rate
        actual_bytes = actual_file.read()
        theoretical_bytes = theoretical_file.read()
        
        y_actual, sr_actual = load_audio(actual_bytes, target_sr=int(comp_sr))
        y_theoretical, sr_theoretical = load_audio(theoretical_bytes, target_sr=int(comp_sr))
        
        # display audio players for both recordings
        col_audio1, col_audio2 = st.columns(2)
        with col_audio1:
            st.write("**Actual Recording:**")
            st.audio(y_actual, sample_rate=int(comp_sr), format="audio/wav")
            
        with col_audio2:
            st.write("**Theoretical Recording:**")
            st.audio(y_theoretical, sample_rate=int(comp_sr), format="audio/wav")
        
        # compute spectra for actual and theoretical recordings
        freq_actual, mag_actual, phase_actual, _ = compute_spectrum(y_actual, int(comp_sr))
        freq_theoretical, mag_theoretical, phase_theoretical, _ = compute_spectrum(y_theoretical, int(comp_sr))
        
        # estimate fundamental frequency from actual recording
        fundamental_freq = detect_fundamental_frequency(freq_actual, mag_actual, min_fund_freq, max_fund_freq)
        
        # present detected fundamental frequency to user
        st.info(f"**Detected Fundamental Frequency from Actual Recording: {fundamental_freq:.2f} Hz**")
        
        # extract harmonic amplitudes for actual recording
        harmonics_actual, harmonic_freqs_actual, target_freqs = find_harmonic_amplitudes(
            freq_actual, mag_actual, fundamental_freq, max_harmonics, window_size
        )
        
        harmonics_theoretical, harmonic_freqs_theoretical, _ = find_harmonic_amplitudes(
            freq_theoretical, mag_theoretical, fundamental_freq, max_harmonics, window_size
        )
        
        # normalize magnitudes for comparable plotting scale
        mag_actual_norm = mag_actual / (np.max(mag_actual) + 1e-9)
        mag_theoretical_norm = mag_theoretical / (np.max(mag_theoretical) + 1e-9)
        
        # plot normalized spectra including search windows and selected harmonic bins
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        ax1.plot(freq_actual, mag_actual_norm, lw=0.8, label='Actual', color='blue')
        # annotate detected fundamental frequency
        ax1.axvline(x=fundamental_freq, color='green', linestyle='--', alpha=0.7, label=f'Fundamental: {fundamental_freq:.1f} Hz')
        
        # highlight search window and selected bin for each harmonic (actual)
        for i, (target_f, actual_f) in enumerate(zip(target_freqs[:max_harmonics], harmonic_freqs_actual[:max_harmonics])):
            if actual_f <= min(max_display_freq, freq_actual[-1]):
                # Show search window as shaded region
                ax1.axvspan(target_f - window_size, target_f + window_size, alpha=0.1, color='orange')
                # Mark actual frequency found
                ax1.axvline(x=actual_f, color='red', linestyle=':', alpha=0.6, 
                           label=f'H{i+1}: {actual_f:.1f}Hz' if i < 5 else '')
        
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Normalized Amplitude")
        ax1.set_title("Actual Recording Spectrum (Normalized)")
        ax1.set_xlim(0, min(max_display_freq, freq_actual[-1]))
        ax1.legend()
        
        ax2.plot(freq_theoretical, mag_theoretical_norm, lw=0.8, label='Theoretical', color='red')
        
        # highlight search window and selected bin for each harmonic (theoretical)
        for i, (target_f, theoretical_f) in enumerate(zip(target_freqs[:max_harmonics], harmonic_freqs_theoretical[:max_harmonics])):
            if theoretical_f <= min(max_display_freq, freq_theoretical[-1]):
                # Show search window as shaded region
                ax2.axvspan(target_f - window_size, target_f + window_size, alpha=0.1, color='orange')
                # Mark actual frequency found
                ax2.axvline(x=theoretical_f, color='blue', linestyle=':', alpha=0.6,
                           label=f'H{i+1}: {theoretical_f:.1f}Hz' if i < 5 else '')
        
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Normalized Amplitude")
        ax2.set_title("Theoretical Recording Spectrum (Normalized)")
        ax2.set_xlim(0, min(max_display_freq, freq_theoretical[-1]))
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        
        # calculate mse metrics
        mse_total, mse_per_harmonic = calculate_harmonic_mse(harmonics_actual, harmonics_theoretical)
        
        # present harmonic analysis results
        st.subheader("Harmonic Analysis Results")
        
        # construct results table including target and selected frequencies
        results_df = {
            "Harmonic (n)": list(range(1, len(harmonics_actual) + 1)),
            "Argmax Target Freq (Hz)": np.round(target_freqs, 2),
            "Argmax Actual Freq (Hz)": np.round(harmonic_freqs_actual, 2),
            "Theoretical Freq (Hz)": np.round(harmonic_freqs_theoretical, 2),
            "Actual Amplitude": np.round(harmonics_actual, 6),
            "Theoretical Amplitude": np.round(harmonics_theoretical, 6),
            "MSE": np.round(mse_per_harmonic, 8)
        }
        
        st.dataframe(results_df, use_container_width=True)
        
        # display aggregate mse
        st.metric("Total MSE", f"{mse_total:.8f}")
        
        # plot bar chart comparing harmonic amplitudes
        fig_harmonics, ax_harm = plt.subplots(figsize=(10, 6))
        
        harmonic_numbers = range(1, len(harmonics_actual) + 1)
        width = 0.35
        x = np.arange(len(harmonic_numbers))
        
        bars1 = ax_harm.bar(x - width/2, harmonics_actual, width, label='Actual', alpha=0.8, color='blue')
        bars2 = ax_harm.bar(x + width/2, harmonics_theoretical, width, label='Theoretical', alpha=0.8, color='red')
        
        ax_harm.set_xlabel('Harmonic Number (n)')
        ax_harm.set_ylabel('Amplitude')
        ax_harm.set_title('Harmonic Amplitudes Comparison')
        ax_harm.set_xticks(x)
        ax_harm.set_xticklabels(harmonic_numbers)
        ax_harm.legend()
        ax_harm.grid(True, alpha=0.3)
        
        # annotate each bar with amplitude value
        for bar in bars1:
            height = bar.get_height()
            ax_harm.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
                           
        for bar in bars2:
            height = bar.get_height()
            ax_harm.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig_harmonics, clear_figure=True)
        
        # plot per-harmonic mse values
        fig_mse, ax_mse = plt.subplots(figsize=(10, 4))
        ax_mse.bar(harmonic_numbers, mse_per_harmonic, color='orange', alpha=0.7)
        ax_mse.set_xlabel('Harmonic Number (n)')
        ax_mse.set_ylabel('MSE')
        ax_mse.set_title('MSE per Harmonic')
        ax_mse.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_mse, clear_figure=True)
        
    else:
        st.info("Please upload both actual and theoretical recordings to perform comparison.")  # require both recordings to proceed with comparison

with tab3:
        st.write("Interactive piano playground using generated notes (G2–C6). Click keys or use keyboard to play/stop.")

        # define note names from G2 to C6 (inclusive) using equal-temperament sequence of semitones
        white_keys_order = [
                # octave 2
                "G2", "A2", "B2",
                # octave 3
                "C3", "D3", "E3", "F3", "G3", "A3", "B3",
                # octave 4
                "C4", "D4", "E4", "F4", "G4", "A4", "B4",
                # octave 5
                "C5", "D5", "E5", "F5", "G5", "A5", "B5",
                # octave 6 (up to C6)
                "C6",
        ]

        # corresponding black keys positioned between white keys where applicable
        black_keys_map = {
                "G2": "G#2", "A2": "A#2",
                "C3": "C#3", "D3": "D#3",
                "F3": "F#3", "G3": "G#3", "A3": "A#3",
                "C4": "C#4", "D4": "D#4",
                "F4": "F#4", "G4": "G#4", "A4": "A#4",
                "C5": "C#5", "D5": "D#5",
                "F5": "F#5", "G5": "G#5", "A5": "A#5",
        }

        # keyboard mapping (US layout) for white and black keys, spatially arranged left to right
        keymap_white = [
                "z", "x", "c",
                "v", "b", "n", "m", ",", ".", "/",
                "q", "w", "e", "r", "t", "y", "u",
                "i", "o", "p", "[", "]", "\\",
                "1",
        ]
        keymap_black = {
                "z": "s", "x": "d",
                "v": "g", "b": "h", "n": "j",
                    "m": "k", ",": "l", ".": ";",
                "q": "2", "w": "3",
                "r": "5", "t": "6", "y": "7",
                "i": "9", "o": "0",
                # higher row approximations
                "p": "-", "[": "=",
                "1": None,
        }

        # attempt to load audio files for each note and compute peak start times
        import base64

        def load_note_audio(note_name: str):
                folder = os.path.join(os.path.dirname(__file__), "gen_sounds", "high_quality")
                path = os.path.join(folder, f"{note_name}.wav")
                if not os.path.exists(path):
                        return None, None
                y, sr = sf.read(path, always_2d=False)
                if y.ndim == 2:
                        y = np.mean(y, axis=1)
                # compute index of maximum absolute amplitude to start playback at loudest point
                idx_peak = int(np.argmax(np.abs(y)))
                peak_time = float(idx_peak) / float(sr)
                # encode to base64 for embedding
                with open(path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                src = f"data:audio/wav;base64,{b64}"
                return src, peak_time

        notes_data = {}
        for wn in white_keys_order:
                src, t0 = load_note_audio(wn)
                if src is not None:
                        notes_data[wn] = {"src": src, "t0": t0}
        for wk, bk in black_keys_map.items():
                # include black keys only if their audio exists
                if bk:
                        src, t0 = load_note_audio(bk)
                        if src is not None:
                                notes_data[bk] = {"src": src, "t0": t0}

        # build a simple predefined (approximate) sequence for fur elise in e minor
        # sequence of (note, start_time_seconds, duration_seconds)
        fur_elise_sequence = [
            # --- Pickup ---
            ("E5", 0.00, 0.15), ("D#5", 0.15, 0.15),

            # --- Measure 1 (Main Motif) ---
            ("E5", 0.30, 0.15), ("D#5", 0.45, 0.15), ("E5", 0.60, 0.15),
            ("B4", 0.75, 0.15), ("D5", 0.90, 0.15), ("C5", 1.05, 0.15),

            # --- Measure 2 (A Minor Arpeggio) ---
            # Melody lands on A4; LH plays A-E-A
            ("A4", 1.20, 0.60), ("A2", 1.20, 0.15), # Downbeat
            ("E3", 1.35, 0.15),                     # LH
            ("A3", 1.50, 0.15),                     # LH
            # RH continues upward arpeggio
            ("C4", 1.65, 0.15), ("E4", 1.80, 0.15), ("A4", 1.95, 0.15),

            # --- Measure 3 (E Major Arpeggio) ---
            # Melody lands on B4; LH plays E-E-G#
            ("B4", 2.10, 0.60), ("E2", 2.10, 0.15), # Downbeat
            ("E3", 2.25, 0.15),                     # LH
            ("G#3", 2.40, 0.15),                    # LH
            # RH continues upward arpeggio
            ("E4", 2.55, 0.15), ("G#4", 2.70, 0.15), ("B4", 2.85, 0.15),

            # --- Measure 4 (Resolution) ---
            # Melody lands on C5; LH plays A-E-A
            ("C5", 3.00, 0.60), ("A2", 3.00, 0.15), # Downbeat
            ("E3", 3.15, 0.15),                     # LH
            ("A3", 3.30, 0.15),                     # LH
            # Pickup to repeat (E-D#...)
            ("E5", 3.45, 0.15), ("D#5", 3.60, 0.15),

            # --- Measure 5 (Repeat of Main Motif) ---
            ("E5", 3.75, 0.15), ("D#5", 3.90, 0.15), ("E5", 4.05, 0.15),
            ("B4", 4.20, 0.15), ("D5", 4.35, 0.15), ("C5", 4.50, 0.15),

            # --- Measure 6 (A Minor Arpeggio) ---
            ("A4", 4.65, 0.60), ("A2", 4.65, 0.15),
            ("E3", 4.80, 0.15),
            ("A3", 4.95, 0.15),
            ("C4", 5.10, 0.15), ("E4", 5.25, 0.15), ("A4", 5.40, 0.15),

            # --- Measure 7 (E Major Arpeggio) ---
            ("B4", 5.55, 0.60), ("E2", 5.55, 0.15),
            ("E3", 5.70, 0.15),
            ("G#3", 5.85, 0.15),
            ("D4", 6.00, 0.15), ("C5", 6.15, 0.15), ("B4", 6.30, 0.15),

            # --- Measure 8 (Final Resolution of Theme A) ---
            # Lands on A4; LH plays A-E-A
            ("A4", 6.45, 0.90), ("A2", 6.45, 0.30),
            ("E3", 6.75, 0.30), ("A3", 7.05, 0.30),
        ]

        st.write("All notes are generated with ODE to mimic violin.")

        # render interactive keyboard via html/js component
        import json
        from streamlit.components.v1 import html

        keyboard_notes = {
                "white": white_keys_order,
                "black": black_keys_map,
                "keymap_white": keymap_white,
                "keymap_black": keymap_black,
                "audio": notes_data,
                "sequence": fur_elise_sequence,
        }

        comp_height = 420
        html_content = f"""
        <style>
            .piano {{
                position: relative;
                width: 100%;
                max-width: 1200px;
                margin: 12px auto;
                user-select: none;
            }}
            .white-keys {{
                display: grid;
                grid-template-columns: repeat({len(white_keys_order)}, 1fr);
                gap: 2px;
            }}
            .white-key {{
                background: #fff;
                border: 1px solid #ccc;
                height: 160px;
                position: relative;
                text-align: center;
                font-family: sans-serif;
                font-size: 12px;
                line-height: 24px;
            }}
            .keycap-label {{
                position: absolute;
                top: 4px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0,0,0,0.06);
                border: 1px solid rgba(0,0,0,0.15);
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 11px;
                color: #333;
            }}
            .white-key.active {{
                background: #cfe8ff;
                border-color: #66a3ff;
            }}
            .black-keys {{
                position: absolute;
                left: 0;
                top: 0;
                width: 100%;
                height: 120px;
                pointer-events: none;
            }}
            .black-key {{
                position: absolute;
                width: calc(100% / {len(white_keys_order)} * 0.6);
                height: 120px;
                background: #000;
                border: 1px solid #333;
                transform: translateX(-50%);
                pointer-events: auto;
            }}
            .black-key .keycap-label {{
                background: rgba(255,255,255,0.15);
                border-color: rgba(255,255,255,0.25);
                color: #f5f5f5;
            }}
            .black-key.active {{
                background: #444;
            }}
            .controls {{
                margin: 8px 0 16px 0;
                font-family: sans-serif;
            }}
            .note-label {{ position: absolute; bottom: 4px; left: 4px; color: #555; }}
        </style>

        <div class="controls">
            <button id="play-seq">Play Fur Elise motif</button>
            <span id="status" style="margin-left:10px;color:#555"></span>
        </div>
        <div class="piano" id="piano"></div>

        <script>
            const data = {json.dumps(keyboard_notes)};

            // create audio elements for each note
            const audioMap = new Map();
            for (const [note, info] of Object.entries(data.audio)) {{
                const a = new Audio();
                a.src = info.src;
                a.preload = 'auto';
                a.loop = true; // sustain when held
                audioMap.set(note, {{ el: a, t0: info.t0 }});
            }}

            // build keyboard UI
            const piano = document.getElementById('piano');
            const whiteWrap = document.createElement('div');
            whiteWrap.className = 'white-keys';
            const blackWrap = document.createElement('div');
            blackWrap.className = 'black-keys';
            piano.appendChild(whiteWrap);
            piano.appendChild(blackWrap);

            // positions for black keys relative to white indices
            function whiteIndex(note) {{ return data.white.indexOf(note); }}

            const keyElems = new Map();

            // build note -> computer key label lookup
            const noteToKey = new Map();
            data.keymap_white.forEach((key, i) => {{
                const note = data.white[i];
                if (note) noteToKey.set(note, key);
            }});
            for (const [wn, bk] of Object.entries(data.black)) {{
                const whiteKeyChar = data.keymap_white[data.white.indexOf(wn)];
                const blackKeyChar = data.keymap_black[whiteKeyChar];
                if (bk && blackKeyChar) noteToKey.set(bk, blackKeyChar);
            }}

            data.white.forEach((note, idx) => {{
                const w = document.createElement('div');
                w.className = 'white-key';
                w.dataset.note = note;
                const keycap = document.createElement('div');
                keycap.className = 'keycap-label';
                keycap.textContent = noteToKey.get(note) || '';
                // shift labels for F, G, A to the left to avoid overlay
                const leading = note.charAt(0);
                if (leading === 'F' || leading === 'G' || leading === 'A' || leading === 'C') {{
                    keycap.style.left = '35%';
                    keycap.style.transform = 'translateX(-50%)';
                }}
                if (leading === 'D') {{
                    keycap.style.left = '35%';
                    keycap.style.transform = 'translateX(-65%)';
                }}
                const noteLbl = document.createElement('div');
                noteLbl.className = 'note-label';
                noteLbl.textContent = note;
                w.appendChild(keycap);
                w.appendChild(noteLbl);
                whiteWrap.appendChild(w);
                keyElems.set(note, w);

                const bk = data.black[note];
                if (bk && data.audio[bk]) {{
                    const b = document.createElement('div');
                    b.className = 'black-key';
                    b.dataset.note = bk;
                    // place over between current and next white key
                      const leftPercent = (idx + 1) / data.white.length * 100;
                      b.style.left = `calc(${{leftPercent}}% - (100% / ${{data.white.length}} * 0.2))`;
                                        const keycapB = document.createElement('div');
                                        keycapB.className = 'keycap-label';
                                        keycapB.textContent = noteToKey.get(bk) || '';
                                        b.appendChild(keycapB);
                    blackWrap.appendChild(b);
                    keyElems.set(bk, b);
                }}
            }});

            function playNote(note) {{
                const item = audioMap.get(note);
                if (!item) return;
                const {{ el, t0 }} = item;
                try {{
                    el.currentTime = t0;
                    el.play();
                    const k = keyElems.get(note);
                    if (k) k.classList.add('active');
                }} catch (e) {{ console.warn('play error', e); }}
            }}

            function stopNote(note) {{
                const item = audioMap.get(note);
                if (!item) return;
                const {{ el }} = item;
                try {{
                    el.pause();
                    const k = keyElems.get(note);
                    if (k) k.classList.remove('active');
                }} catch (e) {{ console.warn('pause error', e); }}
            }}

            // mouse interactions: click to toggle hold; click again to stop
            function handleKeyClick(evt) {{
                const note = evt.target.dataset.note || (evt.target.closest('[data-note]')?.dataset.note);
                if (!note) return;
                const item = audioMap.get(note);
                if (!item) return;
                if (item.el.paused) {{
                    playNote(note);
                }} else {{
                    stopNote(note);
                }}
            }}
            whiteWrap.addEventListener('click', handleKeyClick);
            blackWrap.addEventListener('click', handleKeyClick);

            // keyboard mapping: press to play, release to stop
            const whiteMap = new Map();
            data.keymap_white.forEach((key, i) => {{
                const note = data.white[i];
                if (note) whiteMap.set(key, note);
            }});
            const blackMap = new Map();
            for (const [wn, bk] of Object.entries(data.black)) {{
                const k = data.keymap_black[ data.keymap_white[ data.white.indexOf(wn) ] ];
                if (k && bk) blackMap.set(k, bk);
            }}

            const downSet = new Set();
            window.addEventListener('keydown', (e) => {{
                const key = e.key;
                if (downSet.has(key)) return; // avoid repeats
                let note = whiteMap.get(key) || blackMap.get(key);
                if (note) {{
                    downSet.add(key);
                    playNote(note);
                    e.preventDefault();
                }}
            }});
            window.addEventListener('keyup', (e) => {{
                const key = e.key;
                let note = whiteMap.get(key) || blackMap.get(key);
                if (note) {{
                    downSet.delete(key);
                    stopNote(note);
                    e.preventDefault();
                }}
            }});

            // play predefined fur elise motif and show active notes in realtime
            const statusEl = document.getElementById('status');
            document.getElementById('play-seq').addEventListener('click', () => {{
                statusEl.textContent = 'playing motif...';
                const tStart = performance.now() / 1000.0;
                for (const [note, t, dur] of data.sequence) {{
                    setTimeout(() => {{ playNote(note); }}, Math.max(0, (t - (performance.now()/1000.0 - tStart)) * 1000));
                    setTimeout(() => {{ stopNote(note); }}, Math.max(0, (t + dur - (performance.now()/1000.0 - tStart)) * 1000));
                }}
                setTimeout(() => {{ statusEl.textContent = ''; }}, ((data.sequence.at(-1)[1] + data.sequence.at(-1)[2]) - (performance.now()/1000.0 - tStart)) * 1000 + 200);
            }});
        </script>
        """

        html(html_content, height=comp_height)
