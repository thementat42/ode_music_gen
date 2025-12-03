# ODE Music Analyzer & Violin Note Playground

A Streamlit app for synthesizing violin notes using ODEs.

## Features
- Single audio analysis: FFT spectrum, peak picking, sinusoidal resynthesis.
- Two‑recording comparison: automatic fundamental detection, harmonic window search, amplitude MSE table and plots.
- Playground: clickable piano (G2–C6) and keyboard control; starts playback at each note’s loudest point; highlights pressed keys; demo motif playback.

## Setup

### Option 1: Using Conda (Recommended)

1. Create the conda environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate ode_music_gen
```

### Option 2: Using pip

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure your environment is activated (if using conda)
2. Run the Streamlit app:
```bash
streamlit run web_app.py
```

3. Open your browser to the URL shown in the terminal (usually http://localhost:8501)

## Dependencies

- Python 3.11
- Streamlit - Web framework
- NumPy - Numerical computing
- Matplotlib - Plotting
- Librosa - Audio analysis
- SoundFile - Audio file I/O

## License
This repository is for academic use; no explicit license provided.