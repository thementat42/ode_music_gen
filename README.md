# ODE Music Generator

An audio harmonic analyzer and resynthesizer web application built with Streamlit.

## Features

- Upload audio files (wav, mp3, flac, ogg, m4a)
- View audio spectrum A(f)
- Detect peaks in the frequency domain
- Resynthesize audio from detected peaks for A/B comparison

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

## Usage

1. Upload an audio file using the file uploader
2. Adjust the parameters:
   - Sample rate
   - Maximum peaks to detect
   - Minimum prominence threshold
   - Frequency range (min/max)
   - Resynthesis duration
   - Scaling multiplier
3. View the original spectrum and detected peaks
4. Listen to both original and resynthesized audio for comparison
