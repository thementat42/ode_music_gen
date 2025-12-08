import numpy as np
import scipy.integrate as integrate
import wave
import struct

# Constants
P_H = 0.04  # Height of the triangle wave (maximum displacement)
L = 32.8/100  # Length of the string (32.8 cm)
NOTE = "E"  # change this

def get_constants(note = NOTE):
    # data from https://ccrma.stanford.edu/~jos/stiffbowed/Modeling_stiffness_string.html
    # and from 
    # note: the lower end of the mass/length in the pdf was chosen
    notes = {
        "E": {
            "Y": 15.7,
            "mu": 0.38/1000,  # need kg/m (table is in g/m)
            "diameter": 0.000307,
            "speed": 432.5
        },
        "A": {
            "Y": 7.470,
            "mu": 0.58/1000,  # need kg/m
            "diameter": 0.000676,
            "speed": 288.64
        },
        "D": {
            "Y": 6.4,
            "mu": 0.92/1000,  # need kg/m
            "diameter": 0.000795,
            "speed": 192.67
        },
        "G": {
            "Y": 6.035,
            "mu": 2.12/1000,  # need kg/m
            "diameter": 0.000803,
            "speed": 128.58
        }
    }
    if note not in notes.keys():
        raise Exception(f"Note should be one of {notes.keys()}")
    vals = notes[note]
    Y = vals["Y"]
    mu = vals["mu"]
    d = vals["diameter"]
    c = vals["speed"]

    S = (1/4*np.pi*(d**2))*L  # surface area
    # K = sqrt(I/M)
    # string rotating about end I=1/3 ML^2
    # so K = l/sqrt3
    K = L/np.sqrt(3)

    alpha = Y*S*(K**2)/(mu)
    return c, alpha
    

c, alpha = get_constants()
gamma = 0.1  # Damping constant


# Time parameters
sampling_rate = 44100  # Audio sampling rate (samples per second)
duration = 5  # Duration of the sound in seconds
num_samples = duration * sampling_rate

# Triangle wave for f0(x)
def f0(x):
    slope = 2 * P_H / L
    if x < L / 2:
        return slope * x
    else:
        return slope * (L-x)

# Precompute the inner product to avoid long integral
def precompute_coefficients(n_max = 10):
    coefficients = np.zeros(n_max)

    for n in range(1, n_max+1):  # offset since the sum starts at 1
        integrand = lambda x: f0(x) * np.sin(n * np.pi * x / L)
        coefficients[n-1], *_ = integrate.quad(integrand, 0, L)

    return coefficients

# Define Omega_n for each mode n
def omega_n(n):
    k_n = n*np.pi/L
    omega_n_sq = (c**2) * k_n**2 + alpha * k_n**4
    return np.sqrt(omega_n_sq - (gamma**2 / 4))

# Generate the time series for the wave at each time t
def generate_wave(coefficients, t):
    wave = 0.0
    decay = np.exp(-gamma*t/2)
    for n, coeff in enumerate(coefficients, start=1):
        Omega_n = omega_n(n)
        wave += (2 / L) * coeff * decay * (
            np.cos(Omega_n * t) + (2 * Omega_n / gamma) * np.sin(Omega_n * t)
        ) * np.sin(n * np.pi * t / L)
    return wave

# Generate a .wav file from the computed wave
def generate_wav_file(filename, wave_data):
    """Generate a .wav file from the wave data"""
    # Normalize wave data to fit in the 16-bit audio range
    max_amplitude = np.max(np.abs(wave_data))
    normalized_wave_data = np.int16(wave_data / max_amplitude * 32767)

    # Write the wave data to a .wav file
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)  # Mono sound
        wf.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
        wf.setframerate(sampling_rate)
        for sample in normalized_wave_data:
            wf.writeframes(struct.pack('h', sample))

# Main function to generate the .wav file for the damped wave
def main():
    # Precompute the coefficients (the integrals of f0 with the basis functions)
    coefficients = precompute_coefficients()

    # Time array (t from 0 to duration)
    time_array = np.linspace(0, duration, num_samples, endpoint=False)

    # Generate the wave at each time step
    wave_data = np.array([generate_wave(coefficients, t) for t in time_array])

    # Generate the .wav file
    generate_wav_file(f'gen_sounds/analytic/{NOTE}.wav', wave_data)

# Run the main function
if __name__ == "__main__":
    main()
