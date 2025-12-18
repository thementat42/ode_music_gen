import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.io.wavfile import read, write

#Load violin recording
print("Loading audio file...")
fs, violin = read("gen_sounds/full_ODE/A4.wav")
violin = violin.astype(float)
if violin.ndim > 1:
    violin = violin.mean(axis=1)
violin /= np.max(np.abs(violin))

duration = len(violin) / fs
t = np.linspace(0, duration, len(violin))
print(f"Loaded: {duration:.2f} seconds, {fs} Hz sample rate")

## ----------- room parameters ----------- ##
# #Natural room
# f_rooms = [45, 80, 120, 180, 250]
# zeta = [0.08, 0.1, 0.12, 0.12, 0.14]

#Big Hall Sound
''' The resonances of these modes is amplifying the fundamentals which adds to the bass sound'''
f_rooms = [50, 95, 140, 220, 380]  
zeta = [0.06, 0.07, 0.08, 0.09, 0.1]  # Lower damping = longer resonance ring

# # Tiny boxy room
# f_rooms = [150, 300, 450, 600, 900]
# zeta = [0.08, 0.08, 0.08, 0.08, 0.08]

omega = 2 * np.pi * np.array(f_rooms)

# Coupling strength - controls how much the room affects the sound
coupling = 1.5  # I found anything >0.8 is pretty strong effect



## ----------- room ODE system ----------- ##
def multimode_room(t_now, y):
    dydt = np.zeros_like(y)
    input_force = np.interp(t_now, np.arange(len(violin))/fs, violin)
    
    for i in range(5):  
        pos, vel = 2*i, 2*i+1
        dydt[pos] = y[vel]
        dydt[vel] = -2*zeta[i]*omega[i]*y[vel] - omega[i]**2*y[pos] + coupling * omega[i]**2 * input_force
    return dydt

#Solve ODE system
print("Solving room dynamics...")
y0 = np.zeros(2 * len(f_rooms))  

# Sparser time points
t_eval = np.linspace(0, duration, 2000)  

sol = solve_ivp(multimode_room, [0, duration], y0, t_eval=t_eval, method='RK45')
print(f"Solver complete: {len(sol.t)} points computed")

# # Solve with many timesteps
# n_points = min(len(violin), fs * int(duration) + 1)
# t_eval = np.linspace(0, duration, n_points)

# sol = solve_ivp(multimode_room, [0, duration], y0, t_eval=t_eval, 
#                 method='RK45', max_step=0.001)
# print(f"Solver complete: {len(sol.t)} points computed")

# positions and interpolate to original length
t = np.linspace(0, duration, len(violin))
y_modes = []
for i in range(0, 10, 2):  # 10 values for 5 modes
    y_modes.append(np.interp(t, sol.t, sol.y[i]))
y_modes = np.array(y_modes)

y_room = np.sum(y_modes, axis=0)



## ----------- add echo effects ----------- ##
def add_echo(signal, delays_ms, gains, sample_rate):
    ''' This was pretty important cause it made the resonance sounds much more obvious (also made it sound more dramatic lol)
        Echo delays that are longer create an effect that amplifies and reinforces low frequencies - largely responsible for the strong bass sound
        Maybe add high pass filter since real rooms reflect higher frequencies more than bass
    '''
    output = signal.copy()
    for delay_ms, gain in zip(delays_ms, gains):
        delay_samples = int(delay_ms * sample_rate / 1000)
        if delay_samples < len(signal):
            echo = np.zeros_like(signal)
            echo[delay_samples:] = signal[:-delay_samples] * gain
            output += echo
    return output

# Multiple echoes with decay
#Longer delay echos
# echo_delays = [150, 300, 450, 600]  # milliseconds
# echo_gains = [0.5, 0.3, 0.2, 0.1]   # decreasing volume

# # Shorter, tighter echoes
echo_delays = [25, 50, 75, 100]  
echo_gains = [0.3, 0.3, 0.2, 0.1]   

y_room_with_echo = add_echo(y_room, echo_delays, echo_gains, fs)



## ----------- mix original signal with room response ----------- ##
#mix_ratio = 0.4  # Amount of original sound (1-mix_ratio is % of room resonance sound)
mix_ratio = 0.7 
y_total = mix_ratio * violin + (1 - mix_ratio) * y_room_with_echo

# Normalize 
y_total /= np.max(np.abs(y_total)) * 1.1

# Debug output
print(f"\nSignal stats:")
print(f"y_room range: [{np.min(y_room):.6f}, {np.max(y_room):.6f}]")
print(f"y_total range: [{np.min(y_total):.6f}, {np.max(y_total):.6f}]")
print(f"Max amplitude: {np.max(np.abs(y_total)):.3f}")



## ----------- output ----------- ##
#Save processed sound
scaled = np.int16(y_total * 32767)
write("violin_in_room.wav", fs, scaled)
print("\n✓ Saved 'violin_in_room.wav'")




#Plot comparison
plt.figure(figsize=(12, 8))

plt.subplot(4,1,1)
plt.plot(t, violin, color='gray', linewidth=0.5)
plt.title("Original Violin Input")
plt.ylabel("Amplitude")

plt.subplot(4,1,2)
plt.plot(t, y_room_with_echo, color='blue', linewidth=0.5)
plt.title("Room Response")
plt.ylabel("Amplitude")

plt.subplot(4,1,3)
plt.plot(t, y_total, color='k', linewidth=0.5)
plt.title("Mixed Output (Violin + Room)")
plt.ylabel("Amplitude")

# Spectrogram of output - honestly not really neccesary probably
plt.subplot(4,1,4)
plt.specgram(y_total, Fs=fs, cmap='viridis', NFFT=1024)
plt.title("Spectrogram of Output")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.ylim(0, 2000)

plt.tight_layout()
plt.show()