import wave
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.ndimage import uniform_filter1d
import librosa.feature as lf
from scipy.signal import hilbert

def moving_average(data, window_size):
    """Calculate the moving average of the given data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')
# Open the WAV file
def calculate_zcr(signal, frame_size, hop_size):
    zcr = []
    frame_count = int((len(signal) - frame_size) / hop_size) + 1
    for i in range(frame_count):
        frame = signal[i*hop_size : i*hop_size+frame_size]
        crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        zcr.append(crossings)
    return np.array(zcr)

wav_file = wave.open('/raid/home/kirtil/gautami/up/HF_Lung_V1-master/train/train/steth_20180814_09_37_11.wav', 'r')
audio_data = wav_file.readframes(-1)
audio_data = np.frombuffer(audio_data, dtype=np.int16)
frame_rate = wav_file.getframerate()
duration = len(audio_data) / frame_rate

fft_data = np.fft.fft(audio_data)
frequency = np.fft.fftfreq(len(audio_data), d=1/frame_rate)
fft_data_filtered = fft_data.copy()
fft_data_filtered[(np.abs(frequency) < 50) | (np.abs(frequency) > 1600)] = 0 # removing frequecies from a certain range
filtered_audio_data = np.fft.ifft(fft_data_filtered).real

wavelet_name = 'coif2'
n_layers = 9
# Perform wavelet decomposition
coeffs = pywt.wavedec(filtered_audio_data, wavelet_name, level=n_layers)
sigma = np.median(np.abs(coeffs[-1])) / 0.6745
uthresh = sigma * np.sqrt(2*np.log(len(filtered_audio_data)))
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], uthresh, mode='soft')

denoised_signal = pywt.waverec(coeffs, wavelet_name)

# Find the envelope of the signal using the Hilbert transform
analytic_signal = hilbert(denoised_signal)
amplitude_envelope = np.abs(analytic_signal)

# Example modification: scaling the envelope
modified_envelope = amplitude_envelope * 1.5

# Smoothing the modified envelope using moving average
window_size = 100
smoothed_envelope = moving_average(modified_envelope, window_size)

desired_duration = 15
num_samples = int(desired_duration * frame_rate)
trimmed_time = np.linspace(0, desired_duration, num=num_samples)

audio_data = (audio_data - np.mean(audio_data)) / np.std(audio_data)
denoised_signal = (denoised_signal - np.mean(denoised_signal)) / np.std(denoised_signal)

frame_size = int(0.025 * frame_rate) 
hop_size = int(0.010 * frame_rate) 

fig, axs = plt.subplots(3, 1, figsize=(12, 6))

frame_size = int(0.025 * frame_rate) # Frame size of 25ms
hop_size = int(0.010 * frame_rate) # Hop size of 10ms
zcr = calculate_zcr(denoised_signal, frame_size, hop_size)

axs[0].plot(trimmed_time,audio_data[:num_samples])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Original Waveform of Audio')


axs[1].plot(trimmed_time, denoised_signal[:num_samples])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('Denoised Waveform of Audio')

axs[2].plot(trimmed_time, smoothed_envelope[:num_samples])
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Amplitude')
axs[2].set_title('Smoothed Envelope of Audio')

with open('/raid/home/kirtil/gautami/up/HF_Lung_V1-master/train/train/steth_20180814_09_37_11_label.txt', 'r') as label_file:
    labels = label_file.readlines()

label_colors = {'I': 'blue', 'Rhonchi': 'red', 'E': 'purple', 'Wheeze': 'orange', 'Stridor': 'brown', 'Crackle': 'pink'}
label_styles = {'I': 'solid', 'Rhonchi': 'dashdot', 'E': 'dotted', 'Wheeze': 'solid', 'Stridor': 'dashed', 'Crackle': 'dashdot'}

for label in labels:
    label_parts = label.split()
    label_type = label_parts[0]
    start_time = label_parts[1]
    end_time = label_parts[2]

    start_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(':'))))
    end_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(':'))))

    if label_type in label_colors and label_type in label_styles:
        axs[0].axvspan(start_seconds, end_seconds, alpha=0.3, color=label_colors[label_type], linestyle=label_styles[label_type], label=label_type)
        axs[1].axvspan(start_seconds, end_seconds, alpha=0.3, color=label_colors[label_type], linestyle=label_styles[label_type], label=label_type)
        axs[2].axvspan(start_seconds, end_seconds, alpha=0.3, color=label_colors[label_type], linestyle=label_styles[label_type], label=label_type)

axs[2].legend()
plt.tight_layout()
plt.show()
