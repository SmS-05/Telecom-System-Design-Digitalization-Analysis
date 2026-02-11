import matplotlib.pyplot as plt
import sounddevice as sd
import time as TIME 
from scipy.io import wavfile
import numpy as np
from scipy import signal


fs = 44100
duration = 10
#Recording first file
print("First Recording Starts in 3",end = "")
TIME.sleep(1)
print(" 2",end="")
TIME.sleep(1)
print(" 1")
TIME.sleep(1)
print("Recording...")
audio_1 = sd.rec(int(duration*fs),samplerate=fs, channels=1)
sd.wait()
wavfile.write("audio_1.wav",fs,audio_1.astype(np.float32))
print("First Recording Finished!")
TIME.sleep(1.5)

#Recording second file
print("Second Recording Starts in 3",end = "")
TIME.sleep(1)
print(" 2",end="")
TIME.sleep(1)
print(" 1")
TIME.sleep(1)
print("Recording...")
audio_2 = sd.rec(int(duration*fs),samplerate=fs, channels=1)
sd.wait()
wavfile.write("audio_2.wav",fs,audio_2.astype(np.float32))
print("Second Recording Finished!")
TIME.sleep(1.5)

#Recording third file
print("Third Recording Starts in 3",end = "")
TIME.sleep(1)
print(" 2",end="")
TIME.sleep(1)
print(" 1")
TIME.sleep(1)
print("Recording...")
audio_3 = sd.rec(int(duration*fs),samplerate=fs, channels=1)
sd.wait()
wavfile.write("audio_3.wav",fs,audio_3.astype(np.float32))
print("Third Recording Finished!")


#step 1

x = np.squeeze(audio_1)
xq_8 = np.round(x*(2**(8-1)-1))/(2**(8-1)-1)
xq_3 = np.round(x*(2**(3-1)-1))/(2**(3-1)-1)
wavfile.write("audio_1_8bit.wav",fs,xq_8.astype(np.float32))
wavfile.write("audio_1_3bit.wav",fs,xq_3.astype(np.float32))

peak_index = np.argmax(np.abs(x))
segment_length = int(0.1*fs)

start = max(0, peak_index - segment_length)
end = min(len(x), peak_index + segment_length)

original_samples = x[start:end]
quantized_samples = xq_3[start:end]

idx = np.arange(start, end)
time = idx / fs * 1000

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time, original_samples)
plt.title("Original Signal (around peak)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, quantized_samples, color="orange")
plt.title("Quantized Signal (3-bit) (around peak)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.savefig("Original VS 3-bit Quantized.png", dpi=300)
plt.close()

#Step 2

fc1=5000
fc2=12000
fc3=19000
audio_1 = np.squeeze(audio_1)
audio_2 = np.squeeze(audio_2)
audio_3 = np.squeeze(audio_3)

t = np.arange(len(audio_1))/fs
y1 = audio_1*np.cos(2*np.pi*fc1*t)
y2 = audio_2*np.cos(2*np.pi*fc2*t)
y3 = audio_3*np.cos(2*np.pi*fc3*t)
Yt = y1 + y2 + y3

Yf = np.fft.fft(Yt)

freq = np.fft.fftfreq(len(Yt), d=1/fs)
Yf_mag = np.abs(Yf)
positive_indices_fdm = freq >= 0
positive_freq = freq[positive_indices_fdm]
positive_magnitude = Yf_mag[positive_indices_fdm]



plt.figure(figsize=(12, 6))
plt.plot(positive_freq, positive_magnitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Spectrum of FDM Signal (FFT)")
plt.grid(True)
plt.xlim(0,fs/2)
plt.savefig("Spectrum of Total Signal.png",dpi=300)
plt.close()

#Step 3

fs_high = fs
fs_low  = 20000

N1 = len(Yt)
N2 = int(round(N1 * fs_low / fs_high))

Yt_low = signal.resample(Yt, N2)

Yf1 = np.fft.fft(Yt)
f1 = np.fft.fftfreq(N1, d=1/fs_high)
mag1 = np.abs(Yf1)

positive_idx1 = f1 >= 0
f1_pos = f1[positive_idx1]
mag1_pos = mag1[positive_idx1]

Yf2 = np.fft.fft(Yt_low)
f2 = np.fft.fftfreq(N2, d=1/fs_low)
mag2 = np.abs(Yf2)

positive_idx2 = f2 >= 0
f2_pos = f2[positive_idx2]
mag2_pos = mag2[positive_idx2]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(f1_pos, mag1_pos)
plt.title("Original Signal Spectrum (FS = 44.1 kHz) - No Aliasing")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.xlim(0, fs_high/2)

plt.subplot(2, 1, 2)
plt.plot(f2_pos, mag2_pos,color = 'orange')
plt.title("Aliased Signal Spectrum (FS=20000 Hz) - Aliasing Appeared")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.xlim(0, fs_low/2)

plt.tight_layout()
plt.savefig("Two rate Sampling.png", dpi=300)
plt.close()

#step 4


nyq = 0.5 * fs
low, high = 9000, 15000
order = 200
b_bp = signal.firwin(order + 1, [low/nyq, high/nyq], pass_zero=False)
extracted_y2 = signal.lfilter(b_bp, 1, Yt)
wavfile.write("extracted_y2.wav", fs, extracted_y2.astype(np.float32))

extracted_y2f = np.fft.fft(extracted_y2)
extracted_y2_frequency = np.fft.fftfreq(len(extracted_y2), d=1/fs)

positive_indices_bp = extracted_y2_frequency >= 0
positive_freq = extracted_y2_frequency[positive_indices_bp]
positive_magnitude = np.abs(extracted_y2f)[positive_indices_bp]

plt.figure(figsize=(12, 6))
plt.plot(positive_freq, positive_magnitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Spectrum after Band-pass Filtering ")
plt.grid(True)
plt.xlim(0, positive_freq.max())
plt.savefig("Band pass filtered Signal.png", dpi=300)
plt.close()

#step 5

demod_raw = extracted_y2 * np.cos(2*np.pi * fc2*t) * 2
b_lp = signal.firwin(201, 4000/nyq)
recovered_audio2 = signal.lfilter(b_lp, 1, demod_raw)

wavfile.write("recovered_audio2.wav", fs, recovered_audio2.astype(np.float32))
recovered_audio2 = np.squeeze(recovered_audio2)

n = np.arange(len(audio_2))
time = n / fs

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time, audio_2)
plt.title("Original Audio")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, recovered_audio2, color="orange")
plt.title("Recovered Audio ")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.savefig("Original VS Recovered Signal.png", dpi=300)
plt.close()
