# Telecom-System-Design-Digitalization-Analysis

Digital implementation and analysis of a multi-channel telecommunication
system using sampling, quantization, frequency division multiplexing
(FDM), filtering, and demodulation techniques.

## Course Information

-   University: K. N. Toosi University of Technology (KNTU)
-   Course: Signals and Systems
-   Instructor: Dr. Moradian
-   Term: 4041
-   Programming Language: Python 3.14

## Project Overview

This project presents a complete digital simulation of a
telecommunication system including:

-   Audio signal recording
-   Uniform quantization (3-bit and 8-bit)
-   Frequency Division Multiplexing (FDM)
-   Fourier Transform analysis (FFT)
-   Resampling and Nyquist analysis
-   Band-pass filtering
-   Demodulation and signal recovery

The goal is to analyze how digital signal processing techniques affect
signal quality, spectral characteristics, and system performance.

## System Implementation Steps

### 1. Audio Recording

Three separate audio signals are recorded using `sounddevice`.

### 2. Quantization

The first audio signal is quantized using 8-bit and 3-bit quantization,
and comparison plots are generated.

### 3. Frequency Division Multiplexing (FDM)

Each signal is modulated using cosine carriers: - fc1 = 5 kHz - fc2 = 12
kHz - fc3 = 19 kHz

All modulated signals are summed to form a composite signal.

### 4. Resampling & Nyquist Analysis

The signal is resampled using `resample_poly` and `signal.resample`, and
the effect of sampling rate reduction and aliasing is analyzed.

### 5. Band-Pass Filtering

A FIR band-pass filter is designed using `scipy.signal.firwin` to
extract the second channel.

### 6. Demodulation

The extracted channel is multiplied by its carrier and passed through a
low-pass filter to recover the baseband signal.

## Dependencies

Install required packages:

``` bash
pip install numpy scipy matplotlib sounddevice
```

## How to Run

``` bash
python main.py
```

Ensure microphone access is enabled.

## License

This project is licensed under the MIT License.
