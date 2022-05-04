import pyaudio
import numpy as np

CHUNK = 1
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
RECORD_SECONDS = 60

nb_bits = 16
max_nb_bit = float(2 ** (nb_bits - 1))
max_nb_bit_p1 = max_nb_bit + 1

# Recording and output setup

mic1 = pyaudio.PyAudio()
mic2 = pyaudio.PyAudio()
output = pyaudio.PyAudio()

mic1_noisy = mic1.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK,
                       input_device_index=0)

mic2_noise = mic2.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK,
                       input_device_index=2)

output_cleaned = output.open(format=FORMAT,
                             channels=CHANNELS,
                             rate=RATE,
                             output=True,
                             frames_per_buffer=CHUNK,
                             output_device_index=1)

# Filter setup

FilterLength = 4
mu = 125e-6
weights = np.zeros(FilterLength)

# Noise sample array for filtering

noise_arr = np.zeros(FilterLength)

# Main

print("* recording")

for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
    noisy = mic1_noisy.read(CHUNK)
    noise = mic2_noise.read(CHUNK)

    noisy_norm = np.frombuffer(noisy, dtype=np.int16) / max_nb_bit_p1
    noise_norm = np.frombuffer(noise, dtype=np.int16) / max_nb_bit_p1

    noise_arr = np.concatenate([noise_norm, noise_arr[CHUNK:]])

    signal_out = np.matmul(weights, noise_arr)
    err = noisy_norm - signal_out
    weights = weights + mu * err * noise_arr

    cleaned = ((err * max_nb_bit_p1).astype(np.int16).tobytes())
    output_cleaned.write(cleaned, CHUNK)

print("* done")

mic1_noisy.stop_stream()
mic1_noisy.close()
mic2_noise.stop_stream()
mic2_noise.close()
output_cleaned.stop_stream()
output_cleaned.close()

mic1.terminate()
mic2.terminate()
output.terminate()
