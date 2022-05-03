import pyaudio

CHUNK = 128
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10

mic = pyaudio.PyAudio()
ear = pyaudio.PyAudio()

input_stream = mic.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=0)

output_stream = ear.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         output=True,
                         frames_per_buffer=CHUNK,
                         input_device_index=1)


print("* recording")

for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
    data = input_stream.read(CHUNK)
    output_stream.write(data, CHUNK)

print("* done")

input_stream.stop_stream()
input_stream.close()
output_stream.stop_stream()
output_stream.close()

mic.terminate()
ear.terminate()