import sounddevice as sd
from scipy.io.wavfile import write
import datetime

# Settings
duration = 3  # seconds
sample_rate = 22050
device_index = 1  # use your Q9-1 mic

# Record
print("Recording for 3 seconds...")
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device_index)
sd.wait()

# Filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"mic_recording_{timestamp}.wav"

# Save
write(filename, sample_rate, recording)
print(f"Saved recording as {filename}")