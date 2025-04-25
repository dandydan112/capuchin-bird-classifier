import os
import subprocess

folder = r"data\Forest Recordings"

for fname in os.listdir(folder):
    if fname.lower().endswith(".mp3"):
        mp3_path = os.path.join(folder, fname)
        wav_path = os.path.join(folder, fname.rsplit(".", 1)[0] + ".wav")

        print(f"Converting: {fname} → {os.path.basename(wav_path)}")

        # Run ffmpeg command
        subprocess.run([
            r"C:\Users\Dan\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe",
            "-y",  # Overwrite if needed
            "-i", mp3_path,
            wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("All .mp3 files converted to .wav")
