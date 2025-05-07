import os
import subprocess

folder = r"data\Forest Recordings"

# Path to ffmpeg executable
ffmpeg_path = r"C:\Users\dantn\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

for fname in os.listdir(folder):
    if fname.lower().endswith(".mp3"):
        mp3_path = os.path.join(folder, fname)
        wav_path = os.path.join(folder, fname.rsplit(".", 1)[0] + ".wav")

        print(f"Converting: {fname} → {os.path.basename(wav_path)}")

        result = subprocess.run([
            ffmpeg_path,
            "-y",
            "-i", mp3_path,
            wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result.returncode == 0:
            os.remove(mp3_path)
            print(f"Deleted: {fname}")
        else:
            print(f"Conversion failed for: {fname}")

print("All .mp3 files converted and deleted.")
