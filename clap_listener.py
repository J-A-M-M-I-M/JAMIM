import sounddevice as sd
import numpy as np
import time
import os
import subprocess
from pydub import AudioSegment
from pydub.playback import _play_with_ffplay
import threading

with open('/tmp/devices.txt', 'w') as f:
    f.write(str(sd.query_devices()))
THRESHOLD = 0.7  # 박수 감지 임계값 (실험 필요)
CLAP_INTERVAL = 3  # 초, 박수 두 번 감지할 시간 범위
LOCK_FILE = "/tmp/ui_app.lock"

clap_times = []

def play_bgm():
    print("playing bgm,,")
    sound = AudioSegment.from_file("/Users/hyeongjeongyi/madcamp/JAMIM/src/jamimBGM.mp3", format="mp3")
    threading.Thread(target=lambda: _play_with_ffplay(sound)).start()

def is_app_running():
    return os.path.exists(LOCK_FILE)

def mark_app_running():
    with open(LOCK_FILE, "w") as f:
        f.write("running")

def clear_app_running():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

def audio_callback(indata, frames, time_info, status):
    global clap_times
    volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
    current_time = time.time()

    if volume_norm > THRESHOLD:
        clap_times = [t for t in clap_times if current_time - t < CLAP_INTERVAL]
        clap_times.append(current_time)

        if len(clap_times) >= 2 and not is_app_running():
            print("👏👏 박수 두 번 감지! 앱 실행 중...")
            play_bgm()
            mark_app_running()
            subprocess.Popen(
                ["python3", "/Users/hyeongjeongyi/madcamp/JAMIM/ui.py"],
                start_new_session=True
            ).wait()
            clear_app_running()
            clap_times.clear()
        elif is_app_running():
            print("앱이 이미 실행 중입니다. 무시됨.")

def main():
    print("🎤 박수 감지 대기 중... (Ctrl+C로 종료)")
    if is_app_running():
        clear_app_running()

    with sd.InputStream(callback=audio_callback):
        while True:
            sd.sleep(1000)

if __name__ == "__main__":
    main()
