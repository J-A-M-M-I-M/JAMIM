import subprocess
import os

electron_app_path = "/Users/hyeongjeongyi/madcamp/jammim_ui/jammim/jammim"
log_file = "/tmp/electron_test.log"

with open(log_file, "a") as f:
    f.write("[TEST] Electron 앱 실행 테스트 시작\n")

log = open(log_file, "a")

subprocess.Popen(
    ["npm", "run", "start"],
    cwd=electron_app_path,
    stdout=log,
    stderr=log,
    preexec_fn=os.setpgrp
)
