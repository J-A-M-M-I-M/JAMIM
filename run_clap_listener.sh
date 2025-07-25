#!/bin/bash

# Conda 환경 설정 명확하게
source /Users/hyeongjeongyi/opt/anaconda3/etc/profile.d/conda.sh
export PATH="/Users/hyeongjeongyi/opt/anaconda3/bin:/opt/homebrew/bin:$PATH"


# jamim 환경 활성화
conda activate jamim

# 로그 기록해보기
echo "$(date) - running clap_listener" >> /tmp/claplistener.debug.log
which python >> /tmp/claplistener.debug.log
which ffmpeg >> /tmp/claplistener.debug.log

# 파이썬 실행
python /Users/hyeongjeongyi/madcamp/JAMIM/clap_listener.py >> /tmp/claplistener.stdout.log 2>> /tmp/claplistener.stderr.log
