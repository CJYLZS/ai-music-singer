#!/bin/bash
# E3b: 修正版 E3 —— seed-vc 预转换产物已就位（lead_prevc_*.wav），直接跑 SVC+eval。
# 等 E1b 结束后启动（避免 GPU 争抢）。
set -u
ROOT=/home/rookie/voice
GSV=$ROOT/GPT-SoVITS/.venv/bin/python
SING=$ROOT/sing.py
CER=$ROOT/singing/eval_cer.py
REF=$ROOT/data/voices/ref_talk_sing.wav
cd $ROOT

while ! grep -q "ALL E1B ARMS DONE" .tmp/e1b_run.log 2>/dev/null; do sleep 20; done
echo "E1b 已结束，E3b 开始 $(date +%H:%M:%S)"

run_arm () {
  local song=$1 arm=$2 srcwav=$3
  local d=$ROOT/data/songs/$song
  local log=$d/log_e3b_$arm.log
  echo "==== [$song/$arm] start $(date +%H:%M:%S)" | tee -a $log
  $GSV $SING --ref $REF --ref-is-clean --song $ROOT/data/songs/$song.flac --name $song \
      --stages svc,eval --seed 1 --source-wav "$srcwav" >> $log 2>&1
  cp $d/vocal.wav $d/vocal_e3b_$arm.wav
  cp $d/eval.txt $d/eval_e3b_$arm.txt
  $GSV $CER --src $d/sep2/lead.wav --gen $d/vocal_e3b_$arm.wav --labels $arm \
      --device cuda:0 >> $log 2>&1
  echo "==== [$song/$arm] done $(date +%H:%M:%S)" | tee -a $log
}

for song in 白色毛衣 去有风的地方; do
  d=$ROOT/data/songs/$song
  run_arm $song neutral3 $d/lead_prevc_neutral3.wav
  run_arm $song target $d/lead_prevc_target.wav
done
echo "ALL E3B ARMS DONE $(date +%H:%M:%S)"
