#!/bin/bash
# E3: seed-vc 预转换 A/B —— 先把源 lead 转成"中性第三说话人"/"目标说话参考"，
# 再走 SoulX-Singer（--source-wav 注入），检验"源音色归一化降低 SVC 难度"。
# 三臂 × 两首歌，--seed 1 配对。seed-vc 参考截断 25s（inference.py 硬限）。
set -u
ROOT=/home/rookie/voice
GSV=$ROOT/GPT-SoVITS/.venv/bin/python
SING=$ROOT/sing.py
CER=$ROOT/singing/eval_cer.py
REF=$ROOT/data/voices/ref_talk_sing.wav
SEEDVC=$ROOT/seed-vc/.venv/bin/python
SEEDVC_INF=$ROOT/seed-vc/inference.py
cd $ROOT

# 25s 参考切片（seed-vc 只取前 25s）
$GSV - <<'EOF'
import sys
import soundfile as sf
import numpy as np
jobs = [
    ("data/voices/ref_talk_sing.wav",  "data/voices/ref_talk_sing_25s.wav"),
    ("data/songs/冬眠/sep2/lead.wav",  "data/songs/冬眠/lead_25s.wav"),
]
for src, dst in jobs:
    y, sr = sf.read(src)
    y = y[: sr * 25]
    sf.write(dst, y, sr)
    print("slice", dst, len(y) / sr, "s")
EOF

run_arm () {
  local song=$1 arm=$2 srcwav=$3
  local d=$ROOT/data/songs/$song
  local log=$d/log_e3_$arm.log
  echo "==== [$song/$arm] start $(date +%H:%M:%S)" | tee -a $log
  local extra=()
  if [ -n "$srcwav" ]; then extra=(--source-wav "$srcwav"); fi
  $GSV $SING --ref $REF --ref-is-clean --song $ROOT/data/songs/$song.flac --name $song \
      --stages svc,eval --seed 1 "${extra[@]}" >> $log 2>&1
  cp $d/vocal.wav $d/vocal_e3_$arm.wav
  cp $d/eval.txt $d/eval_e3_$arm.txt
  $GSV $CER --src $d/sep2/lead.wav --gen $d/vocal_e3_$arm.wav --labels $arm \
      --device cuda:0 >> $log 2>&1
  echo "==== [$song/$arm] done $(date +%H:%M:%S)" | tee -a $log
}

for song in 白色毛衣 去有风的地方; do
  d=$ROOT/data/songs/$song
  echo "== [E3/seedvc] $song neutral3  (冬眠 lead 25s 为中性参考)"
  (cd $ROOT/seed-vc && $SEEDVC $SEEDVC_INF \
      --source $d/sep2/lead.wav \
      --target $ROOT/data/songs/冬眠/lead_25s.wav \
      --output $d/lead_seedvc_neutral3.wav \
      --f0-condition True --auto-f0-adjust False --semi-tone-shift 0 --fp16 True) \
      >> $d/log_e3_neutral3.log 2>&1
  run_arm $song neutral3 $d/lead_seedvc_neutral3.wav
  echo "== [E3/seedvc] $song target (ref_talk_sing 25s)"
  (cd $ROOT/seed-vc && $SEEDVC $SEEDVC_INF \
      --source $d/sep2/lead.wav \
      --target $ROOT/data/voices/ref_talk_sing_25s.wav \
      --output $d/lead_seedvc_target.wav \
      --f0-condition True --auto-f0-adjust False --semi-tone-shift 0 --fp16 True) \
      >> $d/log_e3_target.log 2>&1
  run_arm $song target $d/lead_seedvc_target.wav
done
echo "ALL E3 ARMS DONE $(date +%H:%M:%S)"
