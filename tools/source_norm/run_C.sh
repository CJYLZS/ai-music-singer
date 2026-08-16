#!/bin/bash
# C 组: cfg×α 2×2 宽验证 —— kNN α=0.2 在 cfg=0.5 下是否净负？
# 4 首歌 × {cfg 0.5, 1.0} × {α 0, 0.2}，--seed 1 配对，无预转换（纯源）。
set -u
ROOT=/home/rookie/voice
GSV=$ROOT/GPT-SoVITS/.venv/bin/python
SING=$ROOT/sing.py
CER=$ROOT/singing/eval_cer.py
REF=$ROOT/data/voices/ref_talk_sing.wav
cd $ROOT

run_arm () {  # song arm cfg alpha
  local song=$1 arm=$2 cfg=$3 alpha=$4
  local d=$ROOT/data/songs/$song
  local log=$d/log_C_$arm.log
  echo "==== [$song/$arm cfg=$cfg α=$alpha] start $(date +%H:%M:%S)" | tee -a $log
  $GSV $SING --ref $REF --ref-is-clean --song $ROOT/data/songs/$song.flac --name $song \
      --stages svc,eval --seed 1 --cfg $cfg --knn-alpha $alpha >> $log 2>&1
  cp $d/vocal.wav $d/vocal_C_$arm.wav
  cp $d/eval.txt $d/eval_C_$arm.txt
  $GSV $CER --src $d/sep2/lead.wav --gen $d/vocal_C_$arm.wav --labels $arm \
      --device cuda:0 >> $log 2>&1
  echo "==== [$song/$arm] done $(date +%H:%M:%S)" | tee -a $log
}

for song in 白色毛衣 去有风的地方 冬眠 绿色; do
  run_arm $song c05a02 0.5 0.2
  run_arm $song c05a00 0.5 0.0
  run_arm $song c10a02 1.0 0.2
  run_arm $song c10a00 1.0 0.0
done
echo "ALL C-GROUP ARMS DONE $(date +%H:%M:%S)"
