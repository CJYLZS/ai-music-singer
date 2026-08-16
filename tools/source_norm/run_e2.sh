#!/bin/bash
# E2: WORLD 共振峰搬移 A/B —— 谱包络按 源→目标 F1 比例弯折（保 F0/节奏）。
# 两臂（strength 0.5/1.0）× 两首歌，--seed 1 配对；基线复用 E1 的 base 臂。
set -u
ROOT=/home/rookie/voice
GSV=$ROOT/GPT-SoVITS/.venv/bin/python
SN=$ROOT/tools/source_norm/.venv/bin/python
SING=$ROOT/sing.py
CER=$ROOT/singing/eval_cer.py
REF=$ROOT/data/voices/ref_talk_sing.wav
cd $ROOT

run_arm () {
  local song=$1 arm=$2 srcwav=$3
  local d=$ROOT/data/songs/$song
  local log=$d/log_e2_$arm.log
  echo "==== [$song/$arm] start $(date +%H:%M:%S)" | tee -a $log
  $GSV $SING --ref $REF --ref-is-clean --song $ROOT/data/songs/$song.flac --name $song \
      --stages svc,eval --seed 1 --source-wav "$srcwav" >> $log 2>&1
  cp $d/vocal.wav $d/vocal_e2_$arm.wav
  cp $d/eval.txt $d/eval_e2_$arm.txt
  $GSV $CER --src $d/sep2/lead.wav --gen $d/vocal_e2_$arm.wav --labels $arm \
      --device cuda:0 >> $log 2>&1
  echo "==== [$song/$arm] done $(date +%H:%M:%S)" | tee -a $log
}

for song in 白色毛衣 去有风的地方; do
  d=$ROOT/data/songs/$song
  $SN $ROOT/tools/source_norm/formant_shift.py --source $d/sep2/lead.wav \
      --target $REF --out $d/lead_fshift_n05.wav --strength 0.5 --json $d/fshift_n05.json
  run_arm $song fshift05 $d/lead_fshift_n05.wav
  $SN $ROOT/tools/source_norm/formant_shift.py --source $d/sep2/lead.wav \
      --target $REF --out $d/lead_fshift_n10.wav --strength 1.0 --json $d/fshift_n10.json
  run_arm $song fshift10 $d/lead_fshift_n10.wav
done
echo "ALL E2 ARMS DONE $(date +%H:%M:%S)"
