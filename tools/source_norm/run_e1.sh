#!/bin/bash
# E1: LTAS/EQ 匹配 A/B —— 源 lead 谱包络向目标/中性 LTAS 归一化后再 SVC。
# 三臂 × 两首歌（白色毛衣=高基线, 去有风的地方=低基线），--seed 1 配对。
# 每臂: ltas → sing.py svc,eval → 存 vocal/eval → eval_cer
set -u
ROOT=/home/rookie/voice
GSV=$ROOT/GPT-SoVITS/.venv/bin/python
SING=$ROOT/sing.py
LTAS=$ROOT/tools/source_norm/ltas_match.py
CER=$ROOT/singing/eval_cer.py
REF=$ROOT/data/voices/ref_talk_sing.wav
cd $ROOT

run_arm () {
  local song=$1 arm=$2 srcwav=$3
  local d=$ROOT/data/songs/$song
  local songfile=$ROOT/data/songs/$song.flac
  local log=$d/log_e1_$arm.log
  echo "==== [$song/$arm] start $(date +%H:%M:%S)" | tee -a $log
  local extra=()
  if [ -n "$srcwav" ]; then extra=(--source-wav "$srcwav"); fi
  $GSV $SING --ref $REF --ref-is-clean --song "$songfile" --name $song \
      --stages svc,eval --seed 1 "${extra[@]}" >> $log 2>&1
  cp $d/vocal.wav $d/vocal_e1_$arm.wav
  cp $d/eval.txt $d/eval_e1_$arm.txt
  $GSV $CER --src $d/sep2/lead.wav --gen $d/vocal_e1_$arm.wav --labels $arm \
      --device cuda:0 >> $log 2>&1
  echo "==== [$song/$arm] done $(date +%H:%M:%S)" | tee -a $log
}

for song in 白色毛衣 去有风的地方; do
  d=$ROOT/data/songs/$song
  cp $d/eval.txt $d/eval.txt.pre_e1
  run_arm $song base ""
  $GSV $LTAS --source $d/sep2/lead.wav --target $REF \
      --out $d/lead_ltas_n05.wav --strength 0.5 --json $d/ltas_n05.json
  run_arm $song neutral $d/lead_ltas_n05.wav
  $GSV $LTAS --source $d/sep2/lead.wav --target $REF \
      --out $d/lead_ltas_n10.wav --strength 1.0 --json $d/ltas_n10.json
  run_arm $song target $d/lead_ltas_n10.wav
done
echo "ALL E1 ARMS DONE $(date +%H:%M:%S)"
