#!/bin/bash
# B 组: E5 YingMusic-SVC 预转换（唱歌域训练）→ SoulX。对照 A 组 e8a0on（seed-vc 预转换）。
# 参考 = prompt_ext.wav（下游音色锚点，E7 教训）; --semi-tone-shift 0 必须（保留原旋律）。
set -u
ROOT=/home/rookie/voice
GSV=$ROOT/GPT-SoVITS/.venv/bin/python
SING=$ROOT/sing.py
CER=$ROOT/singing/eval_cer.py
REF=$ROOT/data/voices/ref_talk_sing.wav
YM=$ROOT/tools/yingmusic/.venv/bin/python
cd $ROOT

ym_conv () {  # song
  local song=$1
  local d=$ROOT/data/songs/$song
  (cd $ROOT/tools/yingmusic && $YM my_inference.py \
      --source $d/sep2/lead.wav --target $d/prompt_ext.wav \
      --diffusion-steps 100 --checkpoint checkpoints/YingMusic-SVC-full.pt \
      --expname e5_$song --cuda 0 --fp16 True --config ./configs/YingMusic-SVC.yml \
      --semi-tone-shift 0) > $d/log_B_ym.log 2>&1
  local f=$(ls $ROOT/tools/yingmusic/outputs/e5_$song/*.wav 2>/dev/null | head -1)
  if [ -z "$f" ]; then echo "[$song] YingMusic FAILED"; tail -5 $d/log_B_ym.log; exit 1; fi
  mv "$f" $d/lead_prevc_ym.wav
  echo "[$song] YingMusic 预转换完成: $(stat -c%s $d/lead_prevc_ym.wav) bytes"
}

run_arm () {  # song arm srcwav alpha
  local song=$1 arm=$2 srcwav=$3 alpha=$4
  local d=$ROOT/data/songs/$song
  local log=$d/log_B_$arm.log
  echo "==== [$song/$arm] start $(date +%H:%M:%S)" | tee -a $log
  local extra=()
  if [ -n "$srcwav" ]; then extra+=(--source-wav "$srcwav"); fi
  $GSV $SING --ref $REF --ref-is-clean --song $ROOT/data/songs/$song.flac --name $song \
      --stages svc,eval --seed 1 --knn-alpha $alpha "${extra[@]}" >> $log 2>&1
  cp $d/vocal.wav $d/vocal_B_$arm.wav
  cp $d/eval.txt $d/eval_B_$arm.txt
  $GSV $CER --src $d/sep2/lead.wav --gen $d/vocal_B_$arm.wav --labels $arm \
      --device cuda:0 >> $log 2>&1
  echo "==== [$song/$arm] done $(date +%H:%M:%S)" | tee -a $log
}

for song in 白色毛衣 去有风的地方; do
  ym_conv $song
  run_arm $song ym0 $ROOT/data/songs/$song/lead_prevc_ym.wav 0.0
  run_arm $song ym02 $ROOT/data/songs/$song/lead_prevc_ym.wav 0.2
  # 归因: 预转换源自身的 CER（对照 seed-vc 的 0.139/0.274）
  $GSV $CER --src $ROOT/data/songs/$song/sep2/lead.wav \
      --gen $ROOT/data/songs/$song/lead_prevc_ym.wav --labels ym_prevc \
      --device cuda:0 >> $ROOT/data/songs/$song/log_B_ym.log 2>&1
done
echo "ALL B-GROUP ARMS DONE $(date +%H:%M:%S)"
