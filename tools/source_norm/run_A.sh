#!/bin/bash
# A 组实验: E6 迭代自转换 / E7 预转换参考=prompt_ext / E8 kNN×预转换 / E9 预转换迭代
# 全部 --seed 1 配对; 基线复用 E1 base (α=0.2, 无预转换)。
set -u
ROOT=/home/rookie/voice
GSV=$ROOT/GPT-SoVITS/.venv/bin/python
SING=$ROOT/sing.py
CER=$ROOT/singing/eval_cer.py
REF=$ROOT/data/voices/ref_talk_sing.wav
SEEDVC=$ROOT/seed-vc/.venv/bin/python
SEEDVC_INF=$ROOT/seed-vc/inference.py
cd $ROOT

run_arm () {
  local song=$1 arm=$2 srcwav=$3 alpha=$4
  local d=$ROOT/data/songs/$song
  local log=$d/log_A_$arm.log
  echo "==== [$song/$arm] start $(date +%H:%M:%S)" | tee -a $log
  local extra=()
  if [ -n "$srcwav" ]; then extra+=(--source-wav "$srcwav"); fi
  $GSV $SING --ref $REF --ref-is-clean --song $ROOT/data/songs/$song.flac --name $song \
      --stages svc,eval --seed 1 --knn-alpha $alpha "${extra[@]}" >> $log 2>&1
  cp $d/vocal.wav $d/vocal_A_$arm.wav
  cp $d/eval.txt $d/eval_A_$arm.txt
  $GSV $CER --src $d/sep2/lead.wav --gen $d/vocal_A_$arm.wav --labels $arm \
      --device cuda:0 >> $log 2>&1
  echo "==== [$song/$arm] done $(date +%H:%M:%S)" | tee -a $log
}

seedvc_conv () {  # song src out tgtref
  local song=$1 src=$2 out=$3 tgtref=$4
  local d=$ROOT/data/songs/$song
  (cd $ROOT/seed-vc && $SEEDVC $SEEDVC_INF --source "$src" --target "$tgtref" \
      --output "$d/$(basename $out).dir" \
      --f0-condition True --auto-f0-adjust False --semi-tone-shift 0 --fp16 True) \
      >> $d/log_A_seedvc.log 2>&1
  local f=$(ls $d/$(basename $out).dir/*.wav 2>/dev/null | head -1)
  if [ -z "$f" ]; then echo "[$song] seed-vc FAILED for $src"; exit 1; fi
  mv "$f" "$out" && rmdir $d/$(basename $out).dir
  echo "[$song] seed-vc $src -> $out"
}

for song in 白色毛衣 去有风的地方; do
  d=$ROOT/data/songs/$song
  echo "== $song"
  # --- E7: 预转换参考 = prompt_ext.wav（下游音色锚点）---
  seedvc_conv $song $d/sep2/lead.wav $d/lead_prevc_prompt.wav $d/prompt_ext.wav
  run_arm $song e7 $d/lead_prevc_prompt.wav 0.2
  # --- E6: 迭代自转换（pass2 源 = E1 base 输出）---
  run_arm $song e6p2 $d/vocal_e1_base.wav 0.2
  # --- E8: kNN × 预转换 ---
  run_arm $song e8a0 "" 0.0                 # kNN 关, 无预转换
  run_arm $song e8a0on $d/lead_prevc_prompt.wav 0.0   # kNN 关 + 预转换
  run_arm $song e8a1on $d/lead_prevc_prompt.wav 0.1   # kNN 0.1 + 预转换
done
# --- E6 加一轮（白色毛衣 pass3，看收敛）---
run_arm 白色毛衣 e6p3 $ROOT/data/songs/白色毛衣/vocal_A_e6p2.wav 0.2
# --- E9: 预转换迭代（去有风的地方，两遍 seed-vc -> prompt_ext）---
d=$ROOT/data/songs/去有风的地方
seedvc_conv 去有风的地方 $d/lead_prevc_prompt.wav $d/lead_prevc2_p2.wav $d/prompt_ext.wav
run_arm 去有风的地方 e9 $d/lead_prevc2_p2.wav 0.2
echo "ALL A-GROUP ARMS DONE $(date +%H:%M:%S)"
