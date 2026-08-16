#!/bin/bash
# E1b: 修正版 LTAS 匹配（纯形状 + 限带 150-8000Hz）A/B。strength 0.5/1.0 两臂。
# 基线复用 E1 base。等 E2/E3 跑完后自动开始（避免 GPU 争抢）。
set -u
ROOT=/home/rookie/voice
GSV=$ROOT/GPT-SoVITS/.venv/bin/python
SING=$ROOT/sing.py
CER=$ROOT/singing/eval_cer.py
REF=$ROOT/data/voices/ref_talk_sing.wav
cd $ROOT

# 等 E3 结束（E3 的驱动脚本末尾写 E2E3 EXIT= 标记）
while ! grep -q "E2E3 EXIT=" .tmp/e3_run.log 2>/dev/null; do sleep 20; done
echo "E2/E3 已结束，E1b 开始 $(date +%H:%M:%S)"

run_arm () {
  local song=$1 arm=$2 srcwav=$3
  local d=$ROOT/data/songs/$song
  local log=$d/log_e1b_$arm.log
  echo "==== [$song/$arm] start $(date +%H:%M:%S)" | tee -a $log
  $GSV $SING --ref $REF --ref-is-clean --song $ROOT/data/songs/$song.flac --name $song \
      --stages svc,eval --seed 1 --source-wav "$srcwav" >> $log 2>&1
  cp $d/vocal.wav $d/vocal_e1b_$arm.wav
  cp $d/eval.txt $d/eval_e1b_$arm.txt
  $GSV $CER --src $d/sep2/lead.wav --gen $d/vocal_e1b_$arm.wav --labels $arm \
      --device cuda:0 >> $log 2>&1
  echo "==== [$song/$arm] done $(date +%H:%M:%S)" | tee -a $log
}

for song in 白色毛衣 去有风的地方; do
  d=$ROOT/data/songs/$song
  $GSV $ROOT/tools/source_norm/ltas_match.py --source $d/sep2/lead.wav --target $REF \
      --out $d/lead_ltasshape_n05.wav --strength 0.5 --json $d/ltasshape_n05.json
  run_arm $song shape05 $d/lead_ltasshape_n05.wav
  $GSV $ROOT/tools/source_norm/ltas_match.py --source $d/sep2/lead.wav --target $REF \
      --out $d/lead_ltasshape_n10.wav --strength 1.0 --json $d/ltasshape_n10.json
  run_arm $song shape10 $d/lead_ltasshape_n10.wav
done
echo "ALL E1B ARMS DONE $(date +%H:%M:%S)"
