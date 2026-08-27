#!/bin/bash
# 全量重跑：14 首歌 segprompt svc + eval，固定 seed 1234（2026-08-26 定稿）
# 最终参数（已写入代码默认）：seg-min 12 / seg-max 19 / svc-max 22 / min-run 0.2
# - svc22：22s 窗口内部切块无能量塌缩（实测 0/14 衰减，SIM 三档最高）
# - seg-max 19：扩展后 23s 超上限，shrink 只裁尾部到 22s，前缀 2s 保留、后缀 1s
# - min-run 0.2：±2s 冗余缓冲允许短停顿切，配合前奏/尾奏边界修复全部可切
# - 不做响度后处理：段内趋势已与 lead 一致，绝对响度交给 mix 的 --match-lead
# 用法：bash rerun_all.sh  （在仓库根执行）
set -u
cd "$(dirname "$0")"
PY=GPT-SoVITS/.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SONGS="云烟成雨 亲亲 冬眠 去有风的地方 情歌 最后一页 爱的回归线 爱笑的眼睛 白色毛衣 绿色 追光者 那女孩对我说 陪你看星星 雨爱"
for s in $SONGS; do
  echo "===== [$s] $(date +%F\ %T) ====="
  $PY sing.py --ref data/voices/ref_talk_sing.wav --song "data/songs/$s.flac" --name "$s" \
    --stages svc,eval --seed 1234 --gpu 0 2>&1 | tail -5
  echo "===== [$s] done $(date +%T) exit=$? ====="
done
echo "ALL DONE"
