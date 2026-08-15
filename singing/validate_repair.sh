#!/bin/bash
# 修复后验证：漏音扫描(双模型×1s/3s) + eval_sim + CER + 区域前后对比
set -e
D=data/songs/雨爱
SV=seed-vc/.venv/bin/python
GV=GPT-SoVITS/.venv/bin/python
IMPS="--impostor data/songs/冬眠/sep2/lead.wav data/songs/绿色/sep2/lead.wav data/songs/爱笑的眼睛/sep2/lead.wav data/songs/陪你看星星/sep2/lead.wav data/songs/最后一页/sep2/lead.wav"
REF="--ref data/voices/ref_talk_sing.wav"
F0="--f0 $D/target_f0.npy"

echo "===== 1. 漏音扫描：repaired（双模型 × 1s/3s）====="
$SV singing/eval_leak_windows.py --song-dir $D $REF $IMPS $F0 --win 1.0 --csv-out $D/leak_windows_1s_repaired.csv --out $D/vocal_repaired.wav | head -12
$SV singing/eval_leak_windows.py --song-dir $D $REF $IMPS $F0 --win 3.0 --csv-out $D/leak_windows_3s_repaired.csv --out $D/vocal_repaired.wav | head -12
$SV singing/eval_leak_campplus.py --song-dir $D $REF $IMPS $F0 --win 1.0 --csv-out $D/leak_windows_1s_campplus_repaired.csv --out $D/vocal_repaired.wav | head -12
$SV singing/eval_leak_campplus.py --song-dir $D $REF $IMPS $F0 --win 3.0 --csv-out $D/leak_windows_3s_campplus_repaired.csv --out $D/vocal_repaired.wav | head -12

echo "===== 2. eval_sim：repaired vs 参考 ====="
$GV eval_sim.py --ref data/voices/ref_talk_sing.wav --gen $D/vocal_repaired.wav --f0 $D/target_f0.npy --ref-f0 $D/prompt_f0.npy 2>&1 | tail -6

echo "===== 3. CER：orig vs repaired（FunASR，种子固定）====="
$GV singing/eval_cer.py --src $D/sep2/lead.wav --gen $D/vocal.wav $D/vocal_repaired.wav --labels orig repaired --device cuda:0 2>&1 | tail -8

echo "===== 4. 区域前后对比（excess，1s 扫描，双模型）====="
$SV - <<'EOF'
import csv, numpy as np
def load(p):
    d={}
    with open(p) as f:
        for r in csv.DictReader(f):
            t=float(r["t0"])
            if r["excess"] not in ("","nan"): d[t]=float(r["excess"])
    return d
pairs=[("data/songs/雨爱/leak_windows_1s.csv","data/songs/雨爱/leak_windows_1s_repaired.csv","eres2net"),
       ("data/songs/雨爱/leak_windows_1s_campplus.csv","data/songs/雨爱/leak_windows_1s_campplus_repaired.csv","campplus")]
regs=[(73,84),(150,162),(186,198),(210,225)]
print(f"{'region':<12}{'model':<10}{'before':>9}{'after':>9}{'delta':>9}")
for a,b in regs:
    for old,new,m in pairs:
        o=load(old); n=load(new)
        ov=[v for t,v in o.items() if a<=t<b]; nv=[v for t,v in n.items() if a<=t<b]
        if not ov: continue
        bo,an=np.mean(ov),np.mean(nv)
        print(f"[{a}-{b}s]   {m:<10}{bo:>9.4f}{an:>9.4f}{an-bo:>+9.4f}")
EOF

echo "===== 5. 听感片段（repaired）====="
$SV singing/leak_agree.py data/songs/雨爱/leak_windows_1s_repaired.csv data/songs/雨爱/leak_windows_1s_campplus_repaired.csv \
    --names eres2net campplus --out data/songs/雨爱/vocal_repaired.wav --target data/songs/雨爱/sep2/lead.wav \
    --clips-dir data/songs/雨爱/leak_clips_repaired --top 12 --neigh 2 2>&1 | tail -4
