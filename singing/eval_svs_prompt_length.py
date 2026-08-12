#!/usr/bin/env python
"""评测不同 prompt 长度的 SVS 输出质量

指标:
1. 音色相似度 (vs prompt) - ERes2NetV2
2. 音色相似度 (vs 真实语音) - ERes2NetV2  
3. 音高准确度 - FFE/GPE/VDE (vs MIDI 转录的 note_pitch)
4. 吐字清晰度 - CER (vs 歌词)
5. 稳定性 - drift/dropouts

用法:
    GPT-SoVITS/.venv/bin/python singing/eval_svs_prompt_length.py
"""
import argparse
import os
import sys
import subprocess
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def run_eval_sim(ref_wav, gen_wav, f0=None, ref_f0=None):
    """运行 eval_sim.py 返回 mean/median SIM"""
    cmd = [
        "GPT-SoVITS/.venv/bin/python",
        "singing/eval_sim.py",
        "--ref", ref_wav,
        "--gen", gen_wav,
    ]
    if f0:
        cmd += ["--f0", f0]
    if ref_f0:
        cmd += ["--ref-f0", ref_f0]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    out = result.stdout
    
    # 解析输出
    mean_sim = median_sim = 0.0
    for line in out.split('\n'):
        if 'mean SIM' in line:
            mean_sim = float(line.split(':')[-1].strip())
        elif 'median SIM' in line:
            median_sim = float(line.split(':')[-1].strip())
    
    return mean_sim, median_sim


def run_eval_pitch(ref_meta, gen_wav):
    """运行 eval_pitch.py 返回 FFE/GPE/VDE"""
    cmd = [
        "GPT-SoVITS/.venv/bin/python",
        "singing/eval_pitch.py",
        "--ref-meta", ref_meta,
        "--gen", gen_wav,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    out = result.stdout
    
    # 解析输出
    ffe = gpe = vde = 0.0
    for line in out.split('\n'):
        if 'FFE' in line and 'cents' in line:
            ffe = float(line.split(':')[-1].strip().split()[0])
        elif 'GPE' in line and '%' in line:
            gpe = float(line.split(':')[-1].strip().replace('%', ''))
        elif 'VDE' in line and '%' in line:
            vde = float(line.split(':')[-1].strip().replace('%', ''))
    
    return ffe, gpe, vde


def run_eval_cer(src_wav, gen_wav):
    """运行 eval_cer.py 返回 CER"""
    cmd = [
        "GPT-SoVITS/.venv/bin/python",
        "singing/eval_cer.py",
        "--src", src_wav,
        "--gen", gen_wav,
        "--win", "10",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    out = result.stdout
    
    # 解析输出
    cer = 0.0
    for line in out.split('\n'):
        if 'CER' in line and not 'ΔCER' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'CER' and i+1 < len(parts):
                    try:
                        cer = float(parts[i+1])
                        break
                    except:
                        pass
    
    return cer


def run_eval_stab(ref_wav, gen_wav):
    """运行 eval_stab.py 返回 drift/dropouts/worst"""
    cmd = [
        "GPT-SoVITS/.venv/bin/python",
        "singing/eval_stab.py",
        "--ref", ref_wav,
        "--gen", gen_wav,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    out = result.stdout
    
    # 解析输出
    drift = dropouts = worst = 0.0
    for line in out.split('\n'):
        if 'drift' in line.lower() and 'p90' not in line.lower():
            parts = line.split()
            for i, p in enumerate(parts):
                if 'drift' in p.lower() and i+1 < len(parts):
                    try:
                        drift = float(parts[i+1])
                        break
                    except:
                        pass
        elif 'dropouts' in line.lower():
            parts = line.split()
            for i, p in enumerate(parts):
                if 'dropouts' in p.lower() and i+1 < len(parts):
                    try:
                        dropouts = int(parts[i+1].split('/')[0])
                        break
                    except:
                        pass
        elif 'worst' in line.lower():
            parts = line.split()
            for i, p in enumerate(parts):
                if 'worst' in p.lower() and i+1 < len(parts):
                    try:
                        worst = float(parts[i+1])
                        break
                    except:
                        pass
    
    return drift, dropouts, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="绿色")
    ap.add_argument("--lengths", type=int, nargs='+', default=[10, 15, 20, 30, 45, 60])
    args = ap.parse_args()
    
    song_base = f"data/songs/{args.song}"
    ref_vocal = f"{song_base}/refsep2/lead.wav"  # 真实语音
    src_vocal = f"{song_base}/sep2/lead.wav"     # 原唱（用于 CER）
    
    print("=" * 80)
    print("SVS Prompt 长度影响评测")
    print("=" * 80)
    print(f"歌曲: {args.song}")
    print(f"测试长度: {args.lengths}")
    print()
    
    results = []
    
    for dur in args.lengths:
        exp_name = f"{args.song}_svs_{dur}s"
        svs_out = f"data/songs/{exp_name}/svs/svs_vocal.wav"
        prompt_wav = f"data/voices/ref_speech_{dur}s.wav"
        
        if not os.path.exists(svs_out):
            print(f"⚠ 跳过 {dur}s: 未找到 {svs_out}")
            continue
        
        print(f"评测 {dur}s prompt...")
        
        # 1. 音色 vs prompt
        sim_vs_prompt_mean, sim_vs_prompt_median = run_eval_sim(
            prompt_wav, svs_out
        )
        
        # 2. 音色 vs 真实语音
        sim_vs_real_mean, sim_vs_real_median = run_eval_sim(
            ref_vocal, svs_out
        )
        
        # 3. 音高准确度（需要 metadata）
        meta_path = f"data/songs/{exp_name}/svs_prep/metadata.json"
        if os.path.exists(meta_path):
            ffe, gpe, vde = run_eval_pitch(meta_path, svs_out)
        else:
            ffe = gpe = vde = -1
            print(f"  警告: 未找到 {meta_path}，跳过音高评测")
        
        # 4. 吐字清晰度
        cer = run_eval_cer(src_vocal, svs_out)
        
        # 5. 稳定性
        drift, dropouts, worst = run_eval_stab(ref_vocal, svs_out)
        
        result = {
            'duration': dur,
            'sim_vs_prompt_mean': sim_vs_prompt_mean,
            'sim_vs_prompt_median': sim_vs_prompt_median,
            'sim_vs_real_mean': sim_vs_real_mean,
            'sim_vs_real_median': sim_vs_real_median,
            'ffe': ffe,
            'gpe': gpe,
            'vde': vde,
            'cer': cer,
            'drift': drift,
            'dropouts': dropouts,
            'worst': worst,
        }
        results.append(result)
        
        print(f"  ✓ {dur}s: SIM(prompt)={sim_vs_prompt_mean:.4f}, "
              f"SIM(real)={sim_vs_real_mean:.4f}, "
              f"FFE={ffe:.1f}, CER={cer:.3f}, drift={drift:.4f}")
        print()
    
    # 汇总报告
    print("=" * 80)
    print("汇总报告")
    print("=" * 80)
    print()
    
    print("| Prompt | SIM(prompt) | SIM(real) | FFE(cents) | GPE% | VDE% | CER | Drift | Dropouts |")
    print("|--------|-------------|-----------|------------|------|------|-----|-------|----------|")
    
    for r in results:
        print(f"| {r['duration']:3d}s   | "
              f"{r['sim_vs_prompt_mean']:.4f}      | "
              f"{r['sim_vs_real_mean']:.4f}    | "
              f"{r['ffe']:6.1f}     | "
              f"{r['gpe']:4.1f} | "
              f"{r['vde']:4.1f} | "
              f"{r['cer']:.3f} | "
              f"{r['drift']:.4f} | "
              f"{r['dropouts']:2d}       |")
    
    print()
    print("=" * 80)
    print("分析建议:")
    print("=" * 80)
    
    # 找最优配置
    if results:
        best_sim_real = max(results, key=lambda x: x['sim_vs_real_mean'])
        best_ffe = min(results, key=lambda x: x['ffe'] if x['ffe'] > 0 else 999)
        best_cer = min(results, key=lambda x: x['cer'])
        best_drift = min(results, key=lambda x: x['drift'])
        
        print(f"- 最佳音色(vs real): {best_sim_real['duration']}s "
              f"(SIM={best_sim_real['sim_vs_real_mean']:.4f})")
        print(f"- 最佳音准: {best_ffe['duration']}s (FFE={best_ffe['ffe']:.1f} cents)")
        print(f"- 最佳吐字: {best_cer['duration']}s (CER={best_cer['cer']:.3f})")
        print(f"- 最稳定: {best_drift['duration']}s (drift={best_drift['drift']:.4f})")
    
    # 保存 JSON
    out_json = f"data/songs/{args.song}_svs_prompt_length.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n已保存详细数据: {out_json}")


if __name__ == "__main__":
    main()
