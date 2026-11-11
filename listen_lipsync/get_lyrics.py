#!/usr/bin/env python3
"""抓取歌词并缓存成「增强 LRC」，给字幕和口型两条流水线共用。

优先级（先逐字，后行级，都没有就写手编模板）：
    1. 酷狗 KRC —— 匿名可用，自带**逐字**时间戳，且行边界不吞间奏
    2. LRCLIB / 网易云 LRC —— 只有行级时间；字级由 make_cues.py --text 强制对齐补
    3. 写一个带时间轴骨架的模板，人工填词后再跑

产物是歌曲同名 `.lrc`（`data/songs/云烟成雨.flac` → `data/songs/云烟成雨.lrc`），
命中即缓存，**再次运行直接读本地、完全不联网**。`--force` 才会重新抓。

格式用标准「增强 LRC」而不是自造格式：
    [00:18.01]<00:18.01>某<00:18.31>句<00:19.01>歌<00:19.67>词
行首是整行起点，每个 <> 是那个字的起点；没有 <> 就是普通 LRC。好处是任何文本
编辑器都能改、任何播放器都能验、缺字级时能自然降级。

用法:
    python listen_lipsync/get_lyrics.py data/songs/云烟成雨.flac
    python listen_lipsync/get_lyrics.py data/songs/*.flac        # 批量，自动限速
    python listen_lipsync/get_lyrics.py song.flac --force        # 忽略缓存重抓
    python listen_lipsync/get_lyrics.py song.flac --offline      # 只读缓存，永不联网
"""
import argparse
import base64
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zlib

UA = "Mozilla/5.0"
# 酷狗 KRC 的固定异或密钥（krc1 头之后的载荷 XOR 它再 zlib 解压）
KRC_KEY = b"@Gaw^2tGQ61-\xce\xd2ni"

# 元数据行：作词/作曲之类不是歌词，字幕和对齐都不要
META = re.compile(
    r"(作词|作曲|编曲|制作|混音|吉他|监制|出品|录音|母带|贝斯|和声|鼓|弦乐|"
    r"Written|Composed|Produced|Mixing|Mastering|Arrange|OP\b|SP\b)"
)

# 酷狗会静默限流：HTTP 200 + candidates 为空，长得和「这首歌没歌词」一模一样。
# 实测连发 6 次得到 10/10/10/0/10/10，间隔 2s 则 6/6 命中。
PACE = 1.2


def log(*a):
    print(*a, file=sys.stderr)


def http(url, headers=None, timeout=20):
    h = {"User-Agent": UA}
    h.update(headers or {})
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=h), timeout=timeout
    ).read()


def probe(path):
    """从音频文件读 title/artist/duration。标签比联网猜歌手可靠得多。"""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags:format=duration",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    fmt = json.loads(r.stdout or "{}").get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    title = (tags.get("title") or "").strip()
    artist = (tags.get("artist") or tags.get("album_artist") or "").strip()
    dur = float(fmt.get("duration") or 0)
    if not title:  # 没标签就退回文件名
        title = os.path.basename(path).rsplit(".", 1)[0]
    return title, artist, dur


def is_lyric(text):
    t = text.strip()
    if not t:
        return False
    return not (META.search(t) or "：" in t or ":" in t)


# ---------------------------------------------------------------- 酷狗 KRC（逐字）

def kugou_search(keyword, dur_ms, tries=3):
    """搜歌词候选。带重试是必须的 —— 限流失败无任何信号。"""
    for _ in range(tries):
        p = {"ver": 1, "man": "yes", "client": "pc", "keyword": keyword, "hash": ""}
        if dur_ms:
            p["duration"] = dur_ms
        try:
            r = json.loads(http("https://krcs.kugou.com/search?" + urllib.parse.urlencode(p)))
            c = r.get("candidates") or []
            if c:
                return c
        except Exception:
            pass
        time.sleep(PACE * 1.5)
    return []


def kugou_krc(cand):
    """下载并解密一个候选，返回 KRC 文本；失败返回 ''。"""
    url = ("https://lyrics.kugou.com/download?ver=1&client=pc&fmt=krc&charset=utf8"
           f"&accesskey={cand['accesskey']}&id={cand['id']}")
    d = json.loads(http(url))
    raw = base64.b64decode(d.get("content") or "")
    if raw[:4] != b"krc1":
        return ""
    body = bytes(b ^ KRC_KEY[i % len(KRC_KEY)] for i, b in enumerate(raw[4:]))
    return zlib.decompress(body).decode("utf-8", "replace")


def parse_krc(text):
    """KRC → [{start, dur, chars:[(相对起点, 时长, 字)]}]，已剔除元数据行。"""
    out = []
    for line in text.splitlines():
        m = re.match(r"^\[(\d+),(\d+)\](.*)", line)
        if not m:
            continue
        toks = re.findall(r"<(\d+),(\d+),\d+>([^<]*)", m.group(3))
        if not toks:
            continue
        if not is_lyric("".join(c for _, _, c in toks)):
            continue
        out.append({
            "start": int(m.group(1)) / 1000.0,
            "dur": int(m.group(2)) / 1000.0,
            "chars": [(int(a) / 1000.0, int(b) / 1000.0, c) for a, b, c in toks],
        })
    return out


def from_kugou(title, artist, dur):
    """返回逐字行，或 None。候选按时长差排序 —— duration 只是模糊筛选，不能信第一个。"""
    for kw in ([f"{title} {artist}", title] if artist else [title]):
        cands = kugou_search(kw, int(dur * 1000))
        if not cands:
            time.sleep(PACE)
            continue
        cands.sort(key=lambda c: abs(c.get("duration", 0) / 1000.0 - dur))
        for c in cands[:4]:
            time.sleep(PACE)
            try:
                lines = parse_krc(kugou_krc(c))
            except Exception:
                continue
            if len(lines) >= 8:
                log(f"  酷狗 KRC 命中: {len(lines)} 行 / "
                    f"{sum(len(l['chars']) for l in lines)} 字 (kw='{kw}')")
                return lines
        time.sleep(PACE)
    return None


# ------------------------------------------------------------- 行级 LRC（回落）

def parse_lrc(text):
    """普通 LRC → [{start, text}]，一个时间戳带多行文本的情况也展开。"""
    out = []
    for line in text.splitlines():
        stamps = re.findall(r"\[(\d+):(\d+)[.:](\d+)\]", line)
        body = re.sub(r"\[[^\]]*\]", "", line).strip()
        if not stamps or not is_lyric(body):
            continue
        for mm, ss, frac in stamps:
            t = int(mm) * 60 + int(ss) + int(frac) / (1000.0 if len(frac) == 3 else 100.0)
            out.append({"start": t, "text": body})
    out.sort(key=lambda x: x["start"])
    return out


def from_lrclib(title, artist, dur):
    try:
        q = urllib.parse.urlencode({"track_name": title, "artist_name": artist,
                                    "duration": int(dur)})
        j = json.loads(http(f"https://lrclib.net/api/get?{q}",
                            {"User-Agent": "voice-mv/1.0 (+local MV tooling)"}))
        return j.get("syncedLyrics") or ""
    except Exception:
        pass
    try:
        q = urllib.parse.urlencode({"track_name": title, "artist_name": artist})
        r = json.loads(http(f"https://lrclib.net/api/search?{q}",
                            {"User-Agent": "voice-mv/1.0 (+local MV tooling)"}))
        for it in r:
            if abs((it.get("duration") or 0) - dur) <= 3 and it.get("syncedLyrics"):
                return it["syncedLyrics"]
    except Exception:
        pass
    return ""


def from_netease(title, artist, dur):
    ref = {"Referer": "https://music.163.com/"}
    try:
        s = urllib.parse.quote(f"{title} {artist}".strip())
        r = json.loads(http(f"https://music.163.com/api/search/get?s={s}&type=1&limit=10", ref))
        for song in (r.get("result", {}).get("songs") or []):
            if abs(song.get("duration", 0) / 1000.0 - dur) <= 3:
                time.sleep(PACE * 0.5)
                L = json.loads(http("https://music.163.com/api/song/lyric"
                                    f"?id={song['id']}&lv=1&kv=1&tv=-1", ref))
                lrc = (L.get("lrc") or {}).get("lyric") or ""
                if lrc:
                    return lrc
    except Exception:
        pass
    return ""


# --------------------------------------------------------------- 增强 LRC 读写

def parse_yrc(text):
    """网易云 yrc（逐字）→ [{start, dur, chars:[(相对起点, 时长, 字)]}]。

    yrc 一行是 `[行起点ms,行时长ms](字起点ms,字时长ms,0)字...`，**字时间是绝对
    毫秒**（不是 KRC 的相对偏移），这里换算成相对起点，才能对上 write_enhanced。
    开头的 `{"t":0,"c":[...]}` JSON 行（作词/作曲）不以 `[` 开头，自然被跳过。
    """
    out = []
    for line in text.splitlines():
        m = re.match(r"^\[(\d+),(\d+)\](.*)$", line)
        if not m:
            continue
        base = int(m.group(1))
        toks = re.findall(r"\((\d+),(\d+),\d+\)([^(]*)", m.group(3))
        if not toks:
            continue
        if not is_lyric("".join(c for _a, _b, c in toks)):
            continue
        out.append({
            "start": base / 1000.0,
            "dur": int(m.group(2)) / 1000.0,
            "chars": [((int(a) - base) / 1000.0, int(b) / 1000.0, c) for a, b, c in toks],
        })
    return out


def _t2s_same_len(text):
    """繁→简，**长度不变才采用**（逐字时间戳按位置对齐，长度一变就不能换）。

    网易云的数据偶带繁体/异体字（会呼吸的痛「我终於到达」的「於」）。有 opencc
    就顺手转，没装就原样返回 —— 不引入硬依赖。
    """
    try:
        import opencc
    except Exception:
        return text
    conv = opencc.OpenCC("t2s").convert(text)
    return conv if len(conv) == len(text) else text


def from_netease_yrc(title, artist, dur):
    """网易云逐字歌词（yrc）→ 逐字行，或 None。

    比 `from_netease` 的 lv=1 行级 LRC 强在两处：**字级时间戳**（字幕扇光用）与
    **简体的官方文本**；行锚点也是校对过的（会呼吸的痛 首行 18.10s vs 实测起唱
    18.20s，而同曲 lrclib 是 14.43s，早 3.7s）。优先于 lrclib/行级网易云。
    """
    ref = {"Referer": "https://music.163.com/"}
    try:
        s = urllib.parse.quote(f"{title} {artist}".strip())
        r = json.loads(http(f"https://music.163.com/api/search/get?s={s}&type=1&limit=10", ref))
        for song in (r.get("result", {}).get("songs") or []):
            if abs(song.get("duration", 0) / 1000.0 - dur) > 3:
                continue
            time.sleep(PACE * 0.5)
            L = json.loads(http("https://music.163.com/api/song/lyric/v1"
                                f"?id={song['id']}&cp=false&lv=0&kv=0&tv=0&rv=0"
                                f"&yv=1&ytv=1&yrv=1&lrctype=1", ref))
            lines = parse_yrc(((L.get("yrc") or {}).get("lyric") or ""))
            if len(lines) >= 8:
                fixed = 0
                for ln in lines:
                    joined = "".join(c for _o, _d, c in ln["chars"])
                    conv = _t2s_same_len(joined)
                    if conv != joined:
                        ln["chars"] = [(o, d, c2)
                                       for (o, d, _c), c2 in zip(ln["chars"], conv)]
                        fixed += 1
                if fixed:
                    log(f"  繁体字转简: {fixed} 行")
                return lines
    except Exception:
        pass
    return None


def fmt_ts(t):
    t = max(0.0, t)
    return f"{int(t // 60):02d}:{int(t % 60):02d}.{int(round(t % 1 * 100)):02d}"


def write_enhanced(path, lines, meta):
    """写增强 LRC。lines 元素要么有 chars（逐字），要么只有 text（行级）。"""
    buf = [f"[ti:{meta['title']}]", f"[ar:{meta['artist']}]",
           f"[length:{fmt_ts(meta['duration'])}]",
           f"[re:get_lyrics.py]", f"[source:{meta['source']}]", ""]
    for ln in lines:
        if ln.get("chars"):
            body = "".join(f"<{fmt_ts(ln['start'] + off)}>{ch}"
                           for off, _dur, ch in ln["chars"] if ch)
            # 行尾补一个「唱完」时刻，扇光才知道最后一个字何时结束
            body += f"<{fmt_ts(ln['start'] + ln['dur'])}>"
        else:
            body = ln["text"]
        buf.append(f"[{fmt_ts(ln['start'])}]{body}")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(buf) + "\n")
    os.replace(tmp, path)  # 原子写，中途中断不会留半个缓存


def read_enhanced(path):
    """读回增强 LRC → (lines, header)。逐字与行级都能读。"""
    lines, header = [], {}
    for raw in open(path, encoding="utf-8"):
        raw = raw.rstrip("\n")
        m = re.match(r"^\[([a-z]+):(.*)\]\s*$", raw)
        if m and m.group(1) not in ("ti", "ar", "al"):
            header[m.group(1)] = m.group(2)
            continue
        if m and m.group(1) in ("ti", "ar", "al"):
            header[m.group(1)] = m.group(2)
            continue
        m = re.match(r"^\[(\d+):(\d+)[.:](\d+)\](.*)$", raw)
        if not m:
            continue
        start = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100.0
        body = m.group(4)
        toks = re.findall(r"<(\d+):(\d+)[.:](\d+)>([^<]*)", body)
        if toks:
            chars, end = [], start
            for mm, ss, ff, ch in toks:
                t = int(mm) * 60 + int(ss) + int(ff) / 100.0
                if ch:
                    chars.append([t, ch])
                end = max(end, t)
            for i, c in enumerate(chars):
                nxt = chars[i + 1][0] if i + 1 < len(chars) else end
                c.append(max(0.02, nxt - c[0]))
            lines.append({"start": start, "end": end,
                          "chars": [(c[0], c[2], c[1]) for c in chars],
                          "text": "".join(c[1] for c in chars)})
        else:
            txt = re.sub(r"<[^>]*>", "", body).strip()
            if txt:
                lines.append({"start": start, "end": None, "chars": None, "text": txt})
    return lines, header


TEMPLATE_HELP = """\
# 未找到歌词，这是手工填写模板。
#
# 格式：行级只要 [分:秒.厘秒]歌词
#   [00:18.01]这一行的歌词
# 逐字（KTV 扇光更准）在每个字前加 <时间>，行尾再补一个「唱完」时刻：
#   [00:18.01]<00:18.01>这<00:18.35>一<00:18.90>行<00:19.40>
#
# 只填行级也可以：make_cues.py --text 能用强制对齐把字级时间补出来。
# 下面的时间轴是按人声能量自动切出来的乐句骨架，仅供参考，可随意增删。
"""


def vocal_for(path):
    """优先用分离出来的主唱人声切乐句：混音里伴奏一直有能量，门控切不出停顿。

    sing.py 的产物布局是 data/songs/<名>/sep2/lead.wav（两级分离后的主唱）。
    """
    base = os.path.splitext(path)[0]
    for cand in (os.path.join(base, "sep2", "lead.wav"),
                 os.path.join(base, "sep1", "vocal.wav"),
                 os.path.join(base, "vocal.wav")):
        if os.path.exists(cand):
            return cand
    return path


def voiced_phrases(path, dur, gate_db=-40.0, min_sil=0.20, min_len=0.6):
    """按能量切出乐句骨架，给手工填词当脚手架。失败就返回空。"""
    try:
        import numpy as np
        import soundfile as sf
    except Exception:
        return []
    path = vocal_for(path)
    try:
        wav = os.path.join(os.path.dirname(path) or ".", "_lyr_probe.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1",
                        "-ar", "16000", wav], check=True, capture_output=True)
        y, sr = sf.read(wav)
        os.remove(wav)
    except Exception:
        return []
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(1)
    hop = int(sr * 0.05)
    frames = [(i / float(sr), y[i:i + hop]) for i in range(0, len(y) - hop, hop)]
    peak = 20 * np.log10(max(float(abs(y).max()), 1e-9))
    on, spans, start, sil = False, [], 0.0, 0.0
    for t, fr in frames:
        db = 20 * np.log10(max(float((fr.astype("float64") ** 2).mean() ** 0.5), 1e-9)) - peak
        if db > gate_db:
            if not on:
                on, start = True, t
            sil = 0.0
        elif on:
            sil += 0.05
            if sil >= min_sil:
                if t - sil - start >= min_len:
                    spans.append((start, t - sil))
                on = False
    if on and frames and frames[-1][0] - start >= min_len:
        spans.append((start, frames[-1][0]))
    return spans


def write_template(path, meta, spans):
    buf = [TEMPLATE_HELP, f"[ti:{meta['title']}]", f"[ar:{meta['artist']}]",
           f"[length:{fmt_ts(meta['duration'])}]", "[source:TEMPLATE 待人工填写]", ""]
    if spans:
        for a, _b in spans:
            buf.append(f"[{fmt_ts(a)}]")
    else:
        buf.append("[00:00.00]")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(buf) + "\n")
    os.replace(tmp, path)


def stats(lines):
    n = len(lines)
    nch = sum(len(l["chars"]) for l in lines if l.get("chars"))
    kind = "逐字" if nch else "行级"
    span = f"{lines[0]['start']:.1f}s..{max(l.get('end') or l['start'] for l in lines):.1f}s" if n else "-"
    return f"{kind} {n} 行" + (f" / {nch} 字" if nch else "") + f"，覆盖 {span}"


def process(path, args):
    title, artist, dur = probe(path)
    out = os.path.splitext(path)[0] + ".lrc"
    log(f"\n{os.path.basename(path)}  [{title} / {artist or '?'}]  {dur:.1f}s")

    if os.path.exists(out) and not args.force:
        lines, hdr = read_enhanced(out)
        if lines:
            log(f"  缓存命中 {out} —— {stats(lines)}"
                f"{'  (源: ' + hdr['source'] + ')' if hdr.get('source') else ''}")
            return out
        log(f"  缓存存在但没有可用歌词行（待填模板）: {out}")
        return None
    if args.offline:
        log("  --offline 且无缓存，跳过")
        return None

    meta = {"title": title, "artist": artist, "duration": dur, "source": ""}

    lines = from_kugou(title, artist, dur)
    if lines:
        meta["source"] = "kugou-krc (逐字)"
        write_enhanced(out, lines, meta)
        log(f"  写入 {out} —— {stats(read_enhanced(out)[0])}")
        return out

    # 酷狗挂了（实测 2026-09 出口在台湾时报 HTTP 200 + 空候选，连热门歌也是）时，
    # 网易云 yrc 是同样带字级时间戳的来源，优先于下面两个行级源。
    lines = from_netease_yrc(title, artist, dur)
    if lines:
        log(f"  网易云 yrc 逐字命中: {len(lines)} 行 / "
            f"{sum(len(l['chars']) for l in lines)} 字")
        meta["source"] = "netease-yrc (逐字)"
        write_enhanced(out, lines, meta)
        log(f"  写入 {out} —— {stats(read_enhanced(out)[0])}")
        return out

    for name, fn in (("lrclib", from_lrclib), ("netease", from_netease)):
        time.sleep(PACE * 0.5)
        raw = fn(title, artist, dur)
        got = parse_lrc(raw) if raw else []
        if len(got) >= 8:
            log(f"  {name} 行级命中: {len(got)} 行（字级留给 make_cues.py --text 补）")
            meta["source"] = f"{name}-lrc (行级)"
            write_enhanced(out, got, meta)
            log(f"  写入 {out} —— {stats(read_enhanced(out)[0])}")
            return out

    log("  三个源都没有 —— 写手工填写模板")
    meta["source"] = "TEMPLATE"
    write_template(out, meta, voiced_phrases(path, dur))
    log(f"  写入 {out}（请填词后再跑字幕）")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="抓取/缓存歌词为增强 LRC（歌曲同名 .lrc）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("songs", nargs="+", help="音频文件（.flac/.wav/.mp3），支持通配")
    ap.add_argument("--force", action="store_true", help="忽略缓存，重新联网抓取")
    ap.add_argument("--offline", action="store_true", help="只读缓存，绝不联网")
    args = ap.parse_args()

    paths = []
    for s in args.songs:
        paths.extend(sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s])
    ok = 0
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            log(f"跳过（不存在）: {p}")
            continue
        try:
            if process(p, args):
                ok += 1
        except Exception as e:
            log(f"  失败: {type(e).__name__}: {e}")
        if i + 1 < len(paths):
            time.sleep(PACE)
    log(f"\n完成 {ok}/{len(paths)}")


if __name__ == "__main__":
    main()
