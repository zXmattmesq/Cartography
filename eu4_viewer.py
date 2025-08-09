#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eu4_viewer.py — low-RAM, fast renderer
- Reads compressed .eu4 (gamestate) or plaintext
- Builds CSVs (overview/economy/military/development/technology/legitimacy)
- Renders world map with a 24-bit RGB→color LUT (fast & ~120MB total RAM)
"""

import argparse, csv, re, zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# -------- CLI --------
def parse_args():
    ap = argparse.ArgumentParser(description="EU4 save → CSVs + world map")
    ap.add_argument("save", help=".eu4 save (compressed or plaintext)")
    ap.add_argument("--assets", default=".", help="Dir with provinces.bmp, definition.csv, default.map, 00_country_colors.txt")
    ap.add_argument("--out", default="world_map.png", help="Output PNG")
    return ap.parse_args()

ASSET = {
    "provinces_bmp": "provinces.bmp",
    "definition_csv": "definition.csv",
    "default_map": "default.map",
    "colors_txt": "00_country_colors.txt",
    "country_names": "country_names.csv",  # optional
}

DEEP_OCEAN  = (24, 66, 140)
COASTAL_SEA = (54, 149, 255)
UNOCCUPIED  = (160, 160, 160)

# -------- Paradox text helpers --------
_comment = re.compile(r"(?m)^\s*#.*$")
def strip_comments(s: str) -> str: return _comment.sub("", s)

def find_block(s: str, key: str) -> Tuple[int, int]:
    pat = re.compile(rf"\b{re.escape(key)}\s*=\s*{{", re.M)
    m = pat.search(s)
    if not m: return (-1, -1)
    i = m.end() - 1; depth = 0
    for j in range(i, len(s)):
        ch = s[j]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: return (m.start(), j + 1)
    return (-1, -1)

def tokenize_kv(s: str):
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i].isspace(): i += 1
        if i >= n or s[i] == "}": break
        ks = i
        while i < n and not s[i].isspace() and s[i] not in "={}": i += 1
        key = s[ks:i]
        while i < n and s[i].isspace(): i += 1
        if i >= n or s[i] != "=":
            while i < n and s[i] not in "{}\n": i += 1
            continue
        i += 1
        while i < n and s[i].isspace(): i += 1
        if i < n and s[i] == "{":
            depth = 0; vs = i
            while i < n:
                if s[i] == "{": depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1; yield key, s[vs:i]; break
                i += 1
        elif i < n and s[i] == '"':
            i += 1; vs = i
            while i < n and s[i] != '"':
                if s[i] == "\\" and i + 1 < n: i += 2
                else: i += 1
            val = s[vs:i]; i += 1; yield key, val
        else:
            vs = i
            while i < n and not s[i].isspace() and s[i] not in "{}": i += 1
            yield key, s[vs:i]

def parse_scalar(v: str):
    v = v.strip()
    if v.startswith("{") and v.endswith("}"): return v
    if v.startswith('"') and v.endswith('"'): return v[1:-1]
    if v in ("yes","no"): return v == "yes"
    try:
        if "." in v or "e" in v.lower(): return float(v)
        return int(v)
    except Exception:
        return v

def parse_block_to_dict(s: str) -> dict:
    body = s[1:-1] if s.startswith("{") and s.endswith("}") else s
    out = {}
    for k, v in tokenize_kv(body): out[k] = parse_scalar(v)
    return out

# -------- Save reading --------
def read_save_text(path: Path) -> str:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            with z.open("gamestate") as f:
                return f.read().decode("latin-1", "ignore")
    return path.read_text(encoding="latin-1", errors="ignore")

# -------- Minimal parse (countries + provinces we need) --------
def parse_save_minimal(save_text: str) -> dict:
    txt = strip_comments(save_text)
    countries = {}
    s, e = find_block(txt, "countries")
    if s != -1:
        inner = txt[s:e]; inner = inner[inner.find("{")+1: inner.rfind("}")]
        tag_pat = re.compile(r"\b([A-Z0-9]{3})\s*=\s*{", re.M)
        idx = 0
        while True:
            m = tag_pat.search(inner, idx)
            if not m: break
            tag = m.group(1); i = m.end()-1; depth = 0
            for j in range(i, len(inner)):
                if inner[j] == "{": depth += 1
                elif inner[j] == "}":
                    depth -= 1
                    if depth == 0:
                        countries[tag] = parse_block_to_dict(inner[i:j+1]); idx = j+1; break
            else: break

    provinces = {}
    s, e = find_block(txt, "provinces")
    if s != -1:
        inner = txt[s:e]; inner = inner[inner.find("{")+1: inner.rfind("}")]
        id_pat = re.compile(r"\b(\d+)\s*=\s*{", re.M)
        idx = 0
        while True:
            m = id_pat.search(inner, idx)
            if not m: break
            pid = int(m.group(1)); i = m.end()-1; depth = 0
            for j in range(i, len(inner)):
                if inner[j] == "{": depth += 1
                elif inner[j] == "}":
                    depth -= 1
                    if depth == 0:
                        d = parse_block_to_dict(inner[i:j+1])
                        provinces[pid] = {
                            "owner": d.get("owner"),
                            "is_city": True if d.get("is_city") is True else False,
                            "colony": True if d.get("colony") is True else False,
                            "base_tax": float(d.get("base_tax", 0) or 0),
                            "base_production": float(d.get("base_production", 0) or 0),
                            "base_manpower": float(d.get("base_manpower", 0) or 0),
                            "is_sea": True if (d.get("is_sea") is True or d.get("sea_zone") is True) else False,
                        }
                        idx = j+1; break
            else: break

    return {"countries": countries, "provinces": provinces}

# -------- CSVs --------
def num(d, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            try: return float(v)
            except Exception: pass
    return default

def dev_of_province(d: dict) -> float:
    return float(d.get("base_tax",0)) + float(d.get("base_production",0)) + float(d.get("base_manpower",0))

def aggregate_development(parsed: dict) -> dict:
    out = defaultdict(lambda: {"province_count":0, "total_development":0.0})
    for pid, p in parsed["provinces"].items():
        if p.get("is_sea"): continue
        tag = p.get("owner")
        if not tag: continue
        out[tag]["province_count"] += 1
        out[tag]["total_development"] += dev_of_province(p)
    for tag, d in out.items():
        pc = d["province_count"] or 1
        d["avg_development"] = d["total_development"]/pc
    return out

def write_csv(path: Path, fields: List[str], rows: List[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)

def build_rows(parsed: dict, tag_to_name: Dict[str,str]) -> Dict[str, List[dict]]:
    C, D = parsed["countries"], aggregate_development(parsed)
    ov, ec, ml, dv, tc, lg = [], [], [], [], [], []
    for tag, c in C.items():
        name = tag_to_name.get(tag, tag)
        treasury = num(c,"treasury",default=0); income = num(c,"income",default=None)
        inflation = num(c,"inflation",default=0); corruption = num(c,"corruption",default=0)
        war_exhaustion = num(c,"war_exhaustion",default=0)
        loans_count = 0
        if isinstance(c.get("loans"), str) and c["loans"].startswith("{"):
            inner = c["loans"][1:-1]
            loans_count = sum(1 for k,_ in tokenize_kv(inner) if k=="loan")
        interest = num(c,"interest",default=None)
        manpower = num(c,"manpower",default=0); max_manpower = num(c,"max_manpower",default=None)
        land_forcelimit = num(c,"land_forcelimit","land_forcelimit_modifier",default=None)
        army_trad = num(c,"army_tradition",default=None); army_prof = num(c,"army_professionalism",default=None)
        discipline = num(c,"discipline",default=None); land_morale = num(c,"land_morale",default=None)
        adm = num(c,"adm_tech",default=None); dip = num(c,"dip_tech",default=None); mil = num(c,"mil_tech",default=None)
        legitimacy = num(c,"legitimacy",default=None); rep = num(c,"republican_tradition",default=None)
        devo = num(c,"devotion",default=None); horde = num(c,"horde_unity",default=None)
        merit = num(c,"meritocracy",default=None); absu = num(c,"absolutism",default=None)
        gov = num(c,"government_reform_progress",default=None); pres = num(c,"prestige",default=None)
        stab = num(c,"stability",default=None)
        agg = D.get(tag, {"province_count":0,"total_development":0.0,"avg_development":0.0})

        aq = 0.0
        for v,w in [(discipline,2.0),(land_morale,2.0),(army_trad,1.0),(army_prof,1.0),(mil,1.5)]:
            if v is not None: aq += float(v)*w

        base = {"tag":tag,"name":name}
        ov.append({**base,"province_count":agg["province_count"],"total_development":round(agg["total_development"],2),
                   "avg_development":round(agg["avg_development"],2) if agg["province_count"] else 0,
                   "income": income if income is not None else "", "manpower": manpower,
                   "army_quality_score": round(aq,2)})
        ec.append({**base,"income": income if income is not None else "","treasury":treasury,"inflation":inflation,
                   "loans": loans_count or "","interest": interest if interest is not None else "",
                   "war_exhaustion":war_exhaustion,"corruption":corruption})
        ml.append({**base,"army_quality_score":round(aq,2),"manpower":manpower,"max_manpower":max_manpower or "",
                   "land_forcelimit":land_forcelimit or "","army_tradition":army_trad or "",
                   "army_professionalism":army_prof or "","discipline":discipline or "","land_morale":land_morale or ""})
        dv.append({**base,"province_count":agg["province_count"],"total_development":round(agg["total_development"],2),
                   "avg_development":round(agg["avg_development"],2) if agg["province_count"] else 0})
        tc.append({**base,"adm_tech":adm or "","dip_tech":dip or "","mil_tech":mil or ""})
        lg.append({**base,"absolutism":absu or "","legitimacy":legitimacy or "","republican_tradition":rep or "",
                   "devotion":devo or "","horde_unity":horde or "","meritocracy":merit or "",
                   "government_reform_progress":gov or "","prestige":pres or "","stability":stab or ""})
    return {"overview": ov, "economy": ec, "military": ml, "development": dv, "technology": tc, "legitimacy": lg}

def write_all_csvs(parsed: dict, assets_dir: Path, name_csv: Optional[Path] = None):
    tag_to_name = {}
    p = name_csv or (assets_dir / ASSET["country_names"])
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if row and len(row)>=2 and len(row[0].strip())==3:
                    tag_to_name[row[0].strip().upper()] = row[1].strip()
    T = build_rows(parsed, tag_to_name)
    def W(fname, fields, key): write_csv(assets_dir / fname, fields, T[key])
    W("overview.csv",    ["tag","name","province_count","total_development","avg_development","income","manpower","army_quality_score"], "overview")
    W("economy.csv",     ["tag","name","income","treasury","inflation","loans","interest","war_exhaustion","corruption"], "economy")
    W("military.csv",    ["tag","name","army_quality_score","manpower","max_manpower","land_forcelimit","army_tradition","army_professionalism","discipline","land_morale"], "military")
    W("development.csv", ["tag","name","province_count","total_development","avg_development"], "development")
    W("technology.csv",  ["tag","name","adm_tech","dip_tech","mil_tech"], "technology")
    W("legitimacy.csv",  ["tag","name","absolutism","legitimacy","republican_tradition","devotion","horde_unity","meritocracy","government_reform_progress","prestige","stability"], "legitimacy")

# -------- Assets --------
def load_definition_csv(path: Path) -> Dict[Tuple[int,int,int], int]:
    lut = {}
    with path.open("r", encoding="latin-1") as f:
        rdr = csv.reader(f, delimiter=";")
        for row in rdr:
            if not row or row[0].startswith("#"): continue
            try:
                pid = int(row[0]); r,g,b = int(row[1]), int(row[2]), int(row[3])
            except Exception: continue
            lut[(r,g,b)] = pid
    return lut

def parse_default_map(path: Path) -> Dict[str, set]:
    txt = path.read_text(encoding="latin-1", errors="ignore")
    data = {}
    for key in ("sea_starts","only_used_for_random"):
        s,e = find_block(txt, key); vals = set()
        if s != -1:
            inner = txt[s:e]; nums = re.findall(r"\b\d+\b", inner)
            vals = set(int(n) for n in nums)
        data[key] = vals
    return data

def parse_country_colors(path: Path) -> Dict[str, Tuple[int,int,int]]:
    colors = {}
    if not path.exists(): return colors
    txt = path.read_text(encoding="latin-1", errors="ignore")
    tag_pat = re.compile(r"\b([A-Z0-9]{3})\s*=\s*{", re.M); idx = 0
    while True:
        m = tag_pat.search(txt, idx)
        if not m: break
        tag = m.group(1); i = m.end()-1; depth = 0
        for j in range(i, len(txt)):
            if txt[j]=="{": depth+=1
            elif txt[j]=="}":
                depth-=1
                if depth==0:
                    block = txt[i:j+1]
                    c = re.search(r"color1\s*=\s*{\s*(\d+)\s+(\d+)\s+(\d+)", block)
                    if c: colors[tag]=(int(c.group(1)),int(c.group(2)),int(c.group(3)))
                    idx = j+1; break
        else: break
    return colors

def tag_random_color(tag: str) -> Tuple[int,int,int]:
    h = hash(tag) & 0xFFFFFF
    return ((h>>16)&255, (h>>8)&255, h&255)

# -------- Map renderer (24-bit LUT) --------
def pack_rgb_uint32(arr: np.ndarray) -> np.ndarray:
    return (arr[:,:,0].astype(np.uint32)<<16) | (arr[:,:,1].astype(np.uint32)<<8) | arr[:,:,2].astype(np.uint32)

def render_map(provinces_bmp: Path, definition_csv: Path, default_map: Path, parsed: dict,
               colors_txt: Optional[Path], out_path: Path) -> None:
    im = Image.open(provinces_bmp).convert("RGB")
    arr = np.array(im, dtype=np.uint8)            # HxWx3
    H, W, _ = arr.shape

    rgb2id = load_definition_csv(definition_csv)
    dm = parse_default_map(default_map)
    sea_ids = dm.get("sea_starts", set())
    tag_colors = parse_country_colors(colors_txt) if colors_txt else {}
    owners = {pid: d.get("owner") for pid, d in parsed["provinces"].items()}

    def color_for_tag(tag: Optional[str]) -> Tuple[int,int,int]:
        if not tag: return UNOCCUPIED
        return tag_colors.get(tag, tag_random_color(tag))

    # Build 24-bit LUT: (16777216, 3) uint8; initialize to UNOCCUPIED
    lut = np.empty((1<<24, 3), dtype=np.uint8)
    lut[:] = np.array(UNOCCUPIED, dtype=np.uint8)

    # Pre-fill LUT for all known province colors
    for (r,g,b), pid in rgb2id.items():
        if pid in sea_ids:
            lut[(r<<16)|(g<<8)|b] = np.array(DEEP_OCEAN, dtype=np.uint8)
        else:
            tag = owners.get(pid)
            lut[(r<<16)|(g<<8)|b] = np.array(color_for_tag(tag), dtype=np.uint8)

    # Map image in one go
    packed = pack_rgb_uint32(arr)                 # HxW uint32
    out = lut[packed]                             # HxW x 3

    # Land/sea masks for coastal tint
    sea_mask = (out[:,:,0]==DEEP_OCEAN[0]) & (out[:,:,1]==DEEP_OCEAN[1]) & (out[:,:,2]==DEEP_OCEAN[2])
    land_mask = ~sea_mask & ~(
        (out[:,:,0]==UNOCCUPIED[0]) & (out[:,:,1]==UNOCCUPIED[1]) & (out[:,:,2]==UNOCCUPIED[2])
    )
    neighbors = (
        np.pad(land_mask[1: ,:], ((0,1),(0,0))) |
        np.pad(land_mask[:-1,:], ((1,0),(0,0))) |
        np.pad(land_mask[:, 1:], ((0,0),(0,1))) |
        np.pad(land_mask[:, :-1],((0,0),(1,0)))
    )
    coastal = sea_mask & neighbors
    out[coastal] = np.array(COASTAL_SEA, dtype=np.uint8)

    # Save (palette) to reduce memory & size
    Image.fromarray(out, mode="RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=256).save(out_path, optimize=True)

# -------- Main --------
def main():
    args = parse_args()
    assets = Path(args.assets).resolve()
    save_path = Path(args.save).resolve()
    out_path = Path(args.out).resolve()

    # Read save (compressed or plaintext)
    save_text = read_save_text(save_path)
    parsed = parse_save_minimal(save_text)

    # Write CSVs
    write_all_csvs(parsed, assets_dir=assets)

    # Render map
    provinces_bmp = assets / ASSET["provinces_bmp"]
    definition_csv = assets / ASSET["definition_csv"]
    default_map = assets / ASSET["default_map"]
    colors_txt = assets / ASSET["colors_txt"]
    for p in (provinces_bmp, definition_csv, default_map):
        if not p.exists(): raise SystemExit(f"Missing asset: {p}")

    render_map(provinces_bmp, definition_csv, default_map, parsed, colors_txt if colors_txt.exists() else None, out_path)
    print(f"[OK] Map → {out_path}\n[OK] CSVs → {assets}")

if __name__ == "__main__":
    main()
