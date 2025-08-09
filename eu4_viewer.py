#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eu4_viewer.py
- Parse an EU4 save (plaintext) robustly
- Write CSVs that mirror in-game ledger categories
- Render a world ownership map with proper country colors
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

# ========================= CLI =========================

def parse_args():
    ap = argparse.ArgumentParser(description="EU4 save → CSVs + world map")
    ap.add_argument("save", help="Path to .eu4 save (plaintext or already extracted)")
    ap.add_argument("--assets", default=".", help="Folder with provinces.bmp, definition.csv, default.map, 00_country_colors.txt")
    ap.add_argument("--out", default="world_map.png", help="Output PNG path for the map")
    return ap.parse_args()

ASSET_FILENAMES = {
    "provinces_bmp": "provinces.bmp",
    "definition_csv": "definition.csv",
    "default_map": "default.map",
    "colors_txt": "00_country_colors.txt",
    "country_names": "country_names.csv",  # optional (TAG,Name)
}

DEEP_OCEAN = (24, 66, 140)      # deep blue
COASTAL_SEA = (54, 149, 255)    # lighter blue
UNOCCUPIED = (160, 160, 160)    # grey

# ======================================================
# Paradox plaintext helpers (fast + robust)
# ======================================================

_comment = re.compile(r"(?m)^\s*#.*$")
def strip_comments(s: str) -> str:
    return _comment.sub("", s)

def find_block(s: str, key: str) -> Tuple[int, int]:
    pat = re.compile(rf"\b{re.escape(key)}\s*=\s*{{", re.M)
    m = pat.search(s)
    if not m:
        return (-1, -1)
    i = m.end() - 1
    depth = 0
    for j in range(i, len(s)):
        ch = s[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (m.start(), j + 1)
    return (-1, -1)

def tokenize_kv(s: str):
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n or s[i] == "}":
            break
        ks = i
        while i < n and not s[i].isspace() and s[i] not in "={}":
            i += 1
        key = s[ks:i]
        while i < n and s[i].isspace():
            i += 1
        if i >= n or s[i] != "=":
            while i < n and s[i] not in "{}\n":
                i += 1
            continue
        i += 1
        while i < n and s[i].isspace():
            i += 1
        if i < n and s[i] == "{":
            depth = 0
            vs = i
            while i < n:
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        yield key, s[vs:i]
                        break
                i += 1
        elif i < n and s[i] == '"':
            i += 1
            vs = i
            while i < n and s[i] != '"':
                if s[i] == "\\" and i + 1 < n:
                    i += 2
                else:
                    i += 1
            val = s[vs:i]
            i += 1
            yield key, val
        else:
            vs = i
            while i < n and not s[i].isspace() and s[i] not in "{}":
                i += 1
            yield key, s[vs:i]

def parse_scalar(v: str):
    v = v.strip()
    if v.startswith("{") and v.endswith("}"):
        return v
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v in ("yes", "no"):
        return True if v == "yes" else False
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except Exception:
        return v

def parse_block_to_dict(s: str) -> dict:
    if s.startswith("{") and s.endswith("}"):
        body = s[1:-1]
    else:
        body = s
    out = {}
    for k, v in tokenize_kv(body):
        out[k] = parse_scalar(v)
    return out

# ======================================================
# Save → minimal model
# ======================================================

def parse_save_minimal(save_text: str) -> dict:
    """
    Return:
      {
        "countries": { TAG: { fields... } },
        "provinces": { ID: { owner, base_tax, base_production, base_manpower, is_city, colony, is_sea } }
      }
    """
    txt = strip_comments(save_text)

    # Countries
    countries = {}
    s, e = find_block(txt, "countries")
    if s != -1:
        inner = txt[s:e]
        inner = inner[inner.find("{")+1 : inner.rfind("}")]
        tag_pat = re.compile(r"\b([A-Z0-9]{3})\s*=\s*{", re.M)
        idx = 0
        while True:
            m = tag_pat.search(inner, idx)
            if not m: break
            tag = m.group(1)
            i = m.end()-1
            depth = 0
            for j in range(i, len(inner)):
                if inner[j] == "{":
                    depth += 1
                elif inner[j] == "}":
                    depth -= 1
                    if depth == 0:
                        block = inner[i:j+1]
                        countries[tag] = parse_block_to_dict(block)
                        idx = j+1
                        break
            else:
                break

    # Provinces
    provinces = {}
    s, e = find_block(txt, "provinces")
    if s != -1:
        inner = txt[s:e]
        inner = inner[inner.find("{")+1 : inner.rfind("}")]
        id_pat = re.compile(r"\b(\d+)\s*=\s*{", re.M)
        idx = 0
        while True:
            m = id_pat.search(inner, idx)
            if not m: break
            pid = int(m.group(1))
            i = m.end()-1
            depth = 0
            for j in range(i, len(inner)):
                if inner[j] == "{":
                    depth += 1
                elif inner[j] == "}":
                    depth -= 1
                    if depth == 0:
                        block = inner[i:j+1]
                        d = parse_block_to_dict(block)
                        provinces[pid] = {
                            "owner": d.get("owner"),
                            "controller": d.get("controller"),
                            "is_city": True if d.get("is_city") is True else False,
                            "colony": True if d.get("colony") is True else False,
                            "base_tax": float(d.get("base_tax", 0) or 0),
                            "base_production": float(d.get("base_production", 0) or 0),
                            "base_manpower": float(d.get("base_manpower", 0) or 0),
                            "is_sea": True if (d.get("is_sea") is True or d.get("sea_zone") is True) else False,
                        }
                        idx = j+1
                        break
            else:
                break

    return {"countries": countries, "provinces": provinces}

# ======================================================
# CSV builders
# ======================================================

def num(d, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return default

def dev_of_province(d: dict) -> float:
    return float(d.get("base_tax", 0)) + float(d.get("base_production", 0)) + float(d.get("base_manpower", 0))

def aggregate_development(parsed: dict) -> dict:
    provs = parsed["provinces"]
    out = defaultdict(lambda: {"province_count": 0, "total_development": 0.0})
    for pid, p in provs.items():
        tag = p.get("owner")
        if not tag:  # uncolonized/wasteland
            continue
        if p.get("is_sea"):
            continue
        out[tag]["province_count"] += 1
        out[tag]["total_development"] += dev_of_province(p)
    for tag, d in out.items():
        pc = d["province_count"] or 1
        d["avg_development"] = d["total_development"] / pc
    return out

def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def build_rows(parsed: dict, tag_to_name: Dict[str, str]) -> Dict[str, List[dict]]:
    countries = parsed["countries"]
    dev = aggregate_development(parsed)

    rows_overview, rows_econ, rows_mil, rows_dev, rows_tech, rows_legi = [], [], [], [], [], []

    for tag, c in countries.items():
        name = tag_to_name.get(tag, tag)

        treasury = num(c, "treasury", default=0)
        income = num(c, "income", default=None)  # may be absent
        inflation = num(c, "inflation", default=0)
        corruption = num(c, "corruption", default=0)
        war_exhaustion = num(c, "war_exhaustion", default=0)

        loans_count = 0
        if isinstance(c.get("loans"), str) and c["loans"].startswith("{"):
            inner = c["loans"][1:-1]
            loans_count = sum(1 for k, _ in tokenize_kv(inner) if k == "loan")

        interest = num(c, "interest", default=None)

        manpower = num(c, "manpower", default=0)
        max_manpower = num(c, "max_manpower", default=None)
        land_forcelimit = num(c, "land_forcelimit", "land_forcelimit_modifier", default=None)
        army_tradition = num(c, "army_tradition", default=None)
        army_professionalism = num(c, "army_professionalism", default=None)
        discipline = num(c, "discipline", default=None)
        land_morale = num(c, "land_morale", default=None)

        adm_tech = num(c, "adm_tech", default=None)
        dip_tech = num(c, "dip_tech", default=None)
        mil_tech = num(c, "mil_tech", default=None)

        legitimacy = num(c, "legitimacy", default=None)
        republican_tradition = num(c, "republican_tradition", default=None)
        devotion = num(c, "devotion", default=None)
        horde_unity = num(c, "horde_unity", default=None)
        meritocracy = num(c, "meritocracy", default=None)
        absolutism = num(c, "absolutism", default=None)
        gov_reform = num(c, "government_reform_progress", default=None)
        prestige = num(c, "prestige", default=None)
        stability = num(c, "stability", default=None)

        agg = dev.get(tag, {"province_count": 0, "total_development": 0.0, "avg_development": 0.0})

        # simple army quality heuristic; tweak as desired
        aq = 0.0
        for val, w in [(discipline, 2.0), (land_morale, 2.0), (army_tradition, 1.0), (army_professionalism, 1.0), (mil_tech, 1.5)]:
            if val is not None:
                aq += float(val) * w

        base = {"tag": tag, "name": name}

        rows_overview.append({
            **base,
            "province_count": agg["province_count"],
            "total_development": round(agg["total_development"], 2),
            "avg_development": round(agg["avg_development"], 2) if agg["province_count"] else 0,
            "income": income if income is not None else "",
            "manpower": manpower,
            "army_quality_score": round(aq, 2)
        })

        rows_econ.append({
            **base,
            "income": income if income is not None else "",
            "treasury": treasury,
            "inflation": inflation,
            "loans": loans_count or "",
            "interest": interest if interest is not None else "",
            "war_exhaustion": war_exhaustion,
            "corruption": corruption
        })

        rows_mil.append({
            **base,
            "army_quality_score": round(aq, 2),
            "manpower": manpower,
            "max_manpower": max_manpower if max_manpower is not None else "",
            "land_forcelimit": land_forcelimit if land_forcelimit is not None else "",
            "army_tradition": army_tradition if army_tradition is not None else "",
            "army_professionalism": army_professionalism if army_professionalism is not None else "",
            "discipline": discipline if discipline is not None else "",
            "land_morale": land_morale if land_morale is not None else ""
        })

        rows_dev.append({
            **base,
            "province_count": agg["province_count"],
            "total_development": round(agg["total_development"], 2),
            "avg_development": round(agg["avg_development"], 2) if agg["province_count"] else 0
        })

        rows_tech.append({
            **base,
            "adm_tech": adm_tech if adm_tech is not None else "",
            "dip_tech": dip_tech if dip_tech is not None else "",
            "mil_tech": mil_tech if mil_tech is not None else ""
        })

        rows_legi.append({
            **base,
            "absolutism": absolutism if absolutism is not None else "",
            "legitimacy": legitimacy if legitimacy is not None else "",
            "republican_tradition": republican_tradition if republican_tradition is not None else "",
            "devotion": devotion if devotion is not None else "",
            "horde_unity": horde_unity if horde_unity is not None else "",
            "meritocracy": meritocracy if meritocracy is not None else "",
            "government_reform_progress": gov_reform if gov_reform is not None else "",
            "prestige": prestige if prestige is not None else "",
            "stability": stability if stability is not None else ""
        })

    return {
        "overview": rows_overview,
        "economy": rows_econ,
        "military": rows_mil,
        "development": rows_dev,
        "technology": rows_tech,
        "legitimacy": rows_legi,
    }

def write_all_csvs(parsed: dict, assets_dir: Path, tag_to_name_path: Optional[Path] = None) -> None:
    tag_to_name = {}
    p = tag_to_name_path or (assets_dir / ASSET_FILENAMES["country_names"])
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if row and len(row) >= 2 and len(row[0].strip()) == 3:
                    tag_to_name[row[0].strip().upper()] = row[1].strip()

    tables = build_rows(parsed, tag_to_name)

    write_csv(assets_dir / "overview.csv",
              ["tag","name","province_count","total_development","avg_development","income","manpower","army_quality_score"],
              tables["overview"])

    write_csv(assets_dir / "economy.csv",
              ["tag","name","income","treasury","inflation","loans","interest","war_exhaustion","corruption"],
              tables["economy"])

    write_csv(assets_dir / "military.csv",
              ["tag","name","army_quality_score","manpower","max_manpower","land_forcelimit","army_tradition","army_professionalism","discipline","land_morale"],
              tables["military"])

    write_csv(assets_dir / "development.csv",
              ["tag","name","province_count","total_development","avg_development"],
              tables["development"])

    write_csv(assets_dir / "technology.csv",
              ["tag","name","adm_tech","dip_tech","mil_tech"],
              tables["technology"])

    write_csv(assets_dir / "legitimacy.csv",
              ["tag","name","absolutism","legitimacy","republican_tradition","devotion","horde_unity","meritocracy","government_reform_progress","prestige","stability"],
              tables["legitimacy"])

# ======================================================
# Assets: definition.csv, default.map, colors
# ======================================================

def load_definition_csv(path: Path) -> Dict[Tuple[int,int,int], int]:
    """
    definition.csv: id;r;g;b;name;x;x
    Return mapping (r,g,b) -> id
    """
    lut = {}
    with path.open("r", encoding="latin-1") as f:
        rdr = csv.reader(f, delimiter=";")
        for row in rdr:
            if not row or row[0].startswith("#"):  # comments
                continue
            try:
                pid = int(row[0])
                r, g, b = int(row[1]), int(row[2]), int(row[3])
            except Exception:
                continue
            lut[(r,g,b)] = pid
    return lut

def parse_default_map(path: Path) -> Dict[str, set]:
    txt = path.read_text(encoding="latin-1", errors="ignore")
    data = {}
    for key in ("sea_starts", "only_used_for_random"):
        s, e = find_block(txt, key)
        vals = set()
        if s != -1:
            inner = txt[s:e]
            numbers = re.findall(r"\b\d+\b", inner)
            vals = set(int(n) for n in numbers)
        data[key] = vals
    return data

def parse_country_colors(path: Path) -> Dict[str, Tuple[int,int,int]]:
    """
    00_country_colors.txt → TAG -> (r,g,b) using color1
    """
    colors = {}
    if not path.exists():
        return colors
    txt = path.read_text(encoding="latin-1", errors="ignore")
    # blocks like TAG = { ... color1= { r g b } ... }
    tag_pat = re.compile(r"\b([A-Z0-9]{3})\s*=\s*{", re.M)
    idx = 0
    while True:
        m = tag_pat.search(txt, idx)
        if not m: break
        tag = m.group(1)
        i = m.end()-1
        depth = 0
        for j in range(i, len(txt)):
            if txt[j] == "{":
                depth += 1
            elif txt[j] == "}":
                depth -= 1
                if depth == 0:
                    block = txt[i:j+1]
                    c = re.search(r"color1\s*=\s*{\s*(\d+)\s+(\d+)\s+(\d+)", block)
                    if c:
                        colors[tag] = (int(c.group(1)), int(c.group(2)), int(c.group(3)))
                    idx = j+1
                    break
        else:
            break
    return colors

def tag_random_color(tag: str) -> Tuple[int,int,int]:
    # stable "random" color per tag
    h = hash(tag) & 0xFFFFFF
    return ((h >> 16) & 255, (h >> 8) & 255, h & 255)

# ======================================================
# Map rendering
# ======================================================

def render_map(
    provinces_bmp: Path,
    definition_csv: Path,
    default_map: Path,
    parsed: dict,
    colors_txt: Optional[Path],
    out_path: Path
) -> None:

    # Load province image
    im = Image.open(provinces_bmp).convert("RGB")
    arr = np.array(im)  # H x W x 3

    # Build RGB -> province id
    rgb2id = load_definition_csv(definition_csv)

    # Build owner dict
    owners = {pid: data.get("owner") for pid, data in parsed["provinces"].items()}

    # Sea province list (from default.map)
    dm = parse_default_map(default_map)
    sea_ids = dm.get("sea_starts", set())

    # Country colors
    tag_colors = parse_country_colors(colors_txt) if colors_txt else {}
    def color_for_tag(tag: Optional[str]) -> Tuple[int,int,int]:
        if not tag:
            return UNOCCUPIED
        return tag_colors.get(tag, tag_random_color(tag))

    # Build output image by unique colors in provinces.bmp
    H, W, _ = arr.shape
    out = np.zeros_like(arr)

    # Pre-calc: map each unique RGB in the provinces image to final fill color
    uniq = np.unique(arr.reshape(-1, 3), axis=0)
    rgb_tuple_list = [tuple(x) for x in uniq.tolist()]

    fill_for_rgb: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}
    land_mask = np.zeros((H, W), dtype=bool)
    sea_mask = np.zeros((H, W), dtype=bool)

    for rgb in rgb_tuple_list:
        pid = rgb2id.get(rgb)
        if pid is None:
            fill_for_rgb[rgb] = UNOCCUPIED
            continue
        if pid in sea_ids:
            fill_for_rgb[rgb] = DEEP_OCEAN  # refine to coastal later
        else:
            tag = owners.get(pid)
            c = color_for_tag(tag)
            fill_for_rgb[rgb] = c

    # Apply fills
    # Make a small LUT by building a dict from RGB → mask
    for rgb, fill in fill_for_rgb.items():
        mask = (arr[:, :, 0] == rgb[0]) & (arr[:, :, 1] == rgb[1]) & (arr[:, :, 2] == rgb[2])
        out[mask] = fill
        if fill == DEEP_OCEAN:
            sea_mask |= mask
        elif fill != UNOCCUPIED:
            land_mask |= mask

    # Mark coastal sea: any sea pixel that touches land → lighter blue
    # 4-neighborhood adjacency by shifting land mask one pixel in each direction
    lm = land_mask
    neighbors = (
        np.pad(lm[1:, :], ((0,1),(0,0))) |
        np.pad(lm[:-1, :], ((1,0),(0,0))) |
        np.pad(lm[:, 1:], ((0,0),(0,1))) |
        np.pad(lm[:, :-1], ((0,0),(1,0)))
    )
    coastal = sea_mask & neighbors
    out[coastal] = COASTAL_SEA

    Image.fromarray(out).save(out_path)

# ======================================================
# Main
# ======================================================

def main():
    args = parse_args()
    assets = Path(args.assets).resolve()
    save_path = Path(args.save).resolve()
    out_path = Path(args.out).resolve()

    # read save (plaintext). If your save is zipped, unzip outside and pass the plaintext.
    save_text = save_path.read_text(encoding="latin-1", errors="ignore")

    parsed = parse_save_minimal(save_text)
    # Write ledger CSVs next to the script (the bot copies them to its workdir)
    write_all_csvs(parsed, assets_dir=assets)

    # Render map
    provinces_bmp = assets / ASSET_FILENAMES["provinces_bmp"]
    definition_csv = assets / ASSET_FILENAMES["definition_csv"]
    default_map = assets / ASSET_FILENAMES["default_map"]
    colors_txt = assets / ASSET_FILENAMES["colors_txt"]

    if not provinces_bmp.exists():
        raise SystemExit(f"Missing {provinces_bmp}")
    if not definition_csv.exists():
        raise SystemExit(f"Missing {definition_csv}")
    if not default_map.exists():
        raise SystemExit(f"Missing {default_map}")

    render_map(
        provinces_bmp=provinces_bmp,
        definition_csv=definition_csv,
        default_map=default_map,
        parsed=parsed,
        colors_txt=colors_txt if colors_txt.exists() else None,
        out_path=out_path
    )
    print(f"[OK] Wrote map to: {out_path}")
    print(f"[OK] Wrote CSVs to: {assets}")

if __name__ == "__main__":
    main()
