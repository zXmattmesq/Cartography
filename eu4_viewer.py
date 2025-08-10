#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eu4_viewer.py  —  Minimal, robust EU4 save renderer + ledgers (low RAM)

Inputs (assets dir must contain):
- provinces.bmp
- definition.csv
- default.map
- 00_country_colors.txt

Outputs:
- Map image (RGB) to --out
- CSVs to --assets directory:
  overview.csv, economy.csv, military.csv, development.csv, technology.csv, legitimacy.csv
"""

from __future__ import annotations
import argparse
import csv
import io
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------
# Utility / IO
# ---------------------------

def read_text_save(save_path: Path) -> str:
    """Return the EU4 save as text (handles zipped saves)."""
    data = save_path.read_bytes()
    # Zipped saves start with PK
    if len(data) >= 2 and data[0:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # Common entry name is 'gamestate'
            name = "gamestate"
            if name not in zf.namelist():
                # fallback: first non-meta file
                names = [n for n in zf.namelist() if not n.endswith(".meta")]
                if not names:
                    raise RuntimeError("Zip save has no gamestate")
                name = names[0]
            txt = zf.read(name)
            return txt.decode("latin-1", errors="ignore")
    # Plain text
    try:
        return data.decode("latin-1", errors="ignore")
    except Exception:
        return data.decode("utf-8", errors="ignore")


def load_definition(def_csv: Path) -> Dict[Tuple[int, int, int], int]:
    """Map province RGB -> province id."""
    mapping: Dict[Tuple[int, int, int], int] = {}
    with def_csv.open("r", encoding="latin-1") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or row[0].startswith("#"):  # comment
                continue
            try:
                pid = int(row[0])
                r, g, b = int(row[1]), int(row[2]), int(row[3])
            except Exception:
                continue
            if (r, g, b) == (0, 0, 0):
                continue
            mapping[(r, g, b)] = pid
    return mapping


def parse_default_map(default_map: Path) -> Dict[str, set]:
    """Read default.map to get sea provinces and RNW sea, etc."""
    text = default_map.read_text(encoding="latin-1", errors="ignore")
    def _read_set(key: str) -> set:
        m = re.search(rf"{key}\s*=\s*\{{([^}}]+)\}}", text, re.S)
        if not m:
            return set()
        nums = re.findall(r"\d+", m.group(1))
        return set(int(n) for n in nums)
    return {
        "sea": _read_set("sea_starts"),
        "only_used_for_random": _read_set("only_used_for_random"),
    }


def parse_country_colors(colors_txt: Path) -> Dict[str, Tuple[int, int, int]]:
    """TAG -> color1 (r,g,b)."""
    if not colors_txt.exists():
        return {}
    txt = colors_txt.read_text(encoding="latin-1", errors="ignore")
    colors: Dict[str, Tuple[int, int, int]] = {}
    # Blocks: TAG = { ... color1= { r g b } ... }
    tag_pat = re.compile(r"\b([A-Z0-9]{3})\s*=\s*{", re.M)
    idx = 0
    while True:
        m = tag_pat.search(txt, idx)
        if not m:
            break
        tag = m.group(1)
        i = m.end() - 1
        depth = 0
        end = None
        for j in range(i, len(txt)):
            if txt[j] == "{":
                depth += 1
            elif txt[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            break
        block = txt[i:end]
        c = re.search(r"color1\s*=\s*\{\s*(\d+)\s+(\d+)\s+(\d+)\s*\}", block)
        if c:
            r, g, b = int(c.group(1)), int(c.group(2)), int(c.group(3))
            colors[tag] = (r, g, b)
        idx = end
    return colors


# ---------------------------
# Save parsing (minimal/robust)
# ---------------------------

def _strip_comments(s: str) -> str:
    out_lines = []
    for line in s.splitlines():
        # remove anything after '#'
        p = line.find("#")
        if p != -1:
            line = line[:p]
        out_lines.append(line)
    return "\n".join(out_lines)


def _find_block(s: str, key: str) -> Tuple[int, int]:
    """Return slice [start,end) of key={...} or (-1,-1)."""
    m = re.search(rf"\b{re.escape(key)}\s*=\s*{{", s)
    if not m:
        return (-1, -1)
    i = m.end() - 1
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return (m.start(), j + 1)
    return (-1, -1)


def _block_to_dict(block: str) -> Dict[str, object]:
    """Very small 'clausewitz' to python-ish mapping for a single block."""
    inner = block
    # Strip outer braces
    p = inner.find("{")
    q = inner.rfind("}")
    if p != -1 and q != -1 and q > p:
        inner = inner[p + 1 : q]

    d: Dict[str, object] = {}
    # Simple key = value OR key = { ... } (keep as raw string)
    tok = re.finditer(r"([A-Za-z0-9_\.]+)\s*=\s*([^\n{}]+|{[^{}]*})", inner)
    for m in tok:
        k = m.group(1)
        v = m.group(2).strip()
        # Try number/bool
        if v.startswith("{") and v.endswith("}"):
            d[k] = v  # keep raw
        elif v in ("yes", "no", "true", "false"):
            d[k] = v in ("yes", "true")
        else:
            try:
                if "." in v:
                    d[k] = float(v)
                else:
                    d[k] = int(v)
            except Exception:
                v = v.strip('"')
                d[k] = v
    return d


def parse_save_minimal(txt: str) -> Dict[str, object]:
    """Return dict with countries and provinces (owner & a few stats).
       Falls back to owned_provinces when provinces block is absent.
    """
    s = _strip_comments(txt)

    # --- countries ---
    countries: Dict[str, Dict[str, object]] = {}
    a, b = _find_block(s, "countries")
    if a != -1:
        inner = s[a:b]
        inner = inner[inner.find("{") + 1 : inner.rfind("}")]
        tag_pat = re.compile(r"\b([A-Z0-9]{3})\s*=\s*{", re.M)
        idx = 0
        while True:
            m = tag_pat.search(inner, idx)
            if not m:
                break
            tag = m.group(1)
            i = m.end() - 1
            depth = 0
            end = None
            for j in range(i, len(inner)):
                if inner[j] == "{":
                    depth += 1
                elif inner[j] == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end is None:
                break
            block = inner[i:end]
            countries[tag] = _block_to_dict(block)
            idx = end

    # --- provinces ---
    provinces: Dict[int, Dict[str, object]] = {}
    a, b = _find_block(s, "provinces")
    if a != -1:
        inner = s[a:b]
        inner = inner[inner.find("{") + 1 : inner.rfind("}")]
        id_pat = re.compile(r"\b(\d+)\s*=\s*{", re.M)
        idx = 0
        while True:
            m = id_pat.search(inner, idx)
            if not m:
                break
            pid = int(m.group(1))
            i = m.end() - 1
            depth = 0
            end = None
            for j in range(i, len(inner)):
                if inner[j] == "{":
                    depth += 1
                elif inner[j] == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end is None:
                break
            block = inner[i:end]
            d = _block_to_dict(block)
            provinces[pid] = {
                "owner": d.get("owner"),
                "is_city": True if d.get("is_city") is True else False,
                "colony": True if d.get("colony") is True else False,
                "base_tax": float(d.get("base_tax", 0) or 0),
                "base_production": float(d.get("base_production", 0) or 0),
                "base_manpower": float(d.get("base_manpower", 0) or 0),
                "is_sea": True if (d.get("is_sea") is True or d.get("sea_zone") is True) else False,
            }
            idx = end

    # Fallback ownership from owned_provinces if provinces[] absent
    if not provinces:
        num_pat = re.compile(r"\b\d+\b")
        owner_map: Dict[int, str] = {}
        for tag, c in countries.items():
            owned = c.get("owned_provinces")
            if isinstance(owned, str) and owned.startswith("{"):
                for n in num_pat.findall(owned):
                    owner_map[int(n)] = tag
        for pid, tag in owner_map.items():
            provinces[pid] = {
                "owner": tag,
                "is_city": True,
                "colony": False,
                "base_tax": 0.0,
                "base_production": 0.0,
                "base_manpower": 0.0,
                "is_sea": False,
            }

    return {"countries": countries, "provinces": provinces}


# ---------------------------
# Ledger extraction
# ---------------------------

def country_display_name(tag: str, cdict: Dict[str, object]) -> str:
    n = cdict.get("name")
    if isinstance(n, str) and n:
        return n
    return tag


def extract_ledgers(parsed: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    countries: Dict[str, Dict[str, object]] = parsed["countries"]  # type: ignore
    # Aggregate province dev by owner
    provinces: Dict[int, Dict[str, object]] = parsed["provinces"]  # type: ignore
    dev_tot: Dict[str, float] = {}
    prov_cnt: Dict[str, int] = {}
    for pid, d in provinces.items():
        tag = d.get("owner")
        if not isinstance(tag, str):
            continue
        bt = float(d.get("base_tax", 0) or 0)
        bp = float(d.get("base_production", 0) or 0)
        bm = float(d.get("base_manpower", 0) or 0)
        dev_tot[tag] = dev_tot.get(tag, 0.0) + (bt + bp + bm)
        prov_cnt[tag] = prov_cnt.get(tag, 0) + 1

    rows_overview = []
    rows_econ = []
    rows_mil = []
    rows_dev = []
    rows_tech = []
    rows_leg = []

    for tag, c in countries.items():
        name = country_display_name(tag, c)
        total_dev = round(dev_tot.get(tag, 0.0), 3)
        pcount = prov_cnt.get(tag, 0)

        # Common fields (best-effort; many saves omit some)
        treasury = c.get("treasury", "")
        income = c.get("income", "")
        inflation = c.get("inflation", "")
        loans = c.get("loans", "")
        interest = c.get("interest", "")
        war_exhaustion = c.get("war_exhaustion", "")
        corruption = c.get("corruption", "")

        manpower = c.get("manpower", "")
        max_manpower = c.get("max_manpower", "")
        land_fl = c.get("land_forcelimit", "")
        army_prof = c.get("army_professionalism", "")
        army_trad = c.get("army_tradition", "")

        mil_tech = c.get("mil_tech", c.get("technology_military"))
        adm_tech = c.get("adm_tech", c.get("technology_administrative"))
        dip_tech = c.get("dip_tech", c.get("technology_diplomatic"))

        legitimacy = c.get("legitimacy", c.get("republican_tradition", ""))

        rows_overview.append({
            "tag": tag, "name": name,
            "province_count": pcount,
            "total_development": round(total_dev, 3),
            "avg_development": round(total_dev / pcount, 3) if pcount else 0,
            "income": income,
            "manpower": manpower,
            "army_quality_score": round(float(army_prof or 0) + float(army_trad or 0), 3) if str(army_prof).strip() != "" else "",
        })

        rows_econ.append({
            "tag": tag, "name": name,
            "income": income, "treasury": treasury, "inflation": inflation,
            "loans": loans, "interest": interest, "war_exhaustion": war_exhaustion,
            "corruption": corruption,
        })

        rows_mil.append({
            "tag": tag, "name": name,
            "army_quality_score": round(float(army_prof or 0) + float(army_trad or 0), 3) if str(army_prof).strip() != "" else "",
            "manpower": manpower, "max_manpower": max_manpower,
            "land_forcelimit": land_fl, "army_tradition": army_trad,
            "army_professionalism": army_prof,
        })

        rows_dev.append({
            "tag": tag, "name": name,
            "province_count": pcount,
            "total_development": round(total_dev, 3),
            "avg_development": round(total_dev / pcount, 3) if pcount else 0,
        })

        rows_tech.append({
            "tag": tag, "name": name,
            "adm_tech": adm_tech, "dip_tech": dip_tech, "mil_tech": mil_tech,
        })

        rows_leg.append({
            "tag": tag, "name": name,
            "legitimacy_or_republican_tradition": legitimacy,
        })

    return {
        "overview": rows_overview,
        "economy": rows_econ,
        "military": rows_mil,
        "development": rows_dev,
        "technology": rows_tech,
        "legitimacy": rows_leg,
    }


def write_csvs(csvdir: Path, ledgers: Dict[str, List[Dict[str, object]]]) -> None:
    csvdir.mkdir(parents=True, exist_ok=True)
    order = {
        "overview": ["tag", "name", "province_count", "total_development", "avg_development", "income", "manpower", "army_quality_score"],
        "economy": ["tag", "name", "income", "treasury", "inflation", "loans", "interest", "war_exhaustion", "corruption"],
        "military": ["tag", "name", "army_quality_score", "manpower", "max_manpower", "land_forcelimit", "army_tradition", "army_professionalism"],
        "development": ["tag", "name", "province_count", "total_development", "avg_development"],
        "technology": ["tag", "name", "adm_tech", "dip_tech", "mil_tech"],
        "legitimacy": ["tag", "name", "legitimacy_or_republican_tradition"],
    }
    for key, rows in ledgers.items():
        path = csvdir / f"{key}.csv"
        cols = order[key]
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})


# ---------------------------
# Rendering
# ---------------------------

DEEP_BLUE = (28, 80, 160)
COAST_BLUE = (48, 120, 200)
UNOWNED_GRAY = (190, 190, 190)


def render_map(
    provinces_img: Path,
    color_to_pid: Dict[Tuple[int, int, int], int],
    owners: Dict[int, Optional[str]],
    sea_ids: set,
    country_colors: Dict[str, Tuple[int, int, int]],
    out_path: Path,
    chunk: int = 64,
    scale: float = 0.75,
):
    img = Image.open(provinces_img).convert("RGB")
    W, H = img.size

    if scale != 1.0:
        W2, H2 = int(W * scale), int(H * scale)
    else:
        W2, H2 = W, H

    # Create output buffer
    out = np.zeros((H2, W2, 3), dtype=np.uint8)

    # Helper: map province id to display color
    def color_for_pid(pid: int, x: int, y: int, neighbor_land: bool) -> Tuple[int, int, int]:
        if pid in sea_ids:
            return COAST_BLUE if neighbor_land else DEEP_BLUE
        tag = owners.get(pid)
        if not tag:
            return UNOWNED_GRAY
        return country_colors.get(tag, (130, 130, 130))

    # Precompute province id grid, streamed by chunks for memory
    # Also detect if sea pixel touches land (neighbor up/left)
    # We’ll process at full res, then downscale at the end if needed.
    def rgb_to_pid_block(block: np.ndarray) -> np.ndarray:
        # block: (h, w, 3)
        # Convert to tuples and map; vectorized via view then dict lookup per unique color
        h, w, _ = block.shape
        flat = block.reshape(-1, 3)
        # Build pid array with default 0
        pids = np.zeros((h * w,), dtype=np.int32)
        # unique colors -> pids then broadcast
        uniq, idx = np.unique(flat, axis=0, return_inverse=True)
        map_arr = np.zeros((uniq.shape[0],), dtype=np.int32)
        for i, (r, g, b) in enumerate(uniq):
            map_arr[i] = color_to_pid.get((int(r), int(g), int(b)), 0)
        pids = map_arr[idx]
        return pids.reshape(h, w)

    # Work at full size; compute colors per pixel then resize if needed
    out_full = np.zeros((H, W, 3), dtype=np.uint8)

    prev_pid_row = None
    prev_is_land_row = None

    for y0 in range(0, H, chunk):
        y1 = min(H, y0 + chunk)
        block = np.array(img.crop((0, y0, W, y1)), dtype=np.uint8)
        pid_block = rgb_to_pid_block(block)

        # Land mask for this block
        is_sea_block = np.isin(pid_block, list(sea_ids))
        is_land_block = ~is_sea_block

        # Check neighbor with previous row to detect coast sea
        neighbor_land = np.zeros_like(is_sea_block, dtype=bool)
        # left neighbor
        neighbor_land[:, 1:] |= is_land_block[:, :-1]
        # up neighbor (from prev chunk)
        if prev_is_land_row is not None:
            neighbor_land[0, :] |= prev_is_land_row
        neighbor_land[1:, :] |= is_land_block[:-1, :]

        # Fill colors
        for j in range(y1 - y0):
            for i in range(W):
                pid = int(pid_block[j, i])
                col = color_for_pid(pid, i, y0 + j, bool(neighbor_land[j, i]))
                out_full[y0 + j, i, :] = col

        prev_pid_row = pid_block[-1, :]
        prev_is_land_row = is_land_block[-1, :]

    final = out_full
    if scale != 1.0:
        final_img = Image.fromarray(final, mode="RGB").resize((W2, H2), Image.NEAREST)
    else:
        final_img = Image.fromarray(final, mode="RGB")

    final_img.save(out_path)
    print(f"[OK] Map → {out_path}")


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="EU4 minimal renderer + ledgers (low RAM)")
    ap.add_argument("save", type=str, help="Path to .eu4 save")
    ap.add_argument("--assets", type=str, default=".", help="Folder with provinces.bmp, definition.csv, default.map, 00_country_colors.txt")
    ap.add_argument("--out", type=str, default="world_map.png", help="Output map PNG")
    ap.add_argument("--chunk", type=int, default=64, help="Row chunk height for streaming render")
    ap.add_argument("--scale", type=float, default=0.75, help="Downscale factor (1.0 = native)")
    args = ap.parse_args()

    assets = Path(args.assets).resolve()
    provinces_bmp = assets / "provinces.bmp"
    definition_csv = assets / "definition.csv"
    default_map = assets / "default.map"
    colors_txt = assets / "00_country_colors.txt"
    out_map = Path(args.out).resolve()

    if not provinces_bmp.exists():
        raise FileNotFoundError(f"Missing {provinces_bmp}")
    if not definition_csv.exists():
        raise FileNotFoundError(f"Missing {definition_csv}")
    if not default_map.exists():
        print("[!] default.map not found — sea detection will be limited.")
    if not colors_txt.exists():
        print("[!] 00_country_colors.txt not found — using fallback colors.")

    # Load assets
    color_to_pid = load_definition(definition_csv)
    sea_sets = parse_default_map(default_map) if default_map.exists() else {"sea": set(), "only_used_for_random": set()}
    sea_ids = set(sea_sets.get("sea", set())) | set(sea_sets.get("only_used_for_random", set()))
    tag_colors = parse_country_colors(colors_txt)

    # Read save
    save_path = Path(args.save)
    save_text = read_text_save(save_path)
    parsed = parse_save_minimal(save_text)

    # Build owners map (province id -> tag or None)
    owners: Dict[int, Optional[str]] = {}
    provinces: Dict[int, Dict[str, object]] = parsed["provinces"]  # type: ignore
    for pid, d in provinces.items():
        tag = d.get("owner")
        owners[pid] = str(tag) if isinstance(tag, str) and tag else None

    # Render map
    render_map(
        provinces_img=provinces_bmp,
        color_to_pid=color_to_pid,
        owners=owners,
        sea_ids=sea_ids,
        country_colors=tag_colors,
        out_path=out_map,
        chunk=int(args.chunk),
        scale=float(args.scale),
    )

    # Write CSVs to assets (so the bot can fetch them)
    ledgers = extract_ledgers(parsed)
    write_csvs(assets, ledgers)
    print(f"[OK] CSVs → {assets}")


if __name__ == "__main__":
    main()
