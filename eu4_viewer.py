#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EU4 Viewer — lightweight save inspector & map renderer

Features
- Reads uncompressed or zipped (.eu4) saves (text form)
- Extracts province owners, country stats (best-effort), province development
- Renders a country-colored world map using provinces.bmp + definition.csv
- Writes multiple CSV ledgers (overview, economy, military, development, technology, legitimacy)

Designed to run in low-memory environments. Rendering is chunked row-wise and the
save parsing is regex-based (robust enough for most vanilla saves; not a full parser).

CLI
    python eu4_viewer.py <save.eu4> \
        --assets . \
        --out world_map.png \
        --chunk 64 \
        --scale 0.75

Assets required in --assets directory:
    - provinces.bmp                      (from Europa Universalis IV/gfx/map/provinces.bmp)
    - definition.csv                     (from Europa Universalis IV/map/definition.csv)
    - default.map                        (to detect sea provinces)
    - 00_country_colors.txt              (optional; map tag -> color1)
    - tag_names.csv                      (optional; two columns: tag,name)

Outputs:
    - <out>                              rendered map PNG
    - overview.csv, economy.csv, military.csv,
      development.csv, technology.csv, legitimacy.csv  (in assets dir)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

# -------------------------------------------------------------
# Fallback tag->name map (used if tag_names.csv not provided)
# -------------------------------------------------------------
TAG_NAME_FALLBACK: Dict[str, str] = {
    "FRA": "France", "ENG": "England", "GBR": "Great Britain", "CAS": "Castile",
    "SPA": "Spain", "POR": "Portugal", "ARA": "Aragon", "HAB": "Austria", "AUS": "Austria",
    "TUR": "Ottomans", "MAM": "Mamluks", "MOS": "Muscovy", "RUS": "Russia", "POL": "Poland",
    "LIT": "Lithuania", "SWE": "Sweden", "NOR": "Norway", "DEN": "Denmark", "NED": "Netherlands",
    "PRU": "Prussia", "BRA": "Brandenburg", "SAX": "Saxony", "BOH": "Bohemia", "HUN": "Hungary",
    "VEN": "Venice", "PAP": "Papal State", "NAP": "Naples", "MLO": "Milan", "TUS": "Tuscany",
    "SAV": "Savoy", "MNG": "Ming", "QIN": "Qing", "JAP": "Japan", "KOR": "Korea",
    "VIJ": "Vijayanagar", "BHA": "Bharat", "DLH": "Delhi", "MEX": "Mexico", "USA": "United States",
    "BRZ": "Brazil", "CAN": "Canada", "QUE": "Quebec", "LOU": "Louisiana", "AYU": "Ayutthaya",
    "KHM": "Khmer", "LAN": "Lan Xang", "LUA": "Luang Prabang", "DAI": "Dai Viet",
}

# Words that hint a unit/army/fleet name
UNIT_WORDS = (
    "regiment", "samurai", "company", "banner", "cohort", "merc", "fleet", "army", "navy"
)
_name_like_person = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$")
_tag3 = re.compile(r"^[A-Z]{3}$")

# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------

def read_text_save(path: Path) -> str:
    """Read a .eu4 save. If zipped, extract 'gamestate'. Return latin-1 text."""
    data: bytes
    p = Path(path)
    try:
        with zipfile.ZipFile(p) as z:
            # Locate 'gamestate' (Vanilla EU4 zipped saves)
            name = None
            for n in z.namelist():
                if n.endswith("gamestate"):
                    name = n
                    break
            if name is None:
                # fallback: first file
                name = z.namelist()[0]
            data = z.read(name)
            try:
                return data.decode("latin-1")
            except Exception:
                return data.decode("utf-8", errors="ignore")
    except zipfile.BadZipFile:
        # Plain text save
        raw = p.read_bytes()
        try:
            return raw.decode("latin-1")
        except Exception:
            return raw.decode("utf-8", errors="ignore")


def parse_numbers_block(text: str, key: str) -> List[int]:
    """Parse a Paradox-style list of integers:  key = { 1 2 3 }  -> [1,2,3]."""
    m = re.search(rf"{re.escape(key)}\s*=\s*\{{([^}}]*)\}}", text)
    if not m:
        return []
    nums = re.findall(r"-?\d+", m.group(1))
    return [int(n) for n in nums]


def load_default_map_sea_ids(default_map_text: str) -> set[int]:
    sea = set(parse_numbers_block(default_map_text, "sea_starts"))
    # some files also put sea in other blocks; this captures the bulk.
    return sea


def load_definition_csv(path: Path) -> Dict[Tuple[int, int, int], int]:
    """Return mapping (R,G,B) -> province_id."""
    m: Dict[Tuple[int, int, int], int] = {}
    with path.open("r", encoding="latin-1", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            try:
                pid = int(row[0])
                r, g, b = int(row[1]), int(row[2]), int(row[3])
            except Exception:
                continue
            m[(r, g, b)] = pid
    return m


def load_country_colors(path: Path) -> Dict[str, Tuple[int, int, int]]:
    """Parse 00_country_colors.txt and return TAG -> (r,g,b) from color1."""
    if not path.exists():
        return {}
    txt = path.read_text(encoding="latin-1", errors="ignore")
    out: Dict[str, Tuple[int, int, int]] = {}
    # Match TAG = { ... color1= { r g b } ... }
    for m in re.finditer(r"([A-Z]{3})\s*=\s*\{([^}]*)\}", txt, flags=re.S):
        tag = m.group(1)
        block = m.group(2)
        mcol = re.search(r"color1\s*=\s*\{\s*(\d+)\s+(\d+)\s+(\d+)\s*\}", block)
        if mcol:
            out[tag] = (int(mcol.group(1)), int(mcol.group(2)), int(mcol.group(3)))
    return out


def load_tag_names_csv(assets_dir: Path) -> Dict[str, str]:
    p = assets_dir / "tag_names.csv"
    if not p.exists():
        return {}
    mapping: Dict[str, str] = {}
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                tag, nm = parts[0].upper(), ",".join(parts[1:]).strip()
                if tag and nm:
                    mapping[tag] = nm
    return mapping


def looks_unitish(s: str) -> bool:
    ss = s.lower()
    return any(w in ss for w in UNIT_WORDS) or ("'s " in ss) or ss.endswith(" regiment")


def country_label(tag: str, c: Dict[str, object], tag_names: Dict[str, str]) -> str:
    # 1) explicit custom name fields
    for k in ("custom_name", "country_name", "long_name"):
        v = c.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 2) cautious use of 'name'
    v = c.get("name")
    if isinstance(v, str):
        s = v.strip()
        if s and not _tag3.fullmatch(s) and not _name_like_person.match(s) and not looks_unitish(s):
            return s
    # 3) assets map
    if tag in tag_names:
        return tag_names[tag]
    # 4) fallback dict
    if tag in TAG_NAME_FALLBACK:
        return TAG_NAME_FALLBACK[tag]
    # 5)
    return tag


# -------------------------------------------------------------
# Save parsing (very lightweight regex-based)
# -------------------------------------------------------------

@dataclass
class Country:
    tag: str
    raw: Dict[str, object]


def parse_countries_block(save: str) -> Dict[str, Country]:
    """Parse countries={ TAG={ ... } ... } best-effort into dict."""
    out: Dict[str, Country] = {}
    m = re.search(r"countries\s*=\s*\{(.*)\n\}\n", save, flags=re.S)
    if not m:
        return out
    block = m.group(1)
    for m2 in re.finditer(r"\n\s*([A-Z]{3})\s*=\s*\{(.*?)\n\s*\}", block, flags=re.S):
        tag = m2.group(1)
        body = m2.group(2)
        d: Dict[str, object] = {}
        # simple key=value extractions
        for key in (
            "name", "custom_name", "country_name", "long_name", "treasury", "inflation",
            "manpower", "max_manpower", "land_forcelimit", "army_tradition", "army_professionalism",
            "technology_group", "adm_tech", "dip_tech", "mil_tech", "legitimacy", "republican_tradition",
            "horde_unity", "stability", "corruption", "war_exhaustion", "income"
        ):
            mm = re.search(rf"\b{key}\s*=\s*([\-\d\.]+|\"[^\"]*\")", body)
            if mm:
                val = mm.group(1)
                if val.startswith('"') and val.endswith('"'):
                    d[key] = val.strip('"')
                else:
                    try:
                        d[key] = float(val) if ("." in val or "-" in val) else int(val)
                    except Exception:
                        d[key] = val
        # loans count (approx)
        loans_block = re.search(r"loans\s*=\s*\{([^}]*)\}", body, flags=re.S)
        if loans_block:
            d["loans"] = len(re.findall(r"=\s*\{", loans_block.group(1)))
        out[tag] = Country(tag, d)
    return out


def parse_province_owners(save: str) -> Dict[int, str]:
    """Return {province_id: owner_tag}. Best-effort scan of province blocks."""
    owners: Dict[int, str] = {}
    # province= { id=123 owner=TAG ... }
    for m in re.finditer(r"\bid\s*=\s*(\d+)(?:(?!\bid\s*=)[\s\S])*?\bower\s*=\s*([A-Z]{3})", save):
        pid = int(m.group(1))
        owners[pid] = m.group(2)
    return owners


def parse_province_blocks(save: str) -> Dict[int, Dict[str, object]]:
    """Extract minimal province data: base_tax/production/manpower if available."""
    out: Dict[int, Dict[str, object]] = {}
    # Match province blocks roughly: province= { id=123 ... }
    for m in re.finditer(r"province\s*=\s*\{(.*?)\}", save, flags=re.S):
        body = m.group(1)
        mid = re.search(r"\bid\s*=\s*(\d+)", body)
        if not mid:
            continue
        pid = int(mid.group(1))
        d: Dict[str, object] = {}
        for key in ("base_tax", "base_production", "base_manpower", "development", "total_development", "dev", "curr_development"):
            mm = re.search(rf"\b{key}\s*=\s*([\-\d\.]+)", body)
            if mm:
                try:
                    d[key] = float(mm.group(1))
                except Exception:
                    pass
        # Try inside history block
        hist = re.search(r"history\s*=\s*\{([^}]*)\}", body, flags=re.S)
        if hist:
            hb = hist.group(1)
            for key in ("base_tax", "base_production", "base_manpower"):
                mm = re.search(rf"\b{key}\s*=\s*([\-\d\.]+)", hb)
                if mm:
                    try:
                        d[key] = float(mm.group(1))
                    except Exception:
                        pass
        if d:
            out[pid] = d
    return out


def prov_development(p: Dict[str, object]) -> float:
    bt = p.get("base_tax"); bp = p.get("base_production"); bm = p.get("base_manpower")
    try:
        return float(bt) + float(bp) + float(bm)
    except Exception:
        pass
    for k in ("development", "total_development", "dev", "curr_development"):
        v = p.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except Exception:
                pass
    return 0.0


# -------------------------------------------------------------
# Map rendering
# -------------------------------------------------------------

def stable_color_for_tag(tag: str) -> Tuple[int, int, int]:
    # deterministic pastel-ish color
    h = (hash(tag) & 0xFFFFFF) / 0xFFFFFF
    # convert HSV->RGB quickly
    s, v = 0.6, 0.85
    i = int(h * 6)
    f = h * 6 - i
    p = int(255 * v * (1 - s))
    q = int(255 * v * (1 - s * f))
    t = int(255 * v * (1 - s * (1 - f)))
    v255 = int(255 * v)
    i %= 6
    if i == 0:
        return (v255, t, p)
    if i == 1:
        return (q, v255, p)
    if i == 2:
        return (p, v255, t)
    if i == 3:
        return (p, q, v255)
    if i == 4:
        return (t, p, v255)
    return (v255, p, q)


def render_map(
    provinces_bmp: Path,
    defmap: Dict[Tuple[int, int, int], int],
    owners: Dict[int, str],
    sea_ids: set[int],
    tag_colors: Dict[str, Tuple[int, int, int]],
    out_path: Path,
    chunk: int = 64,
    scale: float = 0.75,
) -> None:
    img = Image.open(provinces_bmp).convert("RGB")
    W, H = img.size
    pix = np.array(img)

    # Prepare output
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # Build province-id array (vectorized mapping is tricky; do per-row for memory)
    # We'll also build a quick cache from color->pid to avoid repeated dict lookups
    color_cache: Dict[Tuple[int, int, int], int] = {}

    def color_to_pid_arr(row_rgb: np.ndarray) -> np.ndarray:
        # row_rgb shape: (W, 3)
        pids = np.zeros((row_rgb.shape[0],), dtype=np.int32)
        for x in range(row_rgb.shape[0]):
            r, g, b = int(row_rgb[x, 0]), int(row_rgb[x, 1]), int(row_rgb[x, 2])
            key = (r, g, b)
            pid = color_cache.get(key)
            if pid is None:
                pid = defmap.get(key, 0)
                color_cache[key] = pid
            pids[x] = pid
        return pids

    # Predefine colors
    DEEP = np.array([24, 80, 160], dtype=np.uint8)      # deep ocean
    COAST = np.array([60, 140, 200], dtype=np.uint8)    # coastal sea
    UNOCC = np.array([170, 170, 170], dtype=np.uint8)   # uncolonized

    # Render in chunks
    for y0 in range(0, H, chunk):
        y1 = min(H, y0 + chunk)
        rows = pix[y0:y1, :, :]  # (chunk, W, 3)
        pid_rows = np.zeros((y1 - y0, W), dtype=np.int32)
        for ry in range(rows.shape[0]):
            pid_rows[ry, :] = color_to_pid_arr(rows[ry, :, :])

        # land mask
        is_sea = np.isin(pid_rows, list(sea_ids)) | (pid_rows == 0)
        is_land = ~is_sea

        # coastal sea = sea pixels that touch land in 4-neighborhood
        # use simple np.roll checks
        neigh = np.zeros_like(is_land)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neigh |= np.roll(is_land, shift=dy, axis=0) if dx == 0 else np.roll(is_land, shift=dx, axis=1)
        coastal = (~is_land) & neigh

        # set sea colors
        block = out[y0:y1, :, :]
        block[is_sea] = DEEP
        block[coastal] = COAST

        # color land by owner
        # Build map pid->color for this chunk to reduce dict lookups
        unique_pids = np.unique(pid_rows[is_land])
        pid_color: Dict[int, Tuple[int, int, int]] = {}
        for pid in unique_pids:
            tag = owners.get(int(pid))
            if not tag:
                pid_color[int(pid)] = tuple(UNOCC.tolist())
                continue
            c = tag_colors.get(tag)
            if c is None:
                c = stable_color_for_tag(tag)
                tag_colors[tag] = c
            pid_color[int(pid)] = c

        # apply
        for ry in range(pid_rows.shape[0]):
            row = pid_rows[ry]
            for x in range(W):
                pid = int(row[x])
                if pid in pid_color:
                    block[ry, x] = pid_color[pid]

    # downscale if requested
    if scale and scale != 1.0:
        final_img = Image.fromarray(out, mode="RGB").resize((int(W * scale), int(H * scale)), Image.NEAREST)
    else:
        final_img = Image.fromarray(out, mode="RGB")

    final_img.save(out_path)
    print(f"[OK] Map → {out_path}")


# -------------------------------------------------------------
# CSV writers
# -------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


# -------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("save", help="Path to .eu4 save (text or zipped)")
    ap.add_argument("--assets", default=".", help="Assets directory")
    ap.add_argument("--out", default="world_map.png", help="Output map PNG path")
    ap.add_argument("--chunk", type=int, default=64, help="Row chunk size for rendering")
    ap.add_argument("--scale", type=float, default=0.75, help="Downscale factor for final image")
    args = ap.parse_args(argv)

    assets = Path(args.assets).resolve()
    provinces_bmp = assets / "provinces.bmp"
    definition_csv = assets / "definition.csv"
    default_map = assets / "default.map"
    colors_txt = assets / "00_country_colors.txt"
    out_map = Path(args.out).resolve()

    if not provinces_bmp.exists() or not definition_csv.exists() or not default_map.exists():
        print("[!] Missing required assets (provinces.bmp, definition.csv, default.map)")
        return 2

    # Load assets
    defmap = load_definition_csv(definition_csv)
    sea_ids = load_default_map_sea_ids(default_map.read_text(encoding="latin-1", errors="ignore"))
    tag_colors = load_country_colors(colors_txt)
    tag_names = load_tag_names_csv(assets)

    # Read save
    save_text = read_text_save(Path(args.save))

    # Parse save
    countries = parse_countries_block(save_text)
    owners = parse_province_owners(save_text)
    provs = parse_province_blocks(save_text)

    # Aggregate per-country province counts and development
    prov_count: Dict[str, int] = {}
    prov_dev: Dict[str, float] = {}
    for pid, tag in owners.items():
        prov_count[tag] = prov_count.get(tag, 0) + 1
        d = provs.get(pid)
        if d:
            prov_dev[tag] = prov_dev.get(tag, 0.0) + prov_development(d)

    # Build CSV rows
    rows_overview: List[Dict[str, object]] = []
    rows_econ: List[Dict[str, object]] = []
    rows_mil: List[Dict[str, object]] = []
    rows_dev: List[Dict[str, object]] = []
    rows_tech: List[Dict[str, object]] = []
    rows_leg: List[Dict[str, object]] = []

    tags = set(list(countries.keys()) + list(prov_count.keys()))
    for tag in sorted(tags):
        c = countries.get(tag).raw if tag in countries else {}
        name = country_label(tag, c, tag_names)
        pcount = prov_count.get(tag, 0)
        dtotal = round(prov_dev.get(tag, 0.0), 2)
        davg = round(dtotal / pcount, 2) if pcount else 0.0

        # Overview
        rows_overview.append({
            "tag": tag, "country": name, "name": name,
            "province_count": pcount,
            "total_development": dtotal,
            "avg_development": davg,
            "income": c.get("income", ""),
            "manpower": c.get("manpower", ""),
            "army_quality_score": round(float(c.get("army_tradition", 0)) * 0.6 + float(c.get("army_professionalism", 0)) * 0.4, 3)
                if (isinstance(c.get("army_tradition"), (int, float)) or isinstance(c.get("army_professionalism"), (int, float))) else "",
        })

        # Economy
        rows_econ.append({
            "tag": tag, "country": name, "name": name,
            "income": c.get("income", ""),
            "treasury": c.get("treasury", ""),
            "inflation": c.get("inflation", ""),
            "loans": c.get("loans", ""),
            "interest": "",  # not reliably present in saves
            "war_exhaustion": c.get("war_exhaustion", ""),
            "corruption": c.get("corruption", ""),
        })

        # Military
        rows_mil.append({
            "tag": tag, "country": name, "name": name,
            "army_quality_score": rows_overview[-1]["army_quality_score"],
            "manpower": c.get("manpower", ""),
            "max_manpower": c.get("max_manpower", ""),
            "land_forcelimit": c.get("land_forcelimit", ""),
            "army_tradition": c.get("army_tradition", ""),
            "army_professionalism": c.get("army_professionalism", ""),
        })

        # Development
        rows_dev.append({
            "tag": tag, "country": name, "name": name,
            "province_count": pcount,
            "total_development": dtotal,
            "avg_development": davg,
        })

        # Technology
        rows_tech.append({
            "tag": tag, "country": name, "name": name,
            "adm_tech": c.get("adm_tech", ""),
            "dip_tech": c.get("dip_tech", ""),
            "mil_tech": c.get("mil_tech", ""),
            "technology_group": c.get("technology_group", ""),
        })

        # Legitimacy / gov quality
        rows_leg.append({
            "tag": tag, "country": name, "name": name,
            "legitimacy": c.get("legitimacy", ""),
            "republican_tradition": c.get("republican_tradition", ""),
            "horde_unity": c.get("horde_unity", ""),
            "stability": c.get("stability", ""),
        })

    # Write CSVs
    write_csv(assets / "overview.csv", rows_overview,
              ["tag", "country", "name", "province_count", "total_development", "avg_development", "income", "manpower", "army_quality_score"])
    write_csv(assets / "economy.csv", rows_econ,
              ["tag", "country", "name", "income", "treasury", "inflation", "loans", "interest", "war_exhaustion", "corruption"])
    write_csv(assets / "military.csv", rows_mil,
              ["tag", "country", "name", "army_quality_score", "manpower", "max_manpower", "land_forcelimit", "army_tradition", "army_professionalism"])
    write_csv(assets / "development.csv", rows_dev,
              ["tag", "country", "name", "province_count", "total_development", "avg_development"])
    write_csv(assets / "technology.csv", rows_tech,
              ["tag", "country", "name", "adm_tech", "dip_tech", "mil_tech", "technology_group"])
    write_csv(assets / "legitimacy.csv", rows_leg,
              ["tag", "country", "name", "legitimacy", "republican_tradition", "horde_unity", "stability"])
    print(f"[OK] CSVs → {assets}")

    # Render map
    render_map(
        provinces_bmp=provinces_bmp,
        defmap=defmap,
        owners=owners,
        sea_ids=sea_ids,
        tag_colors=tag_colors,
        out_path=out_map,
        chunk=args.chunk,
        scale=args.scale,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
