#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EU4 Viewer — save inspector & map renderer (low-memory friendly)

Usage example:
    python eu4_viewer.py "<path>/save.eu4" --assets . --out world_map.png --scale 0.75
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

# -------------------------- helpers & defaults -------------------------------

TAG_NAME_FALLBACK: Dict[str, str] = {
    "FRA": "France", "ENG": "England", "GBR": "Great Britain", "CAS": "Castile",
    "SPA": "Spain", "POR": "Portugal", "ARA": "Aragon", "HAB": "Austria",
    "TUR": "Ottomans", "MAM": "Mamluks", "MOS": "Muscovy", "RUS": "Russia",
    "POL": "Poland", "LIT": "Lithuania", "SWE": "Sweden", "NOR": "Norway",
    "DEN": "Denmark", "NED": "Netherlands", "PRU": "Prussia", "BRA": "Brandenburg",
    "SAX": "Saxony", "BOH": "Bohemia", "HUN": "Hungary", "VEN": "Venice",
    "PAP": "Papal State", "NAP": "Naples", "MLO": "Milan", "TUS": "Tuscany",
    "SAV": "Savoy", "MNG": "Ming", "QIN": "Qing", "JAP": "Japan", "KOR": "Korea",
    "VIJ": "Vijayanagar", "USA": "United States", "MEX": "Mexico", "BRZ": "Brazil",
    "CAN": "Canada", "QUE": "Quebec", "LOU": "Louisiana", "AYU": "Ayutthaya",
    "KHM": "Khmer", "DAI": "Dai Viet",
}
UNIT_WORDS = ("regiment", "company", "banner", "cohort", "merc", "fleet", "army", "navy")
_tag3 = re.compile(r"^[A-Z]{3}$")
_name_like_person = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$")


def read_text_save(path: Path) -> str:
    """Read .eu4 (text or zipped). Return latin-1 text."""
    try:
        with zipfile.ZipFile(path) as z:
            name = None
            for n in z.namelist():
                if n.endswith("gamestate"):
                    name = n
                    break
            if name is None:
                name = z.namelist()[0]
            data = z.read(name)
            try:
                return data.decode("latin-1")
            except Exception:
                return data.decode("utf-8", errors="ignore")
    except zipfile.BadZipFile:
        raw = path.read_bytes()
        try:
            return raw.decode("latin-1")
        except Exception:
            return raw.decode("utf-8", errors="ignore")


def parse_numbers_block(text: str, key: str) -> List[int]:
    m = re.search(rf"{re.escape(key)}\s*=\s*\{{([^}}]*)\}}", text)
    if not m:
        return []
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))]


def load_default_map_sea_ids(default_map_text: str) -> set[int]:
    return set(parse_numbers_block(default_map_text, "sea_starts"))


def load_definition_csv(path: Path) -> Dict[int, int]:
    """
    Return packed RGB -> province_id mapping.
    Pack RGB to 24-bit key: (r<<16)|(g<<8)|b
    """
    m: Dict[int, int] = {}
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
            key = (r << 16) | (g << 8) | b
            m[key] = pid
    return m


def load_country_colors(path: Path) -> Dict[str, Tuple[int, int, int]]:
    """Parse 00_country_colors.txt → TAG -> (r,g,b) from color1."""
    if not path.exists():
        return {}
    txt = path.read_text(encoding="latin-1", errors="ignore")
    out: Dict[str, Tuple[int, int, int]] = {}
    for m in re.finditer(r"([A-Z]{3})\s*=\s*\{([^}]*)\}", txt, flags=re.S):
        tag, block = m.group(1), m.group(2)
        c = re.search(r"color1\s*=\s*\{\s*(\d+)\s+(\d+)\s+(\d+)\s*\}", block)
        if c:
            out[tag] = (int(c.group(1)), int(c.group(2)), int(c.group(3)))
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
                mapping[tag] = nm
    return mapping


def looks_unitish(s: str) -> bool:
    s2 = s.lower()
    return any(w in s2 for w in UNIT_WORDS) or ("'s " in s2) or s2.endswith(" regiment")


def country_label(tag: str, c: Dict[str, object], tag_names: Dict[str, str]) -> str:
    # Prefer custom names, then safe 'name', then tag_names.csv, fallback dict, finally tag.
    for k in ("custom_name", "country_name", "long_name"):
        v = c.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = c.get("name")
    if isinstance(v, str):
        s = v.strip()
        if s and not _tag3.fullmatch(s) and not _name_like_person.match(s) and not looks_unitish(s):
            return s
    if tag in tag_names:
        return tag_names[tag]
    if tag in TAG_NAME_FALLBACK:
        return TAG_NAME_FALLBACK[tag]
    return tag


# -------------------------- parsing save -------------------------------------

@dataclass
class Country:
    tag: str
    raw: Dict[str, object]


def parse_countries_block(save: str) -> Dict[str, Country]:
    out: Dict[str, Country] = {}
    m = re.search(r"\bcountries\s*=\s*\{(.*)\n\}\n", save, flags=re.S)
    if not m:
        return out
    block = m.group(1)
    for m2 in re.finditer(r"\n\s*([A-Z]{3})\s*=\s*\{(.*?)\n\s*\}", block, flags=re.S):
        tag, body = m2.group(1), m2.group(2)
        d: Dict[str, object] = {}
        for key in (
            "name", "custom_name", "country_name", "long_name",
            "treasury", "inflation", "income", "corruption",
            "war_exhaustion", "manpower", "max_manpower",
            "land_forcelimit", "army_tradition", "army_professionalism",
            "adm_tech", "dip_tech", "mil_tech", "technology_group",
            "legitimacy", "republican_tradition", "horde_unity", "stability",
        ):
            mm = re.search(rf"\b{key}\s*=\s*([\-\d\.]+|\"[^\"]*\")", body)
            if mm:
                val = mm.group(1)
                if val.startswith('"') and val.endswith('"'):
                    d[key] = val.strip('"')
                else:
                    try:
                        d[key] = float(val) if "." in val else int(val)
                    except Exception:
                        d[key] = val
        loans_block = re.search(r"\bloans\s*=\s*\{([^}]*)\}", body, flags=re.S)
        if loans_block:
            d["loans"] = len(re.findall(r"=\s*\{", loans_block.group(1)))
        out[tag] = Country(tag, d)
    return out


def parse_province_owners(save: str) -> Dict[int, str]:
    """
    Best-effort province owner map. Handles:
      province = { id=123 ... owner=TAG ... }
      id=123 ... owner=TAG ... (within province scope)
    """
    owners: Dict[int, str] = {}
    # Strong form: province block
    for m in re.finditer(r"province\s*=\s*\{(.*?)\}", save, flags=re.S):
        body = m.group(1)
        mid = re.search(r"\bid\s*=\s*(\d+)", body)
        mo = re.search(r"\bower\s*=\s*([A-Z]{3})", body)
        if mid and mo:
            owners[int(mid.group(1))] = mo.group(1)
    if owners:
        return owners
    # Fallback: loose id ... owner pattern
    for m in re.finditer(r"\bid\s*=\s*(\d+)(?:(?!\bid\s*=)[\s\S])*?\bower\s*=\s*([A-Z]{3})", save):
        owners[int(m.group(1))] = m.group(2)
    return owners


def parse_province_blocks(save: str) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for m in re.finditer(r"\bprovince\s*=\s*\{(.*?)\}", save, flags=re.S):
        body = m.group(1)
        mid = re.search(r"\bid\s*=\s*(\d+)", body)
        if not mid:
            continue
        pid = int(mid.group(1))
        d: Dict[str, object] = {}
        for key in ("base_tax", "base_production", "base_manpower",
                    "development", "total_development", "dev", "curr_development"):
            mm = re.search(rf"\b{key}\s*=\s*([\-\d\.]+)", body)
            if mm:
                try:
                    d[key] = float(mm.group(1))
                except Exception:
                    pass
        hist = re.search(r"\bhistory\s*=\s*\{([^}]*)\}", body, flags=re.S)
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
    bt, bp, bm = p.get("base_tax"), p.get("base_production"), p.get("base_manpower")
    if isinstance(bt, (int, float)) and isinstance(bp, (int, float)) and isinstance(bm, (int, float)):
        return float(bt) + float(bp) + float(bm)
    for k in ("development", "total_development", "dev", "curr_development"):
        v = p.get(k)
        try:
            return float(v)  # type: ignore[arg-type]
        except Exception:
            pass
    return 0.0


# ------------------------ color utilities & rendering ------------------------

def stable_color_for_tag(tag: str) -> Tuple[int, int, int]:
    h = (hash(tag) & 0xFFFFFF) / 0xFFFFFF
    s, v = 0.65, 0.9
    i = int(h * 6)
    f = h * 6 - i
    p = int(255 * v * (1 - s))
    q = int(255 * v * (1 - s * f))
    t = int(255 * v * (1 - s * (1 - f)))
    vv = int(255 * v)
    i %= 6
    if i == 0: return (vv, t, p)
    if i == 1: return (q, vv, p)
    if i == 2: return (p, vv, t)
    if i == 3: return (p, q, vv)
    if i == 4: return (t, p, vv)
    return (vv, p, q)


def pack_rgb(arr: np.ndarray) -> np.ndarray:
    """Pack HxWx3 uint8 to HxW uint32 (r<<16|g<<8|b)."""
    arr = arr.astype(np.uint32)
    return (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]


def map_colors_to_pid(packed_rgb: np.ndarray, color_to_pid: Dict[int, int]) -> np.ndarray:
    """Vectorized mapping of packed colors -> province ids (0 when missing)."""
    keys = np.fromiter(color_to_pid.keys(), dtype=np.uint32)
    vals = np.fromiter(color_to_pid.values(), dtype=np.int32)
    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]

    flat = packed_rgb.reshape(-1)
    idx = np.searchsorted(keys, flat)
    match = (idx < keys.size) & (keys[idx] == flat)
    out = np.zeros_like(flat, dtype=np.int32)
    out[match] = vals[idx[match]]
    return out.reshape(packed_rgb.shape)


def render_map(
    provinces_bmp: Path,
    defmap: Dict[int, int],         # packedRGB -> pid
    owners: Dict[int, str],         # pid -> tag
    sea_ids: set[int],
    tag_colors: Dict[str, Tuple[int, int, int]],
    out_path: Path,
    scale: float = 0.75,
) -> None:
    img = Image.open(provinces_bmp).convert("RGB")
    W, H = img.size
    rgb = np.array(img, dtype=np.uint8)           # HxWx3
    packed = pack_rgb(rgb)                        # HxW uint32
    pid = map_colors_to_pid(packed, defmap)       # HxW int32

    if np.all(pid == 0):
        # Definition.csv doesn’t match provinces.bmp (wrong version or mod).
        # Don’t leave a blue rectangle—emit a diagnostic image.
        diag = Image.new("RGB", (W, H), (200, 50, 50))
        diag.save(out_path)
        print("[!] All province IDs resolved to 0. Check that provinces.bmp and definition.csv match.")
        return

    is_sea = np.isin(pid, list(sea_ids)) | (pid == 0)
    is_land = ~is_sea

    # Coastal sea (4-neighborhood touches land)
    coastal = np.zeros_like(is_sea)
    coastal |= np.roll(is_land, 1, axis=0)
    coastal |= np.roll(is_land, -1, axis=0)
    coastal |= np.roll(is_land, 1, axis=1)
    coastal |= np.roll(is_land, -1, axis=1)
    coastal &= ~is_land

    out = np.zeros((H, W, 3), dtype=np.uint8)
    DEEP = np.array([24, 80, 160], dtype=np.uint8)
    COAST = np.array([60, 140, 200], dtype=np.uint8)
    UNOCC = np.array([170, 170, 170], dtype=np.uint8)

    out[is_sea] = DEEP
    out[coastal] = COAST

    # Color land by owner tag
    land_pid = pid[is_land]
    unique_pids = np.unique(land_pid)
    # Build pid -> color
    pid_color: Dict[int, Tuple[int, int, int]] = {}
    for p in unique_pids:
        tag = owners.get(int(p))
        if not tag:
            pid_color[int(p)] = tuple(UNOCC.tolist())
            continue
        c = tag_colors.get(tag)
        if c is None:
            c = stable_color_for_tag(tag)
            tag_colors[tag] = c
        pid_color[int(p)] = c

    # Apply colors
    for p in unique_pids:
        mask = (pid == p)
        c = pid_color[int(p)]
        out[mask] = c

    final = Image.fromarray(out, mode="RGB")
    if scale and scale != 1.0:
        final = final.resize((int(W * scale), int(H * scale)), Image.NEAREST)
    final.save(out_path)
    print(f"[OK] Map → {out_path}")


# ------------------------ CSV output -----------------------------------------

def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


# ------------------------ main pipeline --------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("save", help="Path to .eu4 save (text or zipped)")
    ap.add_argument("--assets", default=".", help="Assets directory")
    ap.add_argument("--out", default="world_map.png", help="Output map PNG")
    ap.add_argument("--scale", type=float, default=0.75, help="Final image scale")
    args = ap.parse_args(argv)

    assets = Path(args.assets).resolve()
    provinces_bmp = assets / "provinces.bmp"
    definition_csv = assets / "definition.csv"
    default_map = assets / "default.map"
    colors_txt = assets / "00_country_colors.txt"

    if not provinces_bmp.exists() or not definition_csv.exists() or not default_map.exists():
        print("[!] Missing required assets (provinces.bmp, definition.csv, default.map)")
        return 2

    # Load assets
    defmap = load_definition_csv(definition_csv)
    sea_ids = load_default_map_sea_ids(default_map.read_text(encoding="latin-1", errors="ignore"))
    tag_colors = load_country_colors(colors_txt)
    tag_names = load_tag_names_csv(assets)

    # Read & parse save
    save_text = read_text_save(Path(args.save))
    countries = parse_countries_block(save_text)
    owners = parse_province_owners(save_text)
    provs = parse_province_blocks(save_text)

    # Aggregate per-country province counts & development
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
        # human-readable country name
        name = country_label(tag, c, tag_names)

        pcount = prov_count.get(tag, 0)
        dtotal = round(prov_dev.get(tag, 0.0), 2)
        davg = round(dtotal / pcount, 2) if pcount else 0.0

        aq = ""
        at, ap = c.get("army_tradition"), c.get("army_professionalism")
        if isinstance(at, (int, float)) or isinstance(ap, (int, float)):
            aq = round(float(at or 0) * 0.6 + float(ap or 0) * 0.4, 3)

        rows_overview.append({
            "tag": tag, "country": name, "name": name,
            "province_count": pcount,
            "total_development": dtotal,
            "avg_development": davg,
            "income": c.get("income", ""),
            "manpower": c.get("manpower", ""),
            "army_quality_score": aq,
        })

        rows_econ.append({
            "tag": tag, "country": name, "name": name,
            "income": c.get("income", ""),
            "treasury": c.get("treasury", ""),
            "inflation": c.get("inflation", ""),
            "loans": c.get("loans", ""),
            "interest": "",
            "war_exhaustion": c.get("war_exhaustion", ""),
            "corruption": c.get("corruption", ""),
        })

        rows_mil.append({
            "tag": tag, "country": name, "name": name,
            "army_quality_score": aq,
            "manpower": c.get("manpower", ""),
            "max_manpower": c.get("max_manpower", ""),
            "land_forcelimit": c.get("land_forcelimit", ""),
            "army_tradition": c.get("army_tradition", ""),
            "army_professionalism": c.get("army_professionalism", ""),
        })

        rows_dev.append({
            "tag": tag, "country": name, "name": name,
            "province_count": pcount,
            "total_development": dtotal,
            "avg_development": davg,
        })

        rows_tech.append({
            "tag": tag, "country": name, "name": name,
            "adm_tech": c.get("adm_tech", ""),
            "dip_tech": c.get("dip_tech", ""),
            "mil_tech": c.get("mil_tech", ""),
            "technology_group": c.get("technology_group", ""),
        })

        rows_leg.append({
            "tag": tag, "country": name, "name": name,
            "legitimacy": c.get("legitimacy", ""),
            "republican_tradition": c.get("republican_tradition", ""),
            "horde_unity": c.get("horde_unity", ""),
            "stability": c.get("stability", ""),
        })

    # Write CSVs next to assets (bot reads from there)
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
        out_path=Path(args.out).resolve(),
        scale=args.scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
