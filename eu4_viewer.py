#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EU4 Viewer — save inspector & map renderer (low-memory friendly)

- Renders a colored world map from assets and a .eu4 save
- Emits CSVs with full country names
- Adds a Battles table (battles, casualties, attrition)
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

# -------------------------- defaults & helpers -------------------------------

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
    """Read .eu4 (zip or text). Return latin-1/utf-8 text."""
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
    # typical default.map contains: sea_starts = { ... }
    return set(parse_numbers_block(default_map_text, "sea_starts"))


def load_definition_csv(path: Path) -> Dict[int, int]:
    """
    Return packed RGB -> province_id mapping.
    EU4 definition.csv is semicolon-delimited: id;r;g;b;name;...
    """
    m: Dict[int, int] = {}
    with path.open("r", encoding="latin-1", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if not row or not row[0] or row[0].startswith("#"):
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
    """Optional friendly-name overrides: assets/tag_names.csv (TAG,Name)."""
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
    """
    Always return a *country* name (never a ruler/unit/tag).
    Priority: custom names -> 'name' if safe -> tag_names.csv -> fallback dict -> TAG
    """
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
    for m in re.finditer(r"province\s*=\s*\{(.*?)\}", save, flags=re.S):
        body = m.group(1)
        mid = re.search(r"\bid\s*=\s*(\d+)", body)
        mo = re.search(r"\bower\s*=\s*([A-Z]{3})", body)
        if mid and mo:
            owners[int(mid.group(1))] = mo.group(1)
    if owners:
        return owners
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


# --------- battle parsing (best-effort; EU4 save formats vary by version) ----

def parse_battles(save: str, tag_to_name: Dict[str, str]) -> List[Dict[str, object]]:
    """
    Returns list of battles with casualties/attrition. Tries multiple patterns:
      battle = { attacker = TAG defender = TAG ... attacker_losses = 1234 defender_losses = 567 }
      combat = { ... attacker = TAG defender = TAG ... attacker_casualties = { ... } ... }
    If fields are missing, leaves them blank.
    """
    results: List[Dict[str, object]] = []

    # Pattern 1: compact battle blocks
    for m in re.finditer(r"\bbattle\s*=\s*\{([^}]*)\}", save, flags=re.S):
        body = m.group(1)
        attacker = _m(body, r"\battacker\s*=\s*([A-Z]{3})")
        defender = _m(body, r"\bdefender\s*=\s*([A-Z]{3})")
        winner = _m(body, r"\bwinner\s*=\s*([A-Z]{3})")
        date = _m(body, r"\bdate\s*=\s*([0-9\.\-]+)")
        province_id = _m(body, r"\bprovince\s*=\s*(\d+)")
        a_loss = _num(body, r"\battacker_(?:losses|casualties)\s*=\s*([0-9]+)")
        d_loss = _num(body, r"\bdefender_(?:losses|casualties)\s*=\s*([0-9]+)")
        a_attr = _num(body, r"\battacker_attrition(?:_losses)?\s*=\s*([0-9]+)")
        d_attr = _num(body, r"\bdefender_attrition(?:_losses)?\s*=\s*([0-9]+)")
        if attacker or defender:
            results.append({
                "date": date or "",
                "province_id": int(province_id) if province_id else "",
                "attacker": tag_to_name.get(attacker, attacker or ""),
                "defender": tag_to_name.get(defender, defender or ""),
                "winner": tag_to_name.get(winner, winner or ""),
                "attacker_casualties": a_loss or "",
                "defender_casualties": d_loss or "",
                "attacker_attrition": a_attr or "",
                "defender_attrition": d_attr or "",
                "total_casualties": (a_loss or 0) + (d_loss or 0),
                "total_attrition": (a_attr or 0) + (d_attr or 0),
            })

    # Pattern 2: generic "combat" blocks
    for m in re.finditer(r"\bcombat\s*=\s*\{([^}]*)\}", save, flags=re.S):
        body = m.group(1)
        attacker = _m(body, r"\battacker\s*=\s*([A-Z]{3})")
        defender = _m(body, r"\bdefender\s*=\s*([A-Z]{3})")
        winner = _m(body, r"\bwinner\s*=\s*([A-Z]{3})")
        date = _m(body, r"\bdate\s*=\s*([0-9\.\-]+)")
        province_id = _m(body, r"\bprovince\s*=\s*(\d+)")
        a_loss = _num(body, r"\battacker_(?:losses|casualties)\s*=\s*([0-9]+)")
        d_loss = _num(body, r"\bdefender_(?:losses|casualties)\s*=\s*([0-9]+)")
        a_attr = _num(body, r"\battacker_attrition(?:_losses)?\s*=\s*([0-9]+)")
        d_attr = _num(body, r"\bdefender_attrition(?:_losses)?\s*=\s*([0-9]+)")
        if attacker or defender:
            results.append({
                "date": date or "",
                "province_id": int(province_id) if province_id else "",
                "attacker": tag_to_name.get(attacker, attacker or ""),
                "defender": tag_to_name.get(defender, defender or ""),
                "winner": tag_to_name.get(winner, winner or ""),
                "attacker_casualties": a_loss or "",
                "defender_casualties": d_loss or "",
                "attacker_attrition": a_attr or "",
                "defender_attrition": d_attr or "",
                "total_casualties": (a_loss or 0) + (d_loss or 0),
                "total_attrition": (a_attr or 0) + (d_attr or 0),
            })
    return results


def _m(text: str, pat: str) -> Optional[str]:
    mm = re.search(pat, text)
    return mm.group(1) if mm else None


def _num(text: str, pat: str) -> Optional[int]:
    mm = re.search(pat, text)
    try:
        return int(mm.group(1)) if mm else None
    except Exception:
        return None


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
    chunk: int | None = None,       # process rows in bands
) -> None:
    img = Image.open(provinces_bmp).convert("RGB")
    W, H = img.size

    def color_for_pid(pid: int) -> Tuple[int, int, int]:
        if pid in sea_ids:
            return (70, 90, 130)  # sea/ogre-blue
        tag = owners.get(pid)
        if not tag:
            return (30, 30, 30)
        return tag_colors.get(tag, stable_color_for_tag(tag))

    out = np.zeros((H, W, 3), dtype=np.uint8)
    band = max(1, int(chunk)) if chunk else H

    y = 0
    while y < H:
        y1 = min(H, y + band)
        rgb = np.array(img.crop((0, y, W, y1)), dtype=np.uint8)   # h x W x 3
        packed = pack_rgb(rgb)                                    # h x W
        pid = map_colors_to_pid(packed, defmap)                   # h x W

        band_pids = np.unique(pid)
        lut: Dict[int, Tuple[int, int, int]] = {int(p): color_for_pid(int(p)) for p in band_pids}
        r = np.zeros_like(pid, dtype=np.uint8)
        g = np.zeros_like(pid, dtype=np.uint8)
        b = np.zeros_like(pid, dtype=np.uint8)
        for p, (rr, gg, bb) in lut.items():
            mask = (pid == p)
            r[mask] = rr
            g[mask] = gg
            b[mask] = bb
        out[y:y1, :, 0] = r
        out[y:y1, :, 1] = g
        out[y:y1, :, 2] = b

        y = y1

    final = Image.fromarray(out, mode="RGB")
    if scale and scale != 1.0:
        final = final.resize((int(W * scale), int(H * scale)), Image.NEAREST)
    final.save(out_path)


# ------------------------ CSV output -----------------------------------------

def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


# ------------------------ main ------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("save", help=".eu4 save file (text or zipped)")
    ap.add_argument("--assets", default=".", help="Folder containing provinces.bmp, definition.csv, default.map, 00_country_colors.txt")
    ap.add_argument("--out", default="world_map.png", help="Output map path (PNG)")
    ap.add_argument("--scale", type=float, default=0.75, help="Downscale factor for output image")
    ap.add_argument("--chunk", type=int, default=None, help="Row band height for low-memory rendering (e.g., 64)")
    args = ap.parse_args()

    assets = Path(args.assets).resolve()
    provinces_bmp = assets / "provinces.bmp"
    definition_csv = assets / "definition.csv"
    default_map = assets / "default.map"
    country_colors = assets / "00_country_colors.txt"

    # Basic existence checks
    required = [provinces_bmp, definition_csv, default_map]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        # Quiet failure: caller (bot) will translate to a friendly message
        return 2

    # Load assets
    defmap = load_definition_csv(definition_csv)
    sea_ids = load_default_map_sea_ids(default_map.read_text(encoding="latin-1", errors="ignore"))
    tag_colors = load_country_colors(country_colors)
    tag_names = load_tag_names_csv(assets)

    # Load & parse save
    save_text = read_text_save(Path(args.save))
    owners = parse_province_owners(save_text)
    countries = parse_countries_block(save_text)
    prov_blocks = parse_province_blocks(save_text)

    # Build tag -> friendly name table once
    tag_to_name: Dict[str, str] = {tag: country_label(tag, c.raw, tag_names) for tag, c in countries.items()}

    # Per-country stats (always use friendly names)
    dev_by_country: Dict[str, float] = {}
    count_by_country: Dict[str, int] = {}
    for pid, tag in owners.items():
        name = tag_to_name.get(tag, TAG_NAME_FALLBACK.get(tag, tag))
        count_by_country[name] = count_by_country.get(name, 0) + 1
        dev_by_country[name] = dev_by_country.get(name, 0.0) + prov_development(prov_blocks.get(pid, {}))

    rows_overview: List[Dict[str, object]] = []
    rows_econ: List[Dict[str, object]] = []
    rows_mil: List[Dict[str, object]] = []
    rows_dev: List[Dict[str, object]] = []
    rows_tech: List[Dict[str, object]] = []
    rows_leg: List[Dict[str, object]] = []

    for tag, c in countries.items():
        cdata = c.raw
        name = tag_to_name.get(tag, TAG_NAME_FALLBACK.get(tag, tag))
        prov_count = count_by_country.get(name, 0)
        total_dev = dev_by_country.get(name, 0.0)
        avg_dev = (total_dev / prov_count) if prov_count else 0.0

        # simple composite for army quality if fields exist
        aq = 0.0
        for k, wt in (("army_tradition", 0.5), ("army_professionalism", 0.5)):
            v = cdata.get(k)
            if isinstance(v, (int, float)):
                aq += wt * float(v)

        rows_overview.append({
            "country": name,
            "province_count": prov_count,
            "total_development": round(total_dev, 2),
            "avg_development": round(avg_dev, 2),
            "income": cdata.get("income", ""),
            "manpower": cdata.get("manpower", ""),
            "army_quality_score": round(aq, 2) if aq else "",
        })

        rows_econ.append({
            "country": name,
            "income": cdata.get("income", ""),
            "treasury": cdata.get("treasury", ""),
            "inflation": cdata.get("inflation", ""),
            "loans": cdata.get("loans", ""),
            "war_exhaustion": cdata.get("war_exhaustion", ""),
            "corruption": cdata.get("corruption", ""),
        })

        rows_mil.append({
            "country": name,
            "army_quality_score": round(aq, 2) if aq else "",
            "manpower": cdata.get("manpower", ""),
            "max_manpower": cdata.get("max_manpower", ""),
            "land_forcelimit": cdata.get("land_forcelimit", ""),
            "army_tradition": cdata.get("army_tradition", ""),
            "army_professionalism": cdata.get("army_professionalism", ""),
        })

        rows_dev.append({
            "country": name,
            "province_count": prov_count,
            "total_development": round(total_dev, 2),
            "avg_development": round(avg_dev, 2),
        })

        rows_tech.append({
            "country": name,
            "adm_tech": cdata.get("adm_tech", ""),
            "dip_tech": cdata.get("dip_tech", ""),
            "mil_tech": cdata.get("mil_tech", ""),
            "technology_group": cdata.get("technology_group", ""),
        })

        rows_leg.append({
            "country": name,
            "legitimacy": cdata.get("legitimacy", ""),
            "republican_tradition": cdata.get("republican_tradition", ""),
            "horde_unity": cdata.get("horde_unity", ""),
            "stability": cdata.get("stability", ""),
        })

    # Battles (best-effort)
    battles = parse_battles(save_text, tag_to_name)

    # Write CSVs next to assets
    write_csv(assets / "overview.csv", rows_overview,
              ["country", "province_count", "total_development", "avg_development", "income", "manpower", "army_quality_score"])
    write_csv(assets / "economy.csv", rows_econ,
              ["country", "income", "treasury", "inflation", "loans", "war_exhaustion", "corruption"])
    write_csv(assets / "military.csv", rows_mil,
              ["country", "army_quality_score", "manpower", "max_manpower", "land_forcelimit", "army_tradition", "army_professionalism"])
    write_csv(assets / "development.csv", rows_dev,
              ["country", "province_count", "total_development", "avg_development"])
    write_csv(assets / "technology.csv", rows_tech,
              ["country", "adm_tech", "dip_tech", "mil_tech", "technology_group"])
    write_csv(assets / "legitimacy.csv", rows_leg,
              ["country", "legitimacy", "republican_tradition", "horde_unity", "stability"])
    write_csv(assets / "battles.csv", battles,
              ["date", "province_id", "attacker", "defender", "winner", "attacker_casualties", "defender_casualties",
               "attacker_attrition", "defender_attrition", "total_casualties", "total_attrition"])

    # Render map
    # Owners dict currently keyed by pid->tag; we still need tag->color mapping, so keep tags here.
    render_map(
        provinces_bmp=provinces_bmp,
        defmap=defmap,
        owners=owners,
        sea_ids=sea_ids,
        tag_colors=load_country_colors(country_colors),
        out_path=Path(args.out).resolve(),
        scale=args.scale,
        chunk=args.chunk,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
