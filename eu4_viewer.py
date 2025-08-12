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
    """Read .eu4 (zip or text). Return text, trying latin-1 then utf-8."""
    data = b""
    try:
        with zipfile.ZipFile(path) as z:
            # Find the main gamestate file within the zip
            gamestate_files = [n for n in z.namelist() if n.endswith("gamestate") or n == "gamestate"]
            name = gamestate_files[0] if gamestate_files else z.namelist()[0]
            data = z.read(name)
    except zipfile.BadZipFile:
        data = path.read_bytes()
    
    try:
        return data.decode("latin-1")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="ignore")


def parse_numbers_block(text: str, key: str) -> List[int]:
    """Extracts a list of integers from a block like `key={ ... }`."""
    m = re.search(rf"\b{re.escape(key)}\s*=\s*\{{([^}}]*)\}}", text, flags=re.IGNORECASE)
    if not m:
        return []
    return [int(x) for x in re.findall(r"\d+", m.group(1))]


def load_default_map_sea_ids(default_map_text: str) -> set[int]:
    """Loads sea province IDs from default.map text."""
    return set(parse_numbers_block(default_map_text, "sea_starts"))


def load_definition_csv(path: Path) -> Dict[int, int]:
    """
    Return packed RGB -> province_id mapping.
    EU4 definition.csv is semicolon-delimited: id;r;g;b;name;...
    """
    mapping: Dict[int, int] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="latin-1", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if not row or not row[0] or not row[0].isdigit():
                continue
            try:
                pid, r, g, b = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                key = (r << 16) | (g << 8) | b
                mapping[key] = pid
            except (ValueError, IndexError):
                continue
    return mapping


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
                tag, nm = parts[0].upper(), ",".join(parts[1:]).strip('"')
                if _tag3.fullmatch(tag):
                    mapping[tag] = nm
    return mapping


def looks_unitish(s: str) -> bool:
    """Heuristic to check if a name string refers to a unit instead of a country."""
    s2 = s.lower()
    return any(w in s2 for w in UNIT_WORDS) or ("'s " in s2)


def get_country_label(tag: str, country_data: Dict[str, object], tag_names: Dict[str, str]) -> str:
    """
    Return a display-friendly country name (never a ruler/unit/tag).
    Priority: custom names -> 'name' if safe -> tag_names.csv -> fallback dict -> TAG
    """
    for key in ("custom_name", "country_name", "long_name"):
        val = country_data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
            
    name_val = country_data.get("name")
    if isinstance(name_val, str):
        s = name_val.strip()
        if s and not _tag3.fullmatch(s) and not _name_like_person.match(s) and not looks_unitish(s):
            return s
            
    return tag_names.get(tag, TAG_NAME_FALLBACK.get(tag, tag))


# -------------------------- parsing save -------------------------------------

def _find_block(text: str, key: str) -> Optional[str]:
    """Finds and returns the content of a {...} block using brace counting."""
    match = re.search(rf"\n{key}\s*=\s*\{{", text)
    if not match:
        return None
    
    start_index = match.end()
    brace_level = 1
    for i in range(start_index, len(text)):
        if text[i] == '{':
            brace_level += 1
        elif text[i] == '}':
            brace_level -= 1
            if brace_level == 0:
                return text[start_index:i]
    return None # Unclosed block

@dataclass
class Country:
    tag: str
    raw: Dict[str, object]

def parse_countries_block(save_text: str) -> Dict[str, Country]:
    """Parses the 'countries' block from the save file."""
    out: Dict[str, Country] = {}
    countries_block = _find_block(save_text, "countries")
    if not countries_block:
        return out

    for m in re.finditer(r"([A-Z]{3})\s*=\s*\{(.*?)\}", countries_block, flags=re.S):
        tag, body = m.group(1), m.group(2)
        data: Dict[str, object] = {}
        
        for item_match in re.finditer(r"\b(\w+)\s*=\s*(\S+)", body):
            key, val_str = item_match.group(1), item_match.group(2)
            if '"' in val_str:
                data[key] = val_str.strip('"')
            else:
                try:
                    data[key] = float(val_str) if "." in val_str else int(val_str)
                except ValueError:
                    data[key] = val_str

        loans_match = re.search(r"\bloans\s*=\s*\{([^}]*)\}", body, flags=re.S)
        if loans_match:
            data["loans"] = len(re.findall(r"\bloan\s*=\s*\{", loans_match.group(1)))
            
        out[tag] = Country(tag, data)
    return out


def parse_province_owners(save_text: str) -> Dict[int, str]:
    """Parses the 'provinces' block to find the owner of each province."""
    owners: Dict[int, str] = {}
    provinces_block = _find_block(save_text, "provinces")
    if not provinces_block:
        print("Warning: 'provinces' block not found in save file.", file=sys.stderr)
        return {}

    # Iterate over province entries. Format is -ID={...owner="TAG"...}
    for m in re.finditer(r"(-(\d+))\s*=\s*\{(.*?)\}", provinces_block, re.S):
        province_id_str, province_body = m.group(2), m.group(3)
        owner_match = re.search(r'owner\s*=\s*"([A-Z]{3})"', province_body)
        if owner_match:
            try:
                pid = int(province_id_str)
                tag = owner_match.group(1)
                owners[pid] = tag
            except ValueError:
                continue
    return owners


def parse_province_blocks(save_text: str) -> Dict[int, Dict[str, object]]:
    """Parses development and other data from province blocks."""
    out: Dict[int, Dict[str, object]] = {}
    provinces_block = _find_block(save_text, "provinces")
    if not provinces_block:
        return {}

    for m in re.finditer(r"(-(\d+))\s*=\s*\{(.*?)\}", provinces_block, re.S):
        pid = int(m.group(2))
        body = m.group(3)
        
        d: Dict[str, object] = {}
        for item_match in re.finditer(r"\b(\w+)\s*=\s*([\d\.]+)", body):
            key, val_str = item_match.group(1), item_match.group(2)
            try:
                d[key] = float(val_str)
            except ValueError:
                pass
        
        hist_match = re.search(r"\bhistory\s*=\s*\{([^}]*)\}", body, flags=re.S)
        if hist_match:
            hist_body = hist_match.group(1)
            for key in ("base_tax", "base_production", "base_manpower"):
                val_match = re.search(rf"\b{key}\s*=\s*([\d\.]+)", hist_body)
                if val_match:
                    try:
                        d[key] = float(val_match.group(1))
                    except ValueError:
                        pass
        if d:
            out[pid] = d
    return out


def get_province_development(p_data: Dict[str, object]) -> float:
    """Calculates total development from province data."""
    for key in ("development", "total_development", "dev", "curr_development"):
        val = p_data.get(key)
        if isinstance(val, (int, float)):
            return float(val)
            
    tax = p_data.get("base_tax", 0.0)
    prod = p_data.get("base_production", 0.0)
    manp = p_data.get("base_manpower", 0.0)
    if isinstance(tax, (int, float)) and isinstance(prod, (int, float)) and isinstance(manp, (int, float)):
        return float(tax) + float(prod) + float(manp)
        
    return 0.0


# --------- battle parsing (best-effort; EU4 save formats vary by version) ----

def parse_battles(save: str, tag_to_name: Dict[str, str]) -> List[Dict[str, object]]:
    """Returns list of battles with casualties/attrition."""
    results: List[Dict[str, object]] = []
    
    for m in re.finditer(r"\b(?:battle|combat)\s*=\s*\{([^}]*)\}", save, flags=re.S):
        body = m.group(1)
        
        def _m(pat: str) -> Optional[str]:
            mm = re.search(pat, body)
            return mm.group(1) if mm else None

        def _num(pat: str) -> Optional[int]:
            mm = re.search(pat, body)
            try:
                return int(mm.group(1)) if mm else None
            except (ValueError, TypeError):
                return None

        attacker = _m(r"\battacker\s*=\s*\"?([A-Z]{3})\"?")
        defender = _m(r"\bdefender\s*=\s*\"?([A-Z]{3})\"?")
        
        if not (attacker and defender):
            continue

        winner = _m(r"\bwinner\s*=\s*\"?([A-Z]{3})\"?")
        date = _m(r'\bdate\s*=\s*"([^"]*)"')
        province_id = _num(r"\bprovince\s*=\s*(\d+)")
        
        a_loss = _num(r"\battacker_(?:losses|casualties)\s*=\s*(\d+)") or 0
        d_loss = _num(r"\bdefender_(?:losses|casualties)\s*=\s*(\d+)") or 0
        a_attr = _num(r"\battacker_attrition(?:_losses)?\s*=\s*(\d+)") or 0
        d_attr = _num(r"\bdefender_attrition(?:_losses)?\s*=\s*(\d+)") or 0

        results.append({
            "date": date or "",
            "province_id": province_id or "",
            "attacker": tag_to_name.get(attacker, attacker),
            "defender": tag_to_name.get(defender, defender),
            "winner": tag_to_name.get(winner, winner or ""),
            "attacker_casualties": a_loss,
            "defender_casualties": d_loss,
            "attacker_attrition": a_attr,
            "defender_attrition": d_attr,
            "total_casualties": a_loss + d_loss,
            "total_attrition": a_attr + d_attr,
        })
        
    return results


# ------------------------ color utilities & rendering ------------------------

def stable_color_for_tag(tag: str) -> Tuple[int, int, int]:
    """Generates a deterministic, visually distinct color for a given tag string."""
    h = (hash(tag) & 0xFFFFFF)
    r = (h >> 16) & 0xFF
    g = (h >> 8) & 0xFF
    b = h & 0xFF
    return (min(255, r + 50), min(255, g + 50), min(255, b + 50))


def pack_rgb(arr: np.ndarray) -> np.ndarray:
    """Pack HxWx3 uint8 to HxW uint32 (r<<16|g<<8|b)."""
    arr = arr.astype(np.uint32)
    return (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]


def map_colors_to_pid(packed_rgb: np.ndarray, color_to_pid: Dict[int, int]) -> np.ndarray:
    """Vectorized mapping of packed colors -> province ids (0 when missing)."""
    unique_colors = np.array(list(color_to_pid.keys()), dtype=np.uint32)
    pids_for_colors = np.array([color_to_pid[c] for c in unique_colors], dtype=np.int32)
    
    sort_indices = np.argsort(unique_colors)
    sorted_colors = unique_colors[sort_indices]
    sorted_pids = pids_for_colors[sort_indices]

    flat_packed = packed_rgb.flatten()
    indices = np.searchsorted(sorted_colors, flat_packed)
    
    out = np.zeros_like(flat_packed, dtype=np.int32)
    
    valid_mask = (indices < len(sorted_colors)) & (sorted_colors[indices] == flat_packed)
    out[valid_mask] = sorted_pids[indices[valid_mask]]
    
    return out.reshape(packed_rgb.shape)


def render_map(
    provinces_bmp: Path,
    defmap: Dict[int, int],
    pid_to_color: Dict[int, Tuple[int, int, int]],
    out_path: Path,
    scale: float = 0.75,
    chunk_height: int | None = None,
) -> None:
    """Renders the final political map."""
    img = Image.open(provinces_bmp).convert("RGB")
    W, H = img.size

    out_img_arr = np.zeros((H, W, 3), dtype=np.uint8)
    band_height = chunk_height if chunk_height and chunk_height > 0 else H

    for y_start in range(0, H, band_height):
        y_end = min(H, y_start + band_height)
        
        band_img = img.crop((0, y_start, W, y_end))
        rgb_band = np.array(band_img, dtype=np.uint8)
        
        packed_band = pack_rgb(rgb_band)
        pid_band = map_colors_to_pid(packed_band, defmap)
        
        r_ch = np.zeros_like(pid_band, dtype=np.uint8)
        g_ch = np.zeros_like(pid_band, dtype=np.uint8)
        b_ch = np.zeros_like(pid_band, dtype=np.uint8)
        
        for pid, (r, g, b) in pid_to_color.items():
            mask = (pid_band == pid)
            r_ch[mask] = r
            g_ch[mask] = g
            b_ch[mask] = b
            
        out_img_arr[y_start:y_end, :, 0] = r_ch
        out_img_arr[y_start:y_end, :, 1] = g_ch
        out_img_arr[y_start:y_end, :, 2] = b_ch

    final_img = Image.fromarray(out_img_arr, mode="RGB")
    if scale and scale != 1.0:
        final_img = final_img.resize((int(W * scale), int(H * scale)), Image.NEAREST)
        
    final_img.save(out_path)


# ------------------------ CSV output -----------------------------------------

def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    """Writes a list of dictionaries to a CSV file."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


# ------------------------ main ------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="EU4 Save Viewer and Map Renderer")
    ap.add_argument("save", help=".eu4 save file (text or zipped)")
    ap.add_argument("--assets", default=".", help="Directory with game assets (provinces.bmp, etc.)")
    ap.add_argument("--out", default="world_map.png", help="Output path for the rendered map (PNG)")
    ap.add_argument("--scale", type=float, default=0.75, help="Downscale factor for the output image")
    ap.add_argument("--chunk", type=int, default=None, help="Row band height for low-memory rendering (e.g., 64)")
    args = ap.parse_args()

    assets = Path(args.assets).resolve()
    provinces_bmp = assets / "provinces.bmp"
    definition_csv = assets / "definition.csv"
    default_map = assets / "default.map"
    country_colors_txt = assets / "00_country_colors.txt"

    for p in [provinces_bmp, definition_csv, default_map]:
        if not p.exists():
            print(f"Error: Missing essential asset file: {p.name}", file=sys.stderr)
            return 2

    defmap = load_definition_csv(definition_csv)
    sea_ids = load_default_map_sea_ids(default_map.read_text(encoding="latin-1", errors="ignore"))
    tag_colors = load_country_colors(country_colors_txt)
    tag_names = load_tag_names_csv(assets)

    save_text = read_text_save(Path(args.save))
    owners = parse_province_owners(save_text)
    countries = parse_countries_block(save_text)
    prov_blocks = parse_province_blocks(save_text)

    tag_to_name: Dict[str, str] = {tag: get_country_label(tag, c.raw, tag_names) for tag, c in countries.items()}

    dev_by_country: Dict[str, float] = {}
    count_by_country: Dict[str, int] = {}
    for pid, tag in owners.items():
        name = tag_to_name.get(tag, tag)
        count_by_country[name] = count_by_country.get(name, 0) + 1
        dev_by_country[name] = dev_by_country.get(name, 0.0) + get_province_development(prov_blocks.get(pid, {}))

    all_rows = []
    for tag, c in countries.items():
        cdata = c.raw
        name = tag_to_name.get(tag, tag)
        prov_count = count_by_country.get(name, 0)
        total_dev = dev_by_country.get(name, 0.0)
        
        aq = 0.0
        tradition = cdata.get("army_tradition", 0.0)
        prof = cdata.get("army_professionalism", 0.0)
        if isinstance(tradition, (int, float)) and isinstance(prof, (int, float)):
            aq = (0.5 * float(tradition)) + (0.5 * float(prof))

        all_rows.append({
            "country": name,
            "province_count": prov_count,
            "total_development": round(total_dev, 2),
            "avg_development": round(total_dev / prov_count, 2) if prov_count else 0.0,
            "income": cdata.get("income", ""),
            "treasury": cdata.get("treasury", ""),
            "inflation": cdata.get("inflation", ""),
            "loans": cdata.get("loans", ""),
            "war_exhaustion": cdata.get("war_exhaustion", ""),
            "corruption": cdata.get("corruption", ""),
            "manpower": cdata.get("manpower", ""),
            "max_manpower": cdata.get("max_manpower", ""),
            "land_forcelimit": cdata.get("land_forcelimit", ""),
            "army_tradition": tradition,
            "army_professionalism": prof,
            "army_quality_score": round(aq, 2) if aq else "",
            "adm_tech": cdata.get("adm_tech", ""),
            "dip_tech": cdata.get("dip_tech", ""),
            "mil_tech": cdata.get("mil_tech", ""),
            "technology_group": cdata.get("technology_group", ""),
            "legitimacy": cdata.get("legitimacy", ""),
            "republican_tradition": cdata.get("republican_tradition", ""),
            "horde_unity": cdata.get("horde_unity", ""),
            "stability": cdata.get("stability", ""),
        })

    battles = parse_battles(save_text, tag_to_name)

    write_csv(assets / "overview.csv", all_rows,
              ["country", "province_count", "total_development", "avg_development", "income", "manpower", "army_quality_score"])
    write_csv(assets / "economy.csv", all_rows,
              ["country", "income", "treasury", "inflation", "loans", "war_exhaustion", "corruption"])
    write_csv(assets / "military.csv", all_rows,
              ["country", "army_quality_score", "manpower", "max_manpower", "land_forcelimit", "army_tradition", "army_professionalism"])
    write_csv(assets / "development.csv", all_rows,
              ["country", "province_count", "total_development", "avg_development"])
    write_csv(assets / "technology.csv", all_rows,
              ["country", "adm_tech", "dip_tech", "mil_tech", "technology_group"])
    write_csv(assets / "legitimacy.csv", all_rows,
              ["country", "legitimacy", "republican_tradition", "horde_unity", "stability"])
    write_csv(assets / "battles.csv", battles,
              ["date", "province_id", "attacker", "defender", "winner", "attacker_casualties", "defender_casualties",
               "attacker_attrition", "defender_attrition", "total_casualties", "total_attrition"])

    pid_to_final_color: Dict[int, Tuple[int, int, int]] = {}
    all_pids = set(defmap.values())
    
    unowned_land_color = (50, 50, 50)
    sea_color = (70, 90, 130)

    for pid in all_pids:
        if pid in sea_ids:
            pid_to_final_color[pid] = sea_color
            continue
        
        owner_tag = owners.get(pid)
        if owner_tag:
            pid_to_final_color[pid] = tag_colors.get(owner_tag, stable_color_for_tag(owner_tag))
        else:
            pid_to_final_color[pid] = unowned_land_color

    render_map(
        provinces_bmp=provinces_bmp,
        defmap=defmap,
        pid_to_color=pid_to_final_color,
        out_path=Path(args.out).resolve(),
        scale=args.scale,
        chunk_height=args.chunk,
    )
    
    print("Processing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
