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
from typing import Dict, List, Optional, Tuple

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
            return val.strip().strip('"')
            
    name_val = country_data.get("name")
    if isinstance(name_val, str):
        s = name_val.strip().strip('"')
        if s and not _tag3.fullmatch(s) and not _name_like_person.match(s) and not looks_unitish(s):
            return s
            
    return tag_names.get(tag, TAG_NAME_FALLBACK.get(tag, tag))


# -------------------------- parsing save -------------------------------------

def _find_block_content(text: str, key: str) -> Optional[str]:
    """Finds and returns the content of a {...} block using brace counting."""
    match = re.search(rf"\b{key}\s*=\s*\{{", text)
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
    return None

@dataclass
class Country:
    tag: str
    raw: Dict[str, object]

def parse_countries_block(save_text: str) -> Dict[str, Country]:
    """Parses the 'countries' block from the save file with robust block handling."""
    out: Dict[str, Country] = {}
    countries_block = _find_block_content(save_text, "countries")
    if not countries_block:
        return out

    # Use a robust iterator that finds the start of each country block
    for m in re.finditer(r'\b([A-Z]{3})\s*=\s*\{', countries_block):
        tag = m.group(1)
        block_start = m.end()
        
        # Find the end of the block by counting braces
        brace_level = 1
        body = ""
        for i in range(block_start, len(countries_block)):
            char = countries_block[i]
            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0:
                    body = countries_block[block_start:i]
                    break
        
        if not body:
            continue

        data: Dict[str, object] = {}
        
        # This regex finds top-level key-value pairs (numbers or quoted strings)
        for item_match in re.finditer(r"^\s*(\w+)\s*=\s*(-?[\d\.]+|\"[^\"]*\")", body, re.MULTILINE):
            key, val_str = item_match.group(1), item_match.group(2).strip('"')
            try:
                data[key] = float(val_str) if "." in val_str else int(val_str)
            except ValueError:
                data[key] = val_str

        # Special handling for nested 'loans' block
        loans_match = re.search(r"loans\s*=\s*\{", body)
        if loans_match:
            loan_body_start = loans_match.end()
            brace_level = 1
            loan_block_end = -1
            for i in range(loan_body_start, len(body)):
                if body[i] == '{': brace_level += 1
                elif body[i] == '}': brace_level -= 1
                if brace_level == 0:
                    loan_block_end = i
                    break
            if loan_block_end != -1:
                loan_block = body[loan_body_start:loan_block_end]
                data["loans"] = len(re.findall(r"loan\s*=\s*\{", loan_block))
        
        out[tag] = Country(tag, data)
    return out


def parse_province_data(save_text: str) -> Tuple[Dict[int, str], Dict[int, Dict[str, object]]]:
    """Parses the 'provinces' block for owners and development data in one pass."""
    owners: Dict[int, str] = {}
    prov_data: Dict[int, Dict[str, object]] = {}
    provinces_block = _find_block_content(save_text, "provinces")
    if not provinces_block:
        print("Warning: 'provinces' block not found in save file.", file=sys.stderr)
        return {}, {}

    # Iterate through each province entry, which starts with -ID={
    for m in re.finditer(r'(-\d+)\s*=\s*\{', provinces_block):
        try:
            pid = int(m.group(1).replace('-', ''))
        except ValueError:
            continue

        # Find the content of this specific province's block using brace counting
        block_start = m.end()
        brace_level = 1
        body = ""
        for i in range(block_start, len(provinces_block)):
            char = provinces_block[i]
            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0:
                    body = provinces_block[block_start:i]
                    break
        
        if not body:
            continue

        # Parse owner from the body (quotes are now optional)
        owner_match = re.search(r'owner\s*=\s*"?([A-Z]{3})"?', body)
        if owner_match:
            owners[pid] = owner_match.group(1)

        # Parse development data from the body
        d: Dict[str, object] = {}
        for item_match in re.finditer(r"(\w+)\s*=\s*(-?[\d\.]+)", body):
            key, val_str = item_match.group(1), item_match.group(2)
            try:
                d[key] = float(val_str)
            except ValueError:
                pass
        
        # Also check inside the history block for development
        hist_match = re.search(r"history\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)
        if hist_match:
            hist_body = hist_match.group(1)
            for key in ("base_tax", "base_production", "base_manpower"):
                val_match = re.search(rf"{key}\s*=\s*(-?[\d\.]+)", hist_body)
                if val_match:
                    try:
                        d[key] = float(val_match.group(1))
                    except ValueError:
                        pass
        if d:
            prov_data[pid] = d
            
    return owners, prov_data


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
    
    # Find the war history block, which contains battles
    war_history_block = _find_block_content(save, "war_history")
    if not war_history_block:
        # Fallback for older save formats
        war_history_block = save

    for m in re.finditer(r"\b(?:battle|combat)\s*=\s*\{([\s\S]*?)\n\t\t\}", war_history_block, flags=re.S):
        body = m.group(1)
        
        def _m(pat: str) -> Optional[str]:
            mm = re.search(pat, body)
            return mm.group(1).strip('"') if mm else None

        def _num(pat: str) -> float:
            s = _m(pat)
            try:
                return float(s) if s else 0.0
            except (ValueError, TypeError):
                return 0.0

        attacker_tag = _m(r"\battacker\s*=\s*\"?([A-Z]{3})\"?")
        defender_tag = _m(r"\bdefender\s*=\s*\"?([A-Z]{3})\"?")
        
        if not (attacker_tag and defender_tag):
            continue

        winner_tag = _m(r"\bresult\s*=\s*\"?(\w+)\"?") # 'yes' for attacker win
        winner = attacker_tag if winner_tag == 'yes' else defender_tag

        # Find losses within the respective combatant blocks
        attacker_losses = 0.0
        defender_losses = 0.0
        
        attacker_block_m = re.search(r"attacker\s*=\s*\{([\s\S]+?)\n\t\t\}", body)
        if attacker_block_m:
            attacker_losses = _num(r"losses\s*=\s*([\d\.]+)")

        defender_block_m = re.search(r"defender\s*=\s*\{([\s\S]+?)\n\t\t\}", body)
        if defender_block_m:
            defender_losses = _num(r"losses\s*=\s*([\d\.]+)")

        results.append({
            "date": _m(r'\bdate\s*=\s*("[\d\.]+")') or "",
            "province_id": _m(r"\bprovince\s*=\s*(\d+)") or "",
            "attacker": tag_to_name.get(attacker_tag, attacker_tag),
            "defender": tag_to_name.get(defender_tag, defender_tag),
            "winner": tag_to_name.get(winner, winner or ""),
            "attacker_casualties": int(attacker_losses),
            "defender_casualties": int(defender_losses),
            "total_casualties": int(attacker_losses + defender_losses),
        })
        
    return results


# ------------------------ color utilities & rendering ------------------------

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
    try:
        img = Image.open(provinces_bmp).convert("RGB")
    except FileNotFoundError:
        print(f"Error: provinces.bmp not found at {provinces_bmp}", file=sys.stderr)
        return
        
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
        
        default_color = (128, 128, 128) # Grey for unspecified
        
        unique_pids_in_band = np.unique(pid_band)
        for pid in unique_pids_in_band:
            if pid == 0: continue
            mask = (pid_band == pid)
            color = pid_to_color.get(pid, default_color)
            r_ch[mask] = color[0]
            g_ch[mask] = color[1]
            b_ch[mask] = color[2]
            
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

    for p in [provinces_bmp, definition_csv, default_map, country_colors_txt]:
        if not p.exists():
            print(f"Error: Missing essential asset file: {p}", file=sys.stderr)
            return 2

    print("Loading assets...")
    defmap = load_definition_csv(definition_csv)
    sea_ids = load_default_map_sea_ids(default_map.read_text(encoding="latin-1", errors="ignore"))
    tag_colors = load_country_colors(country_colors_txt)
    tag_names = load_tag_names_csv(assets)

    print("Reading save file...")
    save_text = read_text_save(Path(args.save))
    
    print("Parsing countries...")
    countries = parse_countries_block(save_text)
    
    print("Parsing provinces...")
    owners, prov_blocks = parse_province_data(save_text)

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
            "income": cdata.get("last_month_income", cdata.get("income", "")),
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

    print("Parsing battles...")
    battles = parse_battles(save_text, tag_to_name)

    print("Writing CSV files...")
    # Write CSVs
    overview_cols = ["country", "province_count", "total_development", "avg_development", "income", "manpower", "army_quality_score"]
    economy_cols = ["country", "income", "treasury", "inflation", "loans", "war_exhaustion", "corruption"]
    military_cols = ["country", "army_quality_score", "manpower", "max_manpower", "land_forcelimit", "army_tradition", "army_professionalism"]
    dev_cols = ["country", "province_count", "total_development", "avg_development"]
    tech_cols = ["country", "adm_tech", "dip_tech", "mil_tech", "technology_group"]
    legitimacy_cols = ["country", "legitimacy", "republican_tradition", "horde_unity", "stability"]
    battles_cols = ["date", "province_id", "attacker", "defender", "winner", "attacker_casualties", "defender_casualties", "total_casualties"]
    
    write_csv(assets / "overview.csv", all_rows, overview_cols)
    write_csv(assets / "economy.csv", all_rows, economy_cols)
    write_csv(assets / "military.csv", all_rows, military_cols)
    write_csv(assets / "development.csv", all_rows, dev_cols)
    write_csv(assets / "technology.csv", all_rows, tech_cols)
    write_csv(assets / "legitimacy.csv", all_rows, legitimacy_cols)
    write_csv(assets / "battles.csv", battles, battles_cols)

    pid_to_final_color: Dict[int, Tuple[int, int, int]] = {}
    
    unowned_land_color = (80, 80, 80)
    sea_color = (70, 90, 130)

    for pid in defmap.values():
        if pid in sea_ids:
            pid_to_final_color[pid] = sea_color
            continue
        
        owner_tag = owners.get(pid)
        if owner_tag:
            pid_to_final_color[pid] = tag_colors.get(owner_tag, unowned_land_color)
        else:
            pid_to_final_color[pid] = unowned_land_color

    print("Rendering map...")
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