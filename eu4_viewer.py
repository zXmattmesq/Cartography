#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, json, os, random, re, zipfile, csv
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw

# ======= Visuals (same as your working version) =======
DEEP_BLUE   = (  0,  40, 104)   # deep ocean
COAST_BLUE  = ( 62, 122, 189)   # coastal water / lakes
GREY_UNOWN  = (140, 140, 140)   # unowned / uncolonized

def parse_args():
    p = argparse.ArgumentParser(description="Render EU4 political map and export multi-sheet stats from a save.")
    p.add_argument("save", help="Path to .eu4 save (ZIP/Ironman or plain text).")
    p.add_argument("--assets", default=".", help="Folder with provinces.bmp, definition.csv, default.map, 00_country_colors.txt.")
    p.add_argument("--out", default="world_map.png", help="Output PNG filename.")
    return p.parse_args()

def make_paths(assets_dir: str):
    ad = os.path.abspath(assets_dir)
    return {
        "bmp": os.path.join(ad, "provinces.bmp"),
        "defcsv": os.path.join(ad, "definition.csv"),
        "pixmap": os.path.join(ad, "province_pixel_map.json"),
        "colors": os.path.join(ad, "00_country_colors.txt"),
        "defaultmap": os.path.join(ad, "default.map"),
        # stat outputs
        "overview": os.path.join(ad, "overview.csv"),
        "economy": os.path.join(ad, "economy.csv"),
        "military": os.path.join(ad, "military.csv"),
        "development": os.path.join(ad, "development.csv"),
        "technology": os.path.join(ad, "technology.csv"),
        "legitimacy": os.path.join(ad, "legitimacy.csv"),
        "xlsx": os.path.join(ad, "eu4_stats.xlsx"),
    }

# ======= definition.csv → (R,G,B)→province_id ; province pixel map =======
def read_definition_colors(csv_path: str) -> Dict[Tuple[int,int,int], int]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"definition.csv not found at {csv_path}")
    mapping: Dict[Tuple[int,int,int], int] = {}
    with open(csv_path, "r", encoding="latin-1", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"): continue
            parts = line.split(";") if ";" in line else line.split(",")
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    pid, r, g, b = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    mapping[(r,g,b)] = pid
                except ValueError:
                    pass
    if not mapping:
        raise ValueError("Parsed definition.csv but found no color rows.")
    return mapping

def generate_pixel_map(bmp_path: str, csv_path: str, out_json: str) -> None:
    print("[+] Generating province_pixel_map.json (first run may take a minute)...")
    color2prov = read_definition_colors(csv_path)
    img = Image.open(bmp_path).convert("RGB")
    w, h = img.size
    px = img.load()
    prov_pixels: Dict[int, List[List[int]]] = {}
    for y in range(h):
        for x in range(w):
            pid = color2prov.get(px[x, y])
            if pid is not None:
                prov_pixels.setdefault(pid, []).append([x, y])
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(prov_pixels, f, separators=(",", ":"))
    print(f"[✓] Saved {out_json} with {len(prov_pixels)} provinces.")

def load_pixel_map(pixmap_path: str, bmp_path: str, csv_path: str) -> Dict[str, List[List[int]]]:
    if not os.path.exists(pixmap_path):
        generate_pixel_map(bmp_path, csv_path, pixmap_path)
    with open(pixmap_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ======= save loading =======
def load_save_text(save_path: str) -> str:
    with open(save_path, "rb") as f:
        magic = f.read(2)
    if magic == b"PK":
        print("[+] Detected ZIP/Ironman save; reading gamestate...")
        with zipfile.ZipFile(save_path) as zf:
            if "gamestate" not in zf.namelist():
                raise FileNotFoundError("ZIP save has no 'gamestate'.")
            return zf.read("gamestate").decode("latin-1", errors="ignore")
    if magic == b"\x1f\x8b":
        print("[+] Detected GZIP-compressed save; decompressing...")
        with gzip.open(save_path, "rt", encoding="latin-1", errors="ignore") as g:
            return g.read()
    print("[+] Detected uncompressed text save.")
    with open(save_path, "rt", encoding="latin-1", errors="ignore") as t:
        return t.read()

# ======= brace helpers =======
def _slice_block(text: str, brace_idx: int) -> tuple[str, int]:
    assert text[brace_idx] == "{"
    depth, i = 1, brace_idx + 1
    while i < len(text) and depth:
        c = text[i]
        if c == "{": depth += 1
        elif c == "}": depth -= 1
        i += 1
    return text[brace_idx+1:i-1], i

def _find_block(text: str, key: str) -> Optional[str]:
    m = re.search(rf"\b{re.escape(key)}\s*=\s*\{{", text)
    if not m: return None
    block, _ = _slice_block(text, m.end() - 1)
    return block

# ======= owners (two strategies) =======
def extract_owners_via_provinces_block(save_text: str) -> Dict[str, str]:
    block = _find_block(save_text, "provinces")
    if not block: return {}
    owners: Dict[str, str] = {}
    i, n = 0, len(block)
    while i < n:
        if block[i].isspace(): i += 1; continue
        m_id = re.match(r"(\d+)\s*=\s*\{", block[i:])
        if not m_id: i += 1; continue
        prov_id = m_id.group(1)
        sub, i_after = _slice_block(block, i + m_id.end() - 1)
        m_owner = re.search(r'owner\s*=\s*"?([A-Za-z0-9]{3})"?', sub)
        if m_owner:
            owners[prov_id] = m_owner.group(1).upper()
        i = i_after
    return owners

def extract_owners_via_countries_owned(save_text: str) -> Dict[str, str]:
    cblock = _find_block(save_text, "countries")
    if not cblock: return {}
    owners: Dict[str, str] = {}
    pos, n = 0, len(cblock)
    while pos < n:
        m_tag = re.search(r'\b([A-Za-z0-9]{3})\s*=\s*\{', cblock[pos:])
        if not m_tag: break
        tag = m_tag.group(1).upper()
        start = pos + m_tag.end() - 1
        tblock, new_pos = _slice_block(cblock, start)
        m_owned = re.search(r'owned_provinces\s*=\s*\{', tblock)
        if m_owned:
            lst, _ = _slice_block(tblock, m_owned.end() - 1)
            for num in re.findall(r"\d+", lst):
                owners[num] = tag
        pos = new_pos
    return owners

def extract_province_owners(save_text: str) -> Dict[str, str]:
    owners = extract_owners_via_provinces_block(save_text)
    if owners:
        print(f"[+] Owners via provinces block: {len(owners)} provinces")
        return owners
    owners = extract_owners_via_countries_owned(save_text)
    print(f"[+] Owners via countries/owned_provinces: {len(owners)} provinces")
    return owners

# ======= province development =======
def extract_province_development(save_text: str) -> Dict[str, int]:
    block = _find_block(save_text, "provinces")
    if not block:
        print("[!] No provinces block found; development totals unavailable.")
        return {}
    prov_dev: Dict[str, int] = {}
    i, n = 0, len(block)
    while i < n:
        if block[i].isspace(): i += 1; continue
        m_id = re.match(r"(\d+)\s*=\s*\{", block[i:])
        if not m_id: i += 1; continue
        prov_id = m_id.group(1)
        sub, i_after = _slice_block(block, i + m_id.end() - 1)
        def pick_int(key: str) -> int:
            m = re.search(rf'\b{key}\s*=\s*(-?\d+)', sub)
            return int(m.group(1)) if m else 0
        dev = pick_int("base_tax") + pick_int("base_production") + pick_int("base_manpower")
        prov_dev[prov_id] = dev
        i = i_after
    print(f"[+] Parsed development for {len(prov_dev)} provinces")
    return prov_dev

# ======= country colors (save overrides first, then file) =======
def load_country_colors_file(color_file_path: str) -> Dict[str, Tuple[int,int,int]]:
    colors: Dict[str, Tuple[int,int,int]] = {}
    if not os.path.exists(color_file_path):
        print("[!] 00_country_colors.txt not found; relying on save overrides or deterministic colors.")
        return colors
    current_tag: str | None = None
    c1 = c3 = c = None
    def triplet(line: str):
        nums = re.findall(r"\d+", line)
        return (int(nums[0]), int(nums[1]), int(nums[2])) if len(nums) >= 3 else None
    with open(color_file_path, "r", encoding="latin-1", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"): continue
            if "=" in line and "{" in line and not line.lower().startswith(("color", "color1", "color2", "color3")):
                if current_tag:
                    picked = c3 or c1 or c
                    if picked: colors[current_tag] = picked
                left = line.split("=")[0].strip()
                current_tag = left.upper() if len(left) == 3 and left.isalnum() else None
                c1 = c3 = c = None
                continue
            if current_tag:
                low = line.lower()
                if low.startswith("color3"): c3 = triplet(line) or c3
                elif low.startswith("color1"): c1 = triplet(line) or c1
                elif low.startswith("color"):  c  = triplet(line) or c
                elif line.startswith("}"):
                    picked = c3 or c1 or c
                    if picked: colors[current_tag] = picked
                    current_tag = None; c1 = c3 = c = None
    if current_tag:
        picked = c3 or c1 or c
        if picked: colors[current_tag] = picked
    print(f"[+] Loaded {len(colors)} colors from 00_country_colors.txt (pref: color3>color1>color)")
    return colors

def extract_country_colors_from_save(save_text: str) -> Dict[str, Tuple[int,int,int]]:
    cblock = _find_block(save_text, "countries")
    if not cblock: return {}
    colors: Dict[str, Tuple[int,int,int]] = {}
    pos, n = 0, len(cblock)
    def triplet(s: str):
        nums = re.findall(r"\d+", s)
        return (int(nums[0]), int(nums[1]), int(nums[2])) if len(nums) >= 3 else None
    while pos < n:
        m_tag = re.search(r'\b([A-Za-z0-9]{3})\s*=\s*\{', cblock[pos:])
        if not m_tag: break
        tag = m_tag.group(1).upper()
        start = pos + m_tag.end() - 1
        tblock, new_pos = _slice_block(cblock, start)
        mc = re.search(r'\bmap_color\s*=\s*\{[^}]*\}', tblock)
        col = re.search(r'(?<!map_)color\s*=\s*\{[^}]*\}', tblock)
        picked = triplet(mc.group(0)) if mc else None
        if not picked and col: picked = triplet(col.group(0))
        if picked: colors[tag] = picked
        pos = new_pos
    if colors:
        print(f"[+] Loaded {len(colors)} country colors from SAVE (map_color/color)")
    return colors

def deterministic_color(tag: str) -> Tuple[int,int,int]:
    rng = random.Random(tag)
    return (rng.randint(40, 210), rng.randint(40, 210), rng.randint(40, 210))

# ======= water IDs =======
def _parse_id_block(text: str, key: str) -> set[int]:
    m = re.search(rf"\b{re.escape(key)}\s*=\s*\{{", text)
    ids: set[int] = set()
    if not m: return ids
    block, _ = _slice_block(text, m.end() - 1)
    for num in re.findall(r"\d+", block): ids.add(int(num))
    return ids

def load_water_ids(default_map_path: str) -> tuple[set[int], set[int]]:
    if not os.path.exists(default_map_path):
        raise FileNotFoundError("default.map is required for water coloring.")
    with open(default_map_path, "r", encoding="latin-1", errors="ignore") as f:
        text = f.read()
    seas  = _parse_id_block(text, "sea_starts")
    lakes = _parse_id_block(text, "lakes")
    print(f"[+] Water IDs: seas={len(seas)} lakes={len(lakes)}")
    return seas, lakes

# ======= neighbor grid for coast detection =======
def build_id_grid(pixel_map: Dict[str, List[List[int]]], bmp_path: str) -> tuple[list[list[int]], int, int]:
    img = Image.open(bmp_path)
    w, h = img.size
    grid = [[-1]*h for _ in range(w)]
    for prov_id_str, coords in pixel_map.items():
        pid = int(prov_id_str)
        for x, y in coords:
            if 0 <= x < w and 0 <= y < h:
                grid[x][y] = pid
    return grid, w, h

def is_coastal_water(pid: int, coords: List[List[int]], grid: list[list[int]], water_ids: set[int], w: int, h: int) -> bool:
    for x, y in coords:
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h:
                nid = grid[nx][ny]
                if nid != -1 and nid not in water_ids:
                    return True
    return False

# ======= draw map =======
def draw_map(
    bmp_path: str,
    pixel_map: Dict[str, List[List[int]]],
    owners: Dict[str, str],
    tag_colors: Dict[str, Tuple[int,int,int]],
    out_path: str,
    sea_ids: set[int],
    lake_ids: set[int],
):
    base = Image.open(bmp_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    water_ids = set(sea_ids) | set(lake_ids)
    grid, w, h = build_id_grid(pixel_map, bmp_path)

    # Paint water
    for prov_id_str, coords in pixel_map.items():
        pid = int(prov_id_str)
        if pid not in water_ids: continue
        coastal = is_coastal_water(pid, coords, grid, water_ids, w, h)
        color = COAST_BLUE if coastal else DEEP_BLUE
        for x, y in coords: draw.point((x, y), fill=color)

    # Paint land
    counts = Counter()
    for prov_id_str, coords in pixel_map.items():
        pid = int(prov_id_str)
        if pid in water_ids: continue
        tag = owners.get(prov_id_str)
        if not tag:
            for x, y in coords: draw.point((x, y), fill=GREY_UNOWN)
            continue
        color = tag_colors.get(tag, deterministic_color(tag))
        for x, y in coords: draw.point((x, y), fill=color)
        counts[tag] += 1

    base.save(out_path)
    print(f"[✓] Saved map → {out_path}")

    if counts:
        print("\nTop 10 by province count (map):")
        for tag, c in counts.most_common(10):
            rgb = tag_colors.get(tag, deterministic_color(tag))
            print(f"  {tag:>3}  {c:4d}  color={rgb}")

# ======= country stats extraction =======
def _get_num(tblock: str, key: str) -> Optional[float]:
    m = re.search(rf'\b{re.escape(key)}\s*=\s*(-?\d+(\.\d+)?)', tblock)
    return float(m.group(1)) if m else None

def extract_country_stats(save_text: str) -> Dict[str, dict]:
    """Grab a wide set of country-level stats; all fields optional."""
    cblock = _find_block(save_text, "countries")
    if not cblock:
        print("[!] No countries block found for stats.")
        return {}

    fields_common = [
        "stability", "prestige", "corruption", "war_exhaustion",
        "manpower","max_manpower","treasury","income",
        "army_professionalism","army_tradition","navy_tradition",
        "land_forcelimit","naval_forcelimit",
        "inflation","loans","interest","government_reform_progress",
    ]
    # “quality-ish”
    fields_quality = [
        "discipline",        # often a modifier on tag (0.05 => +5%)
        "land_morale",       # current land morale (might be “morale” in older saves)
        "morale",            # seen in some versions
    ]
    # governance flavors
    fields_legit = [
        "legitimacy","republican_tradition","devotion","horde_unity",
        "meritocracy","absolutism"
    ]
    # tech
    fields_tech = ["adm_tech", "dip_tech", "mil_tech"]

    stats: Dict[str, dict] = {}
    pos, n = 0, len(cblock)
    while pos < n:
        m_tag = re.search(r'\b([A-Za-z0-9]{3})\s*=\s*\{', cblock[pos:])
        if not m_tag: break
        tag = m_tag.group(1).upper()
        start = pos + m_tag.end() - 1
        tblock, new_pos = _slice_block(cblock, start)

        row = {"tag": tag}
        m_name = re.search(r'\bname\s*=\s*"(.*?)"', tblock)
        row["name"] = m_name.group(1) if m_name else tag

        for fld in (fields_common + fields_quality + fields_legit + fields_tech):
            row[fld] = _get_num(tblock, fld)

        # Compute “army_quality_score” (normalized blend; safe even if some Nones)
        # We treat discipline like 1.05 -> +5% (add 1.0 baseline if <= 0.5)
        disc = row.get("discipline")
        if disc is not None and disc < 0.5:  # 0.05 -> 1.05
            disc = 1.0 + disc
        trad = (row.get("army_tradition") or 0.0) / 100.0
        prof = (row.get("army_professionalism") or 0.0) / 100.0
        morale = row.get("land_morale") if row.get("land_morale") is not None else row.get("morale")
        # Normalize morale against a reasonable late-game cap (e.g., 7.0)
        morale_norm = (morale or 0.0) / 7.0
        disc_norm = (disc or 1.0) / 1.25  # 1.25 ~ 125% discipline “soft cap”

        # Weighted blend — tweak weights if you like
        aq = 0.35*disc_norm + 0.25*morale_norm + 0.20*trad + 0.20*prof
        row["army_quality_score"] = aq

        stats[tag] = row
        pos = new_pos

    print(f"[+] Collected wide stats for {len(stats)} countries")
    return stats

# ======= dev aggregation =======
def aggregate_development_by_owner(prov_dev: Dict[str,int], owners: Dict[str,str]) -> Dict[str,int]:
    by_tag: Dict[str,int] = defaultdict(int)
    for pid, dev in prov_dev.items():
        tag = owners.get(pid)
        if tag: by_tag[tag] += dev
    return by_tag

# ======= writers: CSV + optional XLSX (multi-sheet) =======
def write_csv(path: str, rows: List[dict], cols: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"[✓] Wrote {os.path.basename(path)}")

def try_write_xlsx(path_xlsx: str, sheets: Dict[str, tuple[List[dict], List[str]]]):
    try:
        import pandas as pd
        with pd.ExcelWriter(path_xlsx, engine="xlsxwriter") as writer:
            for sheet_name, (rows, cols) in sheets.items():
                df = pd.DataFrame(rows, columns=cols)
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f"[✓] Also wrote Excel workbook → {path_xlsx}")
    except Exception as e:
        print(f"[i] Skipped Excel export (install pandas+xlsxwriter to enable). Reason: {e}")

# ======= main =======
def main():
    args = parse_args()
    ap = make_paths(args.assets)

    # Required assets for map & water
    for label, p in [("provinces.bmp", ap["bmp"]), ("definition.csv", ap["defcsv"]), ("default.map", ap["defaultmap"])]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {label}: {p}")

    # Pixel map
    pixmap = load_pixel_map(ap["pixmap"], ap["bmp"], ap["defcsv"])

    # Save text
    save_text = load_save_text(args.save)

    # Owners for map + dev aggregation
    owners = extract_province_owners(save_text)
    if not owners:
        print("[!] No owners parsed; exiting.")
        return

    # Colors
    colors_file = load_country_colors_file(ap["colors"])
    colors_save = extract_country_colors_from_save(save_text)
    tags = set(owners.values())
    tag_colors: Dict[str, Tuple[int,int,int]] = {}
    save_hits = file_hits = 0
    for t in tags:
        if t in colors_save:
            tag_colors[t] = colors_save[t]; save_hits += 1
        elif t in colors_file:
            tag_colors[t] = colors_file[t]; file_hits += 1
        else:
            tag_colors[t] = deterministic_color(t)
    print(f"[i] Color sources — from SAVE: {save_hits}, from file: {file_hits}, fallback: {len(tags)-save_hits-file_hits}")

    # Water
    sea_ids, lake_ids = load_water_ids(ap["defaultmap"])

    # Draw map
    draw_map(ap["bmp"], pixmap, owners, tag_colors, args.out, sea_ids, lake_ids)

    # ===== Stats build =====
    # Province development
    prov_dev = extract_province_development(save_text)
    dev_by_tag = aggregate_development_by_owner(prov_dev, owners) if prov_dev else {}
    prov_count_by_tag = Counter(owners.values())

    # Country stats (very wide)
    cstats = extract_country_stats(save_text)

    # Build per-category rows + columns
    all_tags = set(prov_count_by_tag.keys()) | set(cstats.keys()) | set(dev_by_tag.keys())

    # common label map to reduce duplication
    names = {t: (cstats.get(t, {}).get("name") if t in cstats else t) for t in all_tags}

    # Development sheet
    dev_rows, dev_cols = [], ["tag","name","province_count","total_development","avg_development"]
    for t in sorted(all_tags):
        total = dev_by_tag.get(t)
        cnt = prov_count_by_tag.get(t, 0)
        dev_rows.append({
            "tag": t,
            "name": names[t],
            "province_count": cnt,
            "total_development": (total if total is not None else None),
            "avg_development": (round(total/cnt, 2) if total is not None and cnt > 0 else None)
        })

    # Economy sheet
    econ_fields = ["income","treasury","inflation","loans","interest","war_exhaustion","corruption"]
    econ_rows, econ_cols = [], ["tag","name"] + econ_fields
    for t in sorted(all_tags):
        row = {"tag": t, "name": names[t]}
        src = cstats.get(t, {})
        for k in econ_fields: row[k] = src.get(k)
        econ_rows.append(row)

    # Military sheet
    mil_fields = [
        "manpower","max_manpower","land_forcelimit","naval_forcelimit",
        "army_professionalism","army_tradition","navy_tradition",
        "discipline","land_morale","morale","army_quality_score"
    ]
    mil_rows, mil_cols = [], ["tag","name"] + mil_fields
    for t in sorted(all_tags):
        row = {"tag": t, "name": names[t]}
        src = cstats.get(t, {})
        for k in mil_fields: row[k] = src.get(k)
        mil_rows.append(row)

    # Technology sheet
    tech_fields = ["adm_tech","dip_tech","mil_tech"]
    tech_rows, tech_cols = [], ["tag","name"] + tech_fields
    for t in sorted(all_tags):
        row = {"tag": t, "name": names[t]}
        src = cstats.get(t, {})
        for k in tech_fields: row[k] = src.get(k)
        tech_rows.append(row)

    # Legitimacy sheet (government legitimacy-like mechanics)
    leg_fields = ["legitimacy","republican_tradition","devotion","horde_unity","meritocracy","absolutism","government_reform_progress","prestige","stability"]
    leg_rows, leg_cols = [], ["tag","name"] + leg_fields
    for t in sorted(all_tags):
        row = {"tag": t, "name": names[t]}
        src = cstats.get(t, {})
        for k in leg_fields: row[k] = src.get(k)
        leg_rows.append(row)

    # Overview sheet (quick glance)
    ov_rows, ov_cols = [], ["tag","name","province_count","total_development","income","manpower","army_quality_score","adm_tech","dip_tech","mil_tech","prestige","stability"]
    for t in sorted(all_tags):
        row = {
            "tag": t, "name": names[t],
            "province_count": prov_count_by_tag.get(t, 0),
            "total_development": dev_by_tag.get(t),
        }
        src = cstats.get(t, {})
        for k in ["income","manpower","army_quality_score","adm_tech","dip_tech","mil_tech","prestige","stability"]:
            row[k] = src.get(k)
        ov_rows.append(row)

    # Write CSVs
    write_csv(ap["development"], dev_rows, dev_cols)
    write_csv(ap["economy"], econ_rows, econ_cols)
    write_csv(ap["military"], mil_rows, mil_cols)
    write_csv(ap["technology"], tech_rows, tech_cols)
    write_csv(ap["legitimacy"], leg_rows, leg_cols)
    write_csv(ap["overview"], ov_rows, ov_cols)

    # Try single Excel workbook with sheets
    sheets = {
        "Overview": (ov_rows, ov_cols),
        "Development": (dev_rows, dev_cols),
        "Economy": (econ_rows, econ_cols),
        "Military": (mil_rows, mil_cols),
        "Technology": (tech_rows, tech_cols),
        "Legitimacy": (leg_rows, leg_cols),
    }
    try_write_xlsx(ap["xlsx"], sheets)

    # Terminal quick views
    def fmt(x):
        if x is None: return "-"
        return f"{int(x)}" if abs(x - int(x)) < 1e-9 else f"{x:.2f}"
    # Top dev
    top_dev = sorted((r for r in ov_rows if r.get("total_development") is not None), key=lambda r: r["total_development"], reverse=True)[:15]
    print("\nTop 15 — Total Development")
    print(f"{'TAG':<4} {'Name':<18} {'Dev':>8} {'Prov':>6} {'Income':>10} {'AQ Score':>8}")
    for r in top_dev:
        print(f"{r['tag']:<4} {r['name'][:18]:<18} {int(r['total_development']):>8} {r['province_count']:>6} {fmt(cstats.get(r['tag'],{}).get('income') or 0):>10} {fmt(cstats.get(r['tag'],{}).get('army_quality_score') or 0):>8}")

    # Top income
    top_income = sorted((r for r in ov_rows if r.get("income") is not None), key=lambda r: r["income"], reverse=True)[:15]
    print("\nTop 15 — Income")
    print(f"{'TAG':<4} {'Name':<18} {'Income':>10} {'Dev':>8} {'AQ Score':>8}")
    for r in top_income:
        print(f"{r['tag']:<4} {r['name'][:18]:<18} {fmt(r['income']):>10} {int(r.get('total_development') or 0):>8} {fmt(cstats.get(r['tag'],{}).get('army_quality_score') or 0):>8}")

if __name__ == "__main__":
    main()
