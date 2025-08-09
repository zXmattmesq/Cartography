import os
import csv
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands

# ======================== CONFIG ========================
ASSETS_DIR = Path(__file__).parent.resolve()
EU4_SCRIPT = ASSETS_DIR / "eu4_viewer.py"
DEFAULT_TABLE_LIMIT = 10

CATEGORIES = {
    "overview":     {"file": "overview.csv"},
    "economy":      {"file": "economy.csv"},
    "military":     {"file": "military.csv"},
    "development":  {"file": "development.csv"},
    "technology":   {"file": "technology.csv"},
    "legitimacy":   {"file": "legitimacy.csv"},
}

COUNTRY_NAMES_FILE = ASSETS_DIR / "country_names.csv"  # optional tag->name mapping

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# Last processed workspace per guild
LATEST_WORKDIR: Dict[int, Path] = {}

# ======================== UTILITIES ========================
def load_country_name_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    if COUNTRY_NAMES_FILE.exists():
        with COUNTRY_NAMES_FILE.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2: continue
                tag, name = row[0].strip().upper(), row[1].strip()
                if len(tag) == 3 and name:
                    m[tag] = name
    return m

TAG_TO_NAME = load_country_name_map()

async def run_eu4_viewer(save_path: Path, workdir: Path) -> bool:
    """Run the map/stats generator; copy CSVs to workdir. Return True if anything generated."""
    if not EU4_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find eu4_viewer.py at {EU4_SCRIPT}")

    out_map = workdir / "world_map.png"
    cmd = [
        "python3",
        str(EU4_SCRIPT),
        str(save_path),
        "--assets", str(ASSETS_DIR),
        "--out", str(out_map),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(ASSETS_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()

    generated = out_map.exists()
    for meta in CATEGORIES.values():
        src = ASSETS_DIR / meta["file"]
        if src.exists():
            (workdir / meta["file"]).write_bytes(src.read_bytes())
            generated = True
    return generated

def read_csv_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def is_number(x: Optional[str]) -> bool:
    if x is None or x == "": return False
    try:
        float(x); return True
    except Exception:
        return False

def numeric_columns(rows: List[dict], preferred_order: List[str]) -> List[str]:
    """Return numeric columns in a sensible order (first match from preferred_order, then others)."""
    if not rows: return []
    cols = list(rows[0].keys())
    nums = [c for c in cols if c not in ("tag","name") and any(is_number(r.get(c)) for r in rows)]
    # move preferred columns to the front if present
    ordered = [c for c in preferred_order if c in nums]
    for c in nums:
        if c not in ordered:
            ordered.append(c)
    return ordered

def best_country_name(row: dict) -> str:
    """Prefer our tag->name mapping; else try a sane country name from CSV; else tag."""
    tag = (row.get("tag") or "").upper()
    if tag in TAG_TO_NAME:
        return TAG_TO_NAME[tag]
    # CSVs produced by eu4_viewer put ruler names in 'name' sometimes; prefer more stable fields if present
    for key in ("country_name", "localized_name", "long_name"):
        if row.get(key):
            return str(row[key])[:40]
    # fall back to the CSV 'name' only if it's not obviously a monarch-style name
    n = (row.get("name") or "").strip()
    if n and len(n.split()) <= 3 and not any(ch.isdigit() for ch in n):
        # simple heuristic: if it's a very long personal-looking name, ignore
        if len(n) <= 24:
            return n
    return tag or "Unknown"

def coerce_display_rows(rows: List[dict]) -> List[dict]:
    display = []
    for r in rows:
        rr = dict(r)
        rr["name"] = best_country_name(r)
        rr.pop("tag", None)  # never show tag
        display.append(rr)
    return display

def monospace_table(rows: List[dict], cols: List[str], limit: int) -> str:
    rows = rows[:limit]
    # Clamp name width
    for r in rows:
        if "name" in r and r["name"]:
            r["name"] = str(r["name"])[:22]

    def fmt(x: Optional[str]) -> str:
        if x is None or x == "": return "-"
        try:
            fx = float(x)
            if abs(fx - int(fx)) < 1e-9: return f"{int(fx)}"
            return f"{fx:.2f}"
        except Exception:
            return str(x)

    widths = {}
    for c in cols:
        head = c.upper()
        maxw = len(head)
        for r in rows:
            v = fmt(r.get(c))
            if len(v) > maxw: maxw = len(v)
        widths[c] = max(4, min(maxw, 22))

    line_header = " ".join(f"{c.upper():<{widths[c]}}" for c in cols)
    line_sep = "-" * len(line_header)
    lines = [line_header, line_sep]
    for r in rows:
        parts = [f"{fmt(r.get(c)):<{widths[c]}}" for c in cols]
        lines.append(" ".join(parts))
    return "```\n" + "\n".join(lines) + "\n```"

def default_cols_for(category: str, rows: List[dict]) -> List[str]:
    base = {
        "overview":     ["name","province_count","total_development","income","manpower","army_quality_score"],
        "economy":      ["name","income","treasury","inflation","loans","interest","war_exhaustion","corruption"],
        "military":     ["name","army_quality_score","manpower","max_manpower","land_forcelimit",
                         "army_tradition","army_professionalism","discipline","land_morale"],
        "development":  ["name","total_development","province_count","avg_development"],
        "technology":   ["name","mil_tech","adm_tech","dip_tech"],
        "legitimacy":   ["name","absolutism","legitimacy","republican_tradition","devotion",
                         "horde_unity","meritocracy","government_reform_progress","prestige","stability"],
    }.get(category, None)
    if base:
        # keep only existing columns
        have = rows[0].keys()
        base = [c for c in base if c in have]
        if "name" not in base and "name" in have:
            base = ["name"] + base
        return base
    # fallback to all visible columns
    return [c for c in rows[0].keys() if c != "tag"]

def auto_pick_sort_column(category: str, rows: List[dict]) -> Optional[str]:
    prefs = {
        "overview":     ["total_development","income","manpower","army_quality_score"],
        "economy":      ["income","treasury"],
        "military":     ["army_quality_score","manpower","land_forcelimit"],
        "development":  ["total_development","avg_development","province_count"],
        "technology":   ["mil_tech","adm_tech","dip_tech"],
        "legitimacy":   ["absolutism","legitimacy","prestige"],
    }.get(category, [])
    nums = numeric_columns(rows, prefs)
    return nums[0] if nums else None

def sort_rows_by(rows: List[dict], key: str) -> List[dict]:
    def keyfn(r):
        v = r.get(key)
        try: return float(v) if v not in (None, "") else float("-inf")
        except Exception: return float("-inf")
    return sorted(rows, key=keyfn, reverse=True)

# ===== UI View: column picker =====
class ColumnPicker(discord.ui.Select):
    def __init__(self, category: str, rows: List[dict], default_key: Optional[str]):
        options = []
        nums = numeric_columns(rows, [])
        for c in nums[:25]:  # Discord limit
            options.append(discord.SelectOption(label=c, value=c, default=(c == default_key)))
        super().__init__(placeholder="Sort by…", min_values=1, max_values=1, options=options)
        self.category = category
        self.rows = rows
        self.default_key = default_key

    async def callback(self, interaction: discord.Interaction):
        key = self.values[0]
        disp = coerce_display_rows(self.rows)
        cols = default_cols_for(self.category, disp)
        sort_key = key if key in disp[0] else key  # same names after coerce
        disp_sorted = sort_rows_by(disp, sort_key)
        table_txt = monospace_table(disp_sorted, cols, DEFAULT_TABLE_LIMIT)
        await interaction.response.edit_message(content=f"**{self.category.capitalize()} — Top {DEFAULT_TABLE_LIMIT} (sorted by {key})**\n{table_txt}", view=self.view)

class ColumnPickerView(discord.ui.View):
    def __init__(self, category: str, rows: List[dict], default_key: Optional[str]):
        super().__init__(timeout=180)
        self.add_item(ColumnPicker(category, rows, default_key))

# ======================== COMMANDS ========================
@bot.event
async def on_ready():
    try:
        await bot.tree.sync()
    except Exception as e:
        print("Slash sync failed:", e)
    print(f"Logged in as {bot.user} (id={bot.user.id})")

@bot.tree.command(name="submit_save", description="Upload a EU4 save; I'll render the map and generate stats tables.")
@app_commands.describe(savefile="Attach a .eu4 save file")
async def submit_save_cmd(interaction: discord.Interaction, savefile: discord.Attachment):
    # Public response (not ephemeral)
    await interaction.response.defer(thinking=True, ephemeral=False)

    guild_id = interaction.guild_id or interaction.user.id
    workdir = Path(tempfile.gettempdir()) / f"eu4bot_{guild_id}"
    workdir.mkdir(parents=True, exist_ok=True)
    LATEST_WORKDIR[guild_id] = workdir

    save_path = workdir / (savefile.filename or "save.eu4")
    await savefile.save(fp=save_path)

    ok = await run_eu4_viewer(save_path, workdir)

    files = []
    map_file = workdir / "world_map.png"
    if map_file.exists():
        files.append(discord.File(str(map_file), filename="world_map.png"))

    msg = "✅ Processed your save."
    await interaction.followup.send(msg, files=files)

@bot.tree.command(name="map", description="Send the last rendered world map for this server.")
async def map_cmd(interaction: discord.Interaction):
    guild_id = interaction.guild_id or interaction.user.id
    workdir = LATEST_WORKDIR.get(guild_id)
    if not workdir:
        return await interaction.response.send_message("No processed save yet. Use **/submit_save** first.")
    path = workdir / "world_map.png"
    if not path.exists():
        return await interaction.response.send_message("I couldn't find a map for the last run.")
    await interaction.response.send_message(file=discord.File(str(path), filename="world_map.png"))

@bot.tree.command(name="categories", description="List all table categories you can show.")
async def categories_cmd(interaction: discord.Interaction):
    cats = ", ".join(CATEGORIES.keys())
    await interaction.response.send_message(f"Available categories: **{cats}**")

@bot.tree.command(name="table", description="Show a top-N table from a category, sorted by the first numeric column. Use the dropdown to change.")
@app_commands.describe(
    category="Which table to show (overview, economy, military, development, technology, legitimacy)",
    limit=f"How many rows (default {DEFAULT_TABLE_LIMIT}, max 100)"
)
@app_commands.choices(category=[app_commands.Choice(name=n, value=n) for n in CATEGORIES.keys()])
async def table_cmd(interaction: discord.Interaction, category: app_commands.Choice[str], limit: Optional[int] = None):
    await interaction.response.defer(thinking=True, ephemeral=False)
    guild_id = interaction.guild_id or interaction.user.id
    workdir = LATEST_WORKDIR.get(guild_id)
    if not workdir:
        return await interaction.followup.send("No processed save yet. Use **/submit_save** first.")
    csv_path = workdir / CATEGORIES[category.value]["file"]
    if not csv_path.exists():
        return await interaction.followup.send(f"I couldn't find `{csv_path.name}`. Please run **/submit_save** again.")

    rows = read_csv_rows(csv_path)
    if not rows:
        return await interaction.followup.send("That table is empty.")

    # Clean up names and hide tags
    disp = coerce_display_rows(rows)

    # Decide columns & sort key
    cols = default_cols_for(category.value, disp)
    sort_key = auto_pick_sort_column(category.value, disp)
    if sort_key and sort_key not in disp[0]:
        # still fine: columns are identical names after coerce
        pass

    disp_sorted = sort_rows_by(disp, sort_key) if sort_key else disp
    n = max(1, min(limit or DEFAULT_TABLE_LIMIT, 100))
    table_txt = monospace_table(disp_sorted, cols, n)

    # View with column picker
    view = ColumnPickerView(category.value, rows, sort_key)
    await interaction.followup.send(f"**{category.value.capitalize()} — Top {n} (sorted by {sort_key or '—'})**\n{table_txt}", view=view)

# ======================== ENTRYPOINT ========================
def main():
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Set DISCORD_TOKEN environment variable.")
    bot.run(token)

if __name__ == "__main__":
    main()
