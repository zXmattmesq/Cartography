# bot.py
import asyncio
import csv
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands

import subprocess
import shlex

# -----------------------------
# Config
# -----------------------------
TOKEN = os.environ.get("DISCORD_TOKEN")
if not TOKEN:
    raise SystemExit("DISCORD_TOKEN is not set")

ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", ".")).resolve()
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

RENDER_CHUNK = int(os.environ.get("RENDER_CHUNK", "64"))
RENDER_SCALE = float(os.environ.get("RENDER_SCALE", "0.75"))

VIEWER = Path(os.environ.get("VIEWER_PATH", "eu4_viewer.py")).resolve()
MAP_PATH = (ASSETS_DIR / "world_map.png").resolve()

INTENTS = discord.Intents.default()
client = commands.Bot(command_prefix="!", intents=INTENTS)
tree = client.tree  # <-- use the existing tree; DO NOT create CommandTree(client)

# -----------------------------
# Helpers
# -----------------------------
DATASETS = {
    "overview": ["country", "province_count", "total_development", "avg_development", "income", "manpower", "army_quality_score"],
    "economy": ["country", "income", "treasury", "inflation", "loans", "war_exhaustion", "corruption"],
    "military": ["country", "army_quality_score", "manpower", "max_manpower", "land_forcelimit", "army_tradition", "army_professionalism"],
    "development": ["country", "province_count", "total_development", "avg_development"],
    "technology": ["country", "adm_tech", "dip_tech", "mil_tech", "technology_group"],
    "legitimacy": ["country", "legitimacy", "republican_tradition", "horde_unity", "stability"],
    "battles": ["date", "province_id", "attacker", "defender", "winner", "attacker_casualties", "defender_casualties", "attacker_attrition", "defender_attrition", "total_casualties", "total_attrition"],
}

def csv_path(name: str) -> Path:
    return (ASSETS_DIR / f"{name}.csv").resolve()

def load_csv(name: str) -> List[Dict[str, str]]:
    path = csv_path(name)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def human_number(val: str) -> Tuple[float, str]:
    if val is None:
        return (float("-inf"), "")
    s = str(val).replace(",", "").strip()
    try:
        return (float(s), val)
    except:
        return (float("-inf"), val)

def format_table(rows: List[Dict[str, str]], columns: List[str], limit: int = 15) -> str:
    rows = rows[:limit]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    def fmt_row(r: Dict[str, str]) -> str:
        return " | ".join(f"{str(r.get(c, '')):<{widths[c]}}" for c in columns)
    header = " | ".join(f"{c:<{widths[c]}}" for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    body = "\n".join(fmt_row(r) for r in rows)
    return f"```\n{header}\n{sep}\n{body}\n```" if rows else "*(No data)*"

# -----------------------------
# Sortable view
# -----------------------------
class SortView(discord.ui.View):
    def __init__(self, dataset: str, columns: List[str], rows: List[Dict[str, str]], default_sort: Optional[str] = None, timeout: int = 180):
        super().__init__(timeout=timeout)
        self.dataset = dataset
        self.columns = columns
        self.rows = rows
        self.sort_key = default_sort

        numeric_first = [c for c in columns if c != "country"]
        candidates = (["country"] + numeric_first)[:5]
        for col in candidates:
            self.add_item(SortButton(col, self))

    def sorted_rows(self, by: Optional[str]) -> List[Dict[str, str]]:
        if not by:
            return self.rows
        if by == "country":
            return sorted(self.rows, key=lambda r: str(r.get("country", "")).lower(), reverse=True)
        return sorted(self.rows, key=lambda r: human_number(r.get(by, ""))[0], reverse=True)

class SortButton(discord.ui.Button):
    def __init__(self, column: str, controller: SortView):
        super().__init__(label=f"Sort: {column}", style=discord.ButtonStyle.secondary)
        self.column = column
        self.controller = controller

    async def callback(self, interaction: discord.Interaction):
        try:
            rows = self.controller.sorted_rows(self.column)
            txt = format_table(rows, self.controller.columns)
            await interaction.response.edit_message(content=txt, view=self.controller)
        except Exception:
            await interaction.response.edit_message(content="Something went wrong while sorting. Try a different column.", view=self.controller)

# -----------------------------
# Core actions
# -----------------------------
async def run_viewer(save_path: Path) -> None:
    args = [
        shlex.quote(str(VIEWER)),
        shlex.quote(str(save_path)),
        "--assets", shlex.quote(str(ASSETS_DIR)),
        "--out", shlex.quote(str(MAP_PATH)),
        "--scale", str(RENDER_SCALE),
        "--chunk", str(RENDER_CHUNK),
    ]
    proc = await asyncio.create_subprocess_shell(
        "python " + " ".join(args),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("Save processing failed")

async def send_map(interaction: discord.Interaction):
    if MAP_PATH.exists():
        try:
            await interaction.followup.send(file=discord.File(str(MAP_PATH)), ephemeral=False)
        except Exception:
            await interaction.followup.send("Map was generated, but I couldn’t attach it. Try `/map` again.", ephemeral=True)
    else:
        await interaction.followup.send("I processed the save but couldn’t find the map image. Try `/map`.", ephemeral=True)

async def send_table(interaction: discord.Interaction, dataset: str, default_sort: Optional[str] = None):
    rows = load_csv(dataset)
    cols = DATASETS.get(dataset)
    if not cols:
        await interaction.followup.send("Unknown dataset.", ephemeral=True)
        return
    if not rows:
        msg = "No data found for this table."
        if dataset == "battles":
            msg = "No battles recorded in this save."
        await interaction.followup.send(msg, ephemeral=True)
        return
    view = SortView(dataset, cols, rows, default_sort=default_sort)
    initial_sort = default_sort or (cols[1] if len(cols) > 1 else None)
    txt = format_table(view.sorted_rows(initial_sort), cols)
    await interaction.followup.send(txt, view=view, ephemeral=False)

# -----------------------------
# Slash commands
# -----------------------------
@tree.command(name="submit", description="Upload a EU4 save file")
@app_commands.describe(attachment="Your .eu4 save (zip or text)")
async def submit(interaction: discord.Interaction, attachment: discord.Attachment):
    await interaction.response.defer(thinking=True, ephemeral=True)
    try:
        if not attachment.filename.lower().endswith(".eu4"):
            await interaction.followup.send("Please upload a `.eu4` save file.", ephemeral=True)
            return
        tmp = ASSETS_DIR / f"save_{int(time.time())}.eu4"
        data = await attachment.read()
        tmp.write_bytes(data)

        await run_viewer(tmp)

        await send_map(interaction)
        await send_table(interaction, "overview", default_sort="total_development")

        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

        await interaction.followup.send("Processed your save. Use `/table` for other tables or `/battles` for combat stats.", ephemeral=True)
    except Exception:
        await interaction.followup.send("I couldn’t process that save. Please make sure it’s a valid EU4 save and try again.", ephemeral=True)

@tree.command(name="map", description="Show the latest rendered world map")
async def map_cmd(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True, ephemeral=True)
    await send_map(interaction)

@tree.command(name="table", description="Show a sortable table")
@app_commands.describe(name="Which table to show")
@app_commands.choices(name=[
    app_commands.Choice(name="Overview", value="overview"),
    app_commands.Choice(name="Economy", value="economy"),
    app_commands.Choice(name="Military", value="military"),
    app_commands.Choice(name="Development", value="development"),
    app_commands.Choice(name="Technology", value="technology"),
    app_commands.Choice(name="Legitimacy", value="legitimacy"),
    app_commands.Choice(name="Battles", value="battles"),
])
async def table_cmd(interaction: discord.Interaction, name: app_commands.Choice[str]):
    await interaction.response.defer(thinking=True, ephemeral=True)
    default = None
    if name.value in ("overview", "development"):
        default = "total_development"
    elif name.value == "economy":
        default = "income"
    elif name.value == "military":
        default = "army_quality_score"
    elif name.value == "technology":
        default = "mil_tech"
    elif name.value == "battles":
        default = "total_casualties"
    await send_table(interaction, name.value, default_sort=default)

# -----------------------------
# Startup
# -----------------------------
@client.event
async def on_ready():
    try:
        await tree.sync()
    except Exception:
        pass

client.run(TOKEN)
