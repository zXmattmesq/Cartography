# bot.py
import asyncio
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands

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

# --- INTENTS FIX ---
# Enable the privileged intents required by the bot.
# You must also enable these in your bot's settings on the Discord Developer Portal.
INTENTS = discord.Intents.default()
INTENTS.message_content = True
INTENTS.members = True
# --- END FIX ---

client = commands.Bot(command_prefix="!", intents=INTENTS)
tree = client.tree  # use the built-in tree

# -----------------------------
# Helpers for tables (/table only)
# -----------------------------
DATASETS = {
    "overview": ["country", "province_count", "total_development", "avg_development", "income", "manpower", "army_quality_score"],
    "economy": ["country", "income", "treasury", "inflation", "loans", "war_exhaustion", "corruption"],
    "military": ["country", "army_quality_score", "manpower", "max_manpower", "land_forcelimit", "army_tradition", "army_professionalism"],
    "development": ["country", "province_count", "total_development", "avg_development"],
    "technology": ["country", "adm_tech", "dip_tech", "mil_tech", "technology_group"],
    "legitimacy": ["country", "legitimacy", "republican_tradition", "horde_unity", "stability"],
    "battles": ["date", "province_id", "attacker", "defender", "winner", "attacker_casualties", "defender_casualties", "total_casualties"],
}

def csv_path(name: str) -> Path:
    return (ASSETS_DIR / f"{name}.csv").resolve()

def load_csv(name: str) -> List[Dict[str, str]]:
    p = csv_path(name)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def human_number(val: str) -> float:
    if val is None:
        return float("-inf")
    s = str(val).replace(",", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("-inf")

def format_table(rows: List[Dict[str, str]], columns: List[str], limit: int = 15) -> str:
    rows = rows[:limit]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    def fmt_row(r: Dict[str, str]) -> str:
        return " | ".join(f"{str(r.get(c, '')):<{widths[c]}}" for c in columns)
    header = " | ".join(f"{c:<{widths[c]}}" for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    body = "\n".join(fmt_row(r) for r in rows)
    return f"```\n{header}\n{sep}\n{body}\n```" if rows else "*(No data)*"

class SortView(discord.ui.View):
    def __init__(self, dataset: str, columns: List[str], rows: List[Dict[str, str]], default_sort: Optional[str] = None, timeout: int = 180):
        super().__init__(timeout=timeout)
        self.dataset = dataset
        self.columns = columns
        self.rows = rows
        self.default_sort = default_sort

        # up to 5 handy sort buttons
        numeric_first = [c for c in columns if c != "country"]
        for col in (["country"] + numeric_first)[:5]:
            self.add_item(SortButton(col, self))

    def sorted_rows(self, by: Optional[str]) -> List[Dict[str, str]]:
        if not by:
            return self.rows
        if by == "country":
            return sorted(self.rows, key=lambda r: str(r.get("country", "")).lower(), reverse=False)
        return sorted(self.rows, key=lambda r: human_number(r.get(by, "")), reverse=True)

class SortButton(discord.ui.Button):
    def __init__(self, column: str, controller: SortView):
        super().__init__(label=f"Sort: {column}", style=discord.ButtonStyle.secondary)
        self.column = column
        self.controller = controller

    async def callback(self, interaction: discord.Interaction):
        try:
            rows = self.controller.sorted_rows(self.column)
            await interaction.response.edit_message(
                content=format_table(rows, self.controller.columns),
                view=self.controller
            )
        except Exception:
            await interaction.response.edit_message(
                content="Sorting failed. Try a different column.",
                view=self.controller
            )

# -----------------------------
# Core actions
# -----------------------------
async def run_viewer(save_path: Path) -> None:
    args = [
        sys.executable, # Use the same python interpreter that's running the bot
        shlex.quote(str(VIEWER)),
        shlex.quote(str(save_path)),
        "--assets", shlex.quote(str(ASSETS_DIR)),
        "--out", shlex.quote(str(MAP_PATH)),
        "--scale", str(RENDER_SCALE),
        "--chunk", str(RENDER_CHUNK),
    ]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        print(f"Viewer script failed with code {proc.returncode}:", file=sys.stderr)
        print(stderr.decode(), file=sys.stderr)
        raise RuntimeError("Save processing failed")

async def send_map(interaction: discord.Interaction):
    if MAP_PATH.exists():
        try:
            await interaction.followup.send(file=discord.File(str(MAP_PATH)), ephemeral=False)
        except Exception as e:
            print(f"Failed to send map: {e}", file=sys.stderr)
            await interaction.followup.send("Map generated, but I couldn’t attach it. Try `/map`.", ephemeral=True)
    else:
        await interaction.followup.send("Processed the save, but no map was found. Try `/map`.", ephemeral=True)

async def send_table(interaction: discord.Interaction, dataset: str, default_sort: Optional[str] = None):
    rows = load_csv(dataset)
    cols = DATASETS.get(dataset)
    if not cols:
        await interaction.followup.send("Unknown table.", ephemeral=True)
        return
    if not rows:
        await interaction.followup.send("No data found for this table. This usually means the save file was empty or couldn't be read.", ephemeral=True)
        return
    view = SortView(dataset, cols, rows, default_sort=default_sort)
    initial = default_sort or (cols[1] if len(cols) > 1 else None)
    await interaction.followup.send(format_table(view.sorted_rows(initial), cols), view=view, ephemeral=False)

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

        tmp = ASSETS_DIR / f"save_{interaction.user.id}_{int(time.time())}.eu4"
        await attachment.save(tmp)

        await run_viewer(tmp)          # process
        await send_map(interaction)     # show only the map

        try:
            tmp.unlink(missing_ok=True)
        except Exception as e:
            print(f"Failed to clean up temp file {tmp}: {e}", file=sys.stderr)

        await interaction.followup.send("Processed your save. Use `/table` anytime to view stats.", ephemeral=True)
    except Exception as e:
        print(f"Error in /submit command: {e}", file=sys.stderr)
        await interaction.followup.send("I couldn’t process that save. Make sure it’s a valid, uncompressed save file and try again.", ephemeral=True)

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

@client.event
async def on_ready():
    print(f'Logged in as {client.user} (ID: {client.user.id})')
    print('------')
    try:
        synced = await tree.sync()
        print(f"Synced {len(synced)} commands.")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

client.run(TOKEN)
