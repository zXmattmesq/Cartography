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
# It's recommended to use environment variables for sensitive data like tokens
TOKEN = os.environ.get("DISCORD_TOKEN")
if not TOKEN:
    # A fallback for local testing if you create a .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        TOKEN = os.environ.get("DISCORD_TOKEN")
    except ImportError:
        pass
    if not TOKEN:
        raise SystemExit("Error: DISCORD_TOKEN is not set in environment variables.")


# Directory for storing assets like maps and CSVs
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", "assets")).resolve()
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Performance settings for map rendering
RENDER_CHUNK = int(os.environ.get("RENDER_CHUNK", "64"))
RENDER_SCALE = float(os.environ.get("RENDER_SCALE", "0.75"))

# Path to the viewer script
VIEWER_SCRIPT = Path(os.environ.get("VIEWER_PATH", "eu4_viewer.py")).resolve()
if not VIEWER_SCRIPT.exists():
    raise SystemExit(f"Error: Viewer script not found at {VIEWER_SCRIPT}")

# Path for the output map
MAP_PATH = (ASSETS_DIR / "world_map.png").resolve()

INTENTS = discord.Intents.default()
INTENTS.message_content = True # Required for message content access in some cases
client = commands.Bot(command_prefix="!", intents=INTENTS)
tree = client.tree

# -----------------------------
# Helpers for tables (/table only)
# -----------------------------
DATASETS = {
    "overview": ["country", "province_count", "total_development", "avg_development", "income", "manpower", "mil_tech"],
    "economy": ["country", "income", "treasury", "inflation", "loans", "war_exhaustion", "corruption"],
    "military": ["country", "mil_tech", "manpower", "max_manpower", "land_forcelimit", "army_tradition", "army_professionalism"],
    "development": ["country", "province_count", "total_development", "avg_development"],
    "technology": ["country", "adm_tech", "dip_tech", "mil_tech", "technology_group"],
    "legitimacy": ["country", "legitimacy", "republican_tradition", "horde_unity", "stability"],
    "battles": ["date", "attacker", "defender", "winner", "total_casualties"],
}

def csv_path(name: str) -> Path:
    return (ASSETS_DIR / f"{name}.csv").resolve()

def load_csv(name: str) -> List[Dict[str, str]]:
    p = csv_path(name)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except (IOError, csv.Error) as e:
        print(f"Error loading CSV {p}: {e}")
        return []

def human_number(val: Optional[str]) -> float:
    if val is None or val == "":
        return float("-inf")
    s = str(val).replace(",", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("-inf")

def format_table(rows: List[Dict[str, str]], columns: List[str], limit: int = 20) -> str:
    if not rows:
        return "*(No data found for this table. Please `/submit` a save file first.)*"
        
    rows = rows[:limit]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    
    def fmt_row(r: Dict[str, str]) -> str:
        return " | ".join(f"{str(r.get(c, '')):<{widths.get(c, len(c))}}" for c in columns)
        
    header = " | ".join(f"**{c.replace('_', ' ').title()}**".ljust(widths.get(c, len(c))) for c in columns)
    sep = "-+-".join("-" * widths.get(c, len(c)) for c in columns)
    body = "\n".join(fmt_row(r) for r in rows)
    
    return f"```md\n{header}\n{sep}\n{body}\n```"

class SortView(discord.ui.View):
    def __init__(self, dataset: str, columns: List[str], rows: List[Dict[str, str]], default_sort: Optional[str] = None, timeout: int = 300):
        super().__init__(timeout=timeout)
        self.dataset = dataset
        self.columns = columns
        self.rows = rows
        self.default_sort = default_sort

        # Create up to 5 sort buttons for the most relevant columns
        numeric_first = [c for c in columns if c != "country"]
        for col in (["country"] + numeric_first)[:5]:
            self.add_item(SortButton(col, self))

    def sorted_rows(self, by: Optional[str]) -> List[Dict[str, str]]:
        if not by:
            return self.rows
        # Sort alphabetically for 'country' and numerically for all others
        is_reversed = by != "country"
        return sorted(self.rows, key=lambda r: str(r.get(by, "")).lower() if by == 'country' else human_number(r.get(by)), reverse=is_reversed)

class SortButton(discord.ui.Button):
    def __init__(self, column: str, controller: SortView):
        super().__init__(label=f"Sort: {column.replace('_', ' ').title()}", style=discord.ButtonStyle.secondary, custom_id=f"sort_{column}")
        self.column = column
        self.controller = controller

    async def callback(self, interaction: discord.Interaction):
        try:
            # Acknowledge the interaction quickly
            await interaction.response.defer()
            rows = self.controller.sorted_rows(self.column)
            await interaction.edit_original_response(
                content=format_table(rows, self.controller.columns),
                view=self.controller
            )
        except Exception as e:
            print(f"Error during sort button callback: {e}")
            await interaction.followup.send("Sorting failed. Please try again.", ephemeral=True)

# -----------------------------
# Core actions
# -----------------------------
async def run_viewer(save_path: Path) -> Tuple[bool, str]:
    """Runs the viewer script and returns success status and output."""
    # Ensure arguments are quoted to handle paths with spaces
    args = [
        "python",
        shlex.quote(str(VIEWER_SCRIPT)),
        shlex.quote(str(save_path)),
        "--assets", shlex.quote(str(ASSETS_DIR)),
        "--out", shlex.quote(str(MAP_PATH)),
        "--scale", str(RENDER_SCALE),
        "--chunk", str(RENDER_CHUNK),
    ]
    command = " ".join(args)
    
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    
    output = stdout.decode('utf-8', errors='ignore') + "\n" + stderr.decode('utf-8', errors='ignore')

    if proc.returncode != 0:
        print(f"Viewer script failed with code {proc.returncode}:\n{output}")
        return False, f"Save processing failed on the backend.\n```\n{output[-1500:]}\n```" # Show last bit of error
    
    return True, "Processing successful."

async def send_map(interaction: discord.Interaction, ephemeral: bool = False):
    if MAP_PATH.exists():
        try:
            await interaction.followup.send(file=discord.File(str(MAP_PATH)), ephemeral=ephemeral)
        except Exception as e:
            print(f"Error sending map: {e}")
            await interaction.followup.send("Map was generated, but I couldn’t attach it. Please try the `/map` command.", ephemeral=True)
    else:
        await interaction.followup.send("No map has been generated yet. Please `/submit` a save file first.", ephemeral=True)

async def send_table(interaction: discord.Interaction, dataset: str, default_sort: Optional[str] = None):
    rows = load_csv(dataset)
    cols = DATASETS.get(dataset)
    
    if not cols:
        await interaction.followup.send("Unknown table specified.", ephemeral=True)
        return
    if not rows:
        await interaction.followup.send("No data found. Please ensure a save has been processed with `/submit`.", ephemeral=True)
        return
        
    view = SortView(dataset, cols, rows, default_sort=default_sort)
    # Determine the initial sort column
    initial_sort_col = default_sort or (cols[1] if len(cols) > 1 else cols[0])
    sorted_data = view.sorted_rows(initial_sort_col)
    
    await interaction.followup.send(format_table(sorted_data, cols), view=view, ephemeral=False)

# -----------------------------
# Slash commands
# -----------------------------
@tree.command(name="submit", description="Upload a EU4 save file to generate a map and stats")
@app_commands.describe(attachment="Your .eu4 save file (can be zipped or plain text)")
async def submit(interaction: discord.Interaction, attachment: discord.Attachment):
    await interaction.response.defer(thinking=True, ephemeral=True)
    
    if not attachment.filename.lower().endswith(".eu4"):
        await interaction.followup.send("This doesn't look like a `.eu4` save file. Please try again.", ephemeral=True)
        return

    # Save the attachment to a temporary file
    save_file_path = ASSETS_DIR / f"save_{interaction.user.id}_{int(time.time())}.eu4"
    try:
        await attachment.save(save_file_path)

        # Run the processing script
        success, message = await run_viewer(save_file_path)
        
        if not success:
            await interaction.followup.send(f"I couldn’t process that save file. {message}", ephemeral=True)
            return

        # On success, send the map publicly and a confirmation message
        await send_map(interaction, ephemeral=False)
        await interaction.followup.send("Processed your save! Use the `/table` command to view detailed statistics.", ephemeral=True)

    except Exception as e:
        print(f"An error occurred during /submit: {e}")
        await interaction.followup.send("A critical error occurred while handling your request.", ephemeral=True)
    finally:
        # Clean up the temporary save file
        if save_file_path.exists():
            save_file_path.unlink()

@tree.command(name="map", description="Shows the most recently generated world map")
async def map_cmd(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True, ephemeral=False)
    # Re-use the send_map function, making it non-ephemeral
    await send_map(interaction, ephemeral=False)

@tree.command(name="table", description="Shows sortable statistics from the last processed save")
@app_commands.describe(name="The category of statistics to display")
@app_commands.choices(name=[
    app_commands.Choice(name=cat.title(), value=cat) for cat in DATASETS.keys()
])
async def table_cmd(interaction: discord.Interaction, name: app_commands.Choice[str]):
    await interaction.response.defer(thinking=True, ephemeral=False)
    
    # Set a smart default sort column for each table
    default_sort_map = {
        "overview": "total_development",
        "economy": "income",
        "military": "mil_tech",
        "development": "total_development",
        "technology": "mil_tech",
        "battles": "total_casualties",
        "legitimacy": "stability",
    }
    await send_table(interaction, name.value, default_sort=default_sort_map.get(name.value))

@client.event
async def on_ready():
    print(f'Logged in as {client.user} (ID: {client.user.id})')
    print('------')
    try:
        synced = await tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

client.run(TOKEN)
