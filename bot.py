#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discord bot for EU4 Viewer (low-RAM friendly)
- /submit save: upload .eu4, process with eu4_viewer.py (chunked), post map + CSVs
- /table category [limit] [column]: show top N by column (descending). Default = first numeric column.
- /map: show latest map for this guild
- Memory-safe file sending (no files=None bug)
- Keeps only the latest results per guild; cleans old ones
- Health HTTP server on :10000 for Render
"""

import asyncio
import csv
import os
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

# ---------- Config ----------
load_dotenv()
TOKEN = os.environ.get("DISCORD_TOKEN")
CHUNK = os.environ.get("VIEWER_CHUNK", "64")           # str for subprocess
SCALE = os.environ.get("VIEWER_SCALE", "0.75")         # str for subprocess
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", ".")).resolve()

# paths inside ASSETS_DIR
MAP_NAME = "world_map.png"
CSV_FILES = [
    "overview.csv",
    "economy.csv",
    "military.csv",
    "development.csv",
    "technology.csv",
    "legitimacy.csv",
]

# ---------- Bot ----------
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# per-guild latest results directory
LATEST: Dict[int, Path] = {}
# single worker semaphore to respect tiny CPU/RAM
WORK_SEM = asyncio.Semaphore(1)

# ---------- tiny http health server (Render keeps it alive) ----------
async def _health_server():
    from aiohttp import web
    async def ok(_):
        return web.Response(text="ok")
    app = web.Application()
    app.router.add_get("/", ok)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 10000)
    await site.start()

@bot.event
async def on_ready():
    try:
        await bot.tree.sync()
    except Exception:
        pass
    bot.loop.create_task(_health_server())
    print(f"Logged in as {bot.user} (id={bot.user.id})")

# ---------- helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)

def find_first_numeric_header(headers: List[str], rows: List[Dict[str, str]]) -> Optional[str]:
    # skip tag/name; pick the first column that parses as number for at least one row
    skip = {"tag", "name"}
    for h in headers:
        if h in skip:
            continue
        for r in rows:
            v = r.get(h, "")
            try:
                float(v)
                return h
            except Exception:
                continue
    return None

def load_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str,str]]]:
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames or []
        rows = [row for row in rdr]
    return headers, rows

def sort_rows(rows: List[Dict[str,str]], key: str) -> List[Dict[str,str]]:
    def to_num(x):
        try:
            return float(x.get(key, ""))
        except Exception:
            return float("-inf")
    return sorted(rows, key=to_num, reverse=True)

def format_table(headers: List[str], rows: List[Dict[str,str]], limit: int) -> str:
    # hide tag, ruler names etc.; display name + helpful numeric columns
    H = [h for h in headers if h != "tag"]
    # take only displayed columns that exist in rows
    H = [h for h in H if any(r.get(h, "") != "" for r in rows)]
    # build simple fixed-width table
    widths = {h: max(len(h), *(len(str(r.get(h,""))) for r in rows[:limit])) for h in H}
    line = " | ".join(h.ljust(widths[h]) for h in H)
    sep = "-+-".join("-"*widths[h] for h in H)
    body = []
    for r in rows[:limit]:
        body.append(" | ".join(str(r.get(h,"")).ljust(widths[h]) for h in H))
    return "```\n" + line + "\n" + sep + "\n" + "\n".join(body) + "\n```"

async def run_viewer(save_path: Path, work_dir: Path) -> Tuple[Optional[Path], List[Path]]:
    """
    Runs eu4_viewer.py with low-RAM options.
    Returns (map_path_or_None, list_of_csv_paths_found)
    """
    ensure_dir(work_dir)
    map_path = work_dir / MAP_NAME

    cmd = [
        sys.executable, "eu4_viewer.py",
        str(save_path),
        "--assets", str(ASSETS_DIR),
        "--out", str(map_path),
        "--chunk", str(CHUNK),
        "--scale", str(SCALE),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path(__file__).parent.resolve()),
    )
    out, err = await proc.communicate()
    if out:
        print(out.decode("utf-8", "ignore"))
    if err:
        print(err.decode("utf-8", "ignore"), file=sys.stderr)

    csv_paths = []
    for name in CSV_FILES:
        p = ASSETS_DIR / name
        if p.exists():
            csv_paths.append(p)

    return (map_path if map_path.exists() else None, csv_paths)

def build_discord_files(map_path: Optional[Path], csv_paths: List[Path]) -> List[discord.File]:
    files: List[discord.File] = []
    if map_path and map_path.exists():
        files.append(discord.File(str(map_path), filename=MAP_NAME))
    for p in csv_paths:
        try:
            files.append(discord.File(str(p), filename=p.name))
        except Exception:
            continue
    return files

# ---------- commands ----------
@app_commands.command(name="submit", description="Submit a EU4 save (.eu4) for processing")
@app_commands.describe(save="Attach your .eu4 save file")
async def submit_save_cmd(interaction: discord.Interaction, save: discord.Attachment):
    await interaction.response.defer(ephemeral=False, thinking=True)

    if not save.filename.lower().endswith(".eu4"):
        return await interaction.followup.send("Please upload a `.eu4` save file.")

    # Serialize processing with a global semaphore (tiny plan)
    async with WORK_SEM:
        # temp per-job workspace
        tmpdir = Path(tempfile.mkdtemp(prefix="eu4job_"))
        try:
            local_save = tmpdir / save.filename
            await save.save(str(local_save))

            # run viewer
            map_path, csv_paths = await run_viewer(local_save, tmpdir)

            # move "latest" for this guild
            guild_id = interaction.guild_id or 0
            guild_dir = Path("data") / str(guild_id) / "latest"
            clean_dir(guild_dir)

            if map_path and map_path.exists():
                shutil.copy2(map_path, guild_dir / MAP_NAME)
            for p in csv_paths:
                if p.exists():
                    shutil.copy2(p, guild_dir / p.name)

            LATEST[guild_id] = guild_dir

            # send result (ONLY message + files)
            files = build_discord_files(guild_dir / MAP_NAME, [guild_dir / n for n in CSV_FILES if (guild_dir / n).exists()])
            msg = "Processed your save. Map + CSVs below."
            if files:
                try:
                    await interaction.followup.send(msg, files=files)
                finally:
                    # close file handles to release FDs
                    for f in files:
                        try:
                            fp = getattr(f, "fp", None)
                            if fp and not fp.closed:
                                fp.close()
                        except Exception:
                            pass
            else:
                await interaction.followup.send(msg)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

@bot.tree.command(name="map", description="Show the latest processed map")
async def map_cmd(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False, thinking=True)
    guild_id = interaction.guild_id or 0
    base = LATEST.get(guild_id)
    if not base:
        return await interaction.followup.send("No processed map yet. Use `/submit` first.")
    path = base / MAP_NAME
    if not path.exists():
        return await interaction.followup.send("Map not found. Submit a new save.")
    f = discord.File(str(path), filename=MAP_NAME)
    try:
        await interaction.followup.send(files=[f])
    finally:
        fp = getattr(f, "fp", None)
        if fp and not fp.closed:
            fp.close()

CATEGORIES = {
    "overview": "overview.csv",
    "economy": "economy.csv",
    "military": "military.csv",
    "development": "development.csv",
    "technology": "technology.csv",
    "legitimacy": "legitimacy.csv",
}

@app_commands.command(name="table", description="Show a stats table")
@app_commands.describe(category="Which table", limit="How many rows (default 10)", column="Sort by this column (desc)")
@app_commands.choices(category=[app_commands.Choice(name=k, value=k) for k in CATEGORIES.keys()])
async def table_cmd(interaction: discord.Interaction, category: app_commands.Choice[str], limit: Optional[int] = 10, column: Optional[str] = None):
    await interaction.response.defer(ephemeral=False, thinking=True)
    guild_id = interaction.guild_id or 0
    base = LATEST.get(guild_id)
    if not base:
        return await interaction.followup.send("No data yet. Use `/submit` first.")

    csv_path = base / CATEGORIES[category.value]
    if not csv_path.exists():
        return await interaction.followup.send(f"No CSV found for `{category.value}`. Submit a new save.")

    headers, rows = load_csv_rows(csv_path)
    if not rows:
        return await interaction.followup.send("No rows in CSV.")

    # default sort: first numeric column (descending)
    sort_key = column
    if not sort_key or sort_key not in headers:
        sort_key = find_first_numeric_header(headers, rows) or "province_count"  # fallback

    rows_sorted = sort_rows(rows, sort_key)
    limit = max(1, min(int(limit or 10), 100))

    table_txt = format_table(headers, rows_sorted, limit)
    await interaction.followup.send(f"**{category.value.capitalize()}** — sorted by **{sort_key}** (desc)\n{table_txt}")

# ---------- run ----------
if __name__ == "__main__":
    if not TOKEN:
        print("Missing DISCORD_TOKEN in env", file=sys.stderr)
        sys.exit(1)

    # Graceful shutdown on SIGTERM (Render)
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.ensure_future(bot.close()))
        except NotImplementedError:
            pass

    bot.run(TOKEN)
