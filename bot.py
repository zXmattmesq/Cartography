#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discord bot for EU4 Viewer (low-RAM friendly)
- /submit: upload a .eu4 save; renders map & CSVs via eu4_viewer.py (chunked)
- /map:    posts the most recent map for this server (no defer)
- /table:  shows a sorted table (default top 10; no defer)
- /sync:   force-resync slash commands (admin)
- Public messages (not ephemeral)
- Single-job semaphore to fit tiny CPU/RAM
- Health HTTP server on :10000 (keeps Render alive)
"""

import asyncio
import csv
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

# ---------- Config ----------
load_dotenv()
TOKEN = os.environ.get("DISCORD_TOKEN")
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", ".")).resolve()
VIEWER_CHUNK = os.environ.get("VIEWER_CHUNK", "64")
VIEWER_SCALE = os.environ.get("VIEWER_SCALE", "1.0")
WORK_DIR = Path(os.environ.get("WORK_DIR", "data")).resolve()

# If set, we sync commands to this guild immediately (faster availability)
GUILD_ID_ENV = os.environ.get("GUILD_ID")
TEST_GUILD = discord.Object(id=int(GUILD_ID_ENV)) if GUILD_ID_ENV else None

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
intents = discord.Intents.default()  # slash commands don't need message_content intent
bot = commands.Bot(command_prefix="!", intents=intents)

# Per-guild latest output folder, and a global semaphore to serialize heavy work
LATEST: Dict[int, Path] = {}
WORK_SEM = asyncio.Semaphore(1)

# ---------- Health server (Render expects an open port) ----------
async def _health_server():
    from aiohttp import web
    async def ok(_): return web.Response(text="ok")
    app = web.Application()
    app.router.add_get("/", ok)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 10000)
    await site.start()

@bot.event
async def on_ready():
    try:
        if TEST_GUILD:
            # Guild-only first (instant), then global
            await bot.tree.sync(guild=TEST_GUILD)
            # Copy all commands to global so other servers get them after cache propagation
            bot.tree.copy_global_to(guild=TEST_GUILD)
        # Global sync (can take up to ~1h to fully propagate, but usually minutes)
        await bot.tree.sync()
        print("Slash commands synced.")
    except Exception as e:
        print(f"Slash sync failed: {e}", file=sys.stderr)
    bot.loop.create_task(_health_server())
    print(f"Logged in as {bot.user} (id={bot.user.id})")

# ---------- Helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)

def load_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames or []
        rows = [row for row in rdr]
    return headers, rows

def find_first_numeric_header(headers: List[str], rows: List[Dict[str, str]]) -> Optional[str]:
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
                pass
    return None

def sort_rows(rows: List[Dict[str, str]], key: str) -> List[Dict[str, str]]:
    def to_num(row: Dict[str, str]) -> float:
        try:
            return float(row.get(key, ""))
        except Exception:
            return float("-inf")
    return sorted(rows, key=to_num, reverse=True)

def format_table(headers: List[str], rows: List[Dict[str, str]], limit: int) -> str:
    headers = [h for h in headers if h != "tag"]
    headers = [h for h in headers if any((row.get(h, "") != "") for row in rows)]
    if not headers:
        return "```(no columns to display)```"

    widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows[:limit])) for h in headers}
    head = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    body = []
    for r in rows[:limit]:
        body.append(" | ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers))
    return "```\n" + head + "\n" + sep + "\n" + "\n".join(body) + "\n```"

async def run_viewer(save_path: Path, work_dir: Path) -> Tuple[Optional[Path], List[Path]]:
    ensure_dir(work_dir)
    out_map = work_dir / MAP_NAME
    cmd = [
        sys.executable, "eu4_viewer.py",
        str(save_path),
        "--assets", str(ASSETS_DIR),
        "--out", str(out_map),
        "--chunk", str(VIEWER_CHUNK),
        "--scale", str(VIEWER_SCALE),
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

    csv_paths: List[Path] = []
    for name in CSV_FILES:
        p = ASSETS_DIR / name
        if p.exists():
            csv_paths.append(p)
    return (out_map if out_map.exists() else None, csv_paths)

def build_files(map_path: Optional[Path], csv_paths: List[Path]) -> List[discord.File]:
    files: List[discord.File] = []
    if map_path and map_path.exists():
        files.append(discord.File(str(map_path), filename=MAP_NAME))
    for p in csv_paths:
        try:
            files.append(discord.File(str(p), filename=p.name))
        except Exception:
            pass
    return files

def latest_dir_for_guild(guild_id: int) -> Path:
    p = WORK_DIR / str(guild_id) / "latest"
    ensure_dir(p)
    return p

# ---------- Commands (bind directly to bot.tree) ----------
@bot.tree.command(name="submit", description="Upload a .eu4 save to generate a map and stats tables", guild=TEST_GUILD)
@app_commands.describe(save="Attach your .eu4 save file")
async def submit(interaction: discord.Interaction, save: discord.Attachment):
    # This job can be slow → defer, but guard against Unknown Interaction
    try:
        await interaction.response.defer(ephemeral=False, thinking=True)
    except discord.NotFound:
        # Rare race; we'll just proceed and use followup/send later if possible
        pass

    if not save.filename.lower().endswith(".eu4"):
        # If we already responded, use followup; else use response
        if interaction.response.is_done():
            return await interaction.followup.send("Please upload a `.eu4` file.")
        else:
            return await interaction.response.send_message("Please upload a `.eu4` file.")

    async with WORK_SEM:
        tmpdir = Path(tempfile.mkdtemp(prefix="eu4job_"))
        try:
            local = tmpdir / save.filename
            await save.save(str(local))

            # run viewer
            map_path, csv_paths = await run_viewer(local, tmpdir)

            # move results to per-guild "latest"
            gid = interaction.guild_id or 0
            outdir = latest_dir_for_guild(gid)
            shutil.rmtree(outdir, ignore_errors=True)
            outdir.mkdir(parents=True, exist_ok=True)

            if map_path and map_path.exists():
                shutil.copy2(map_path, outdir / MAP_NAME)
            for p in csv_paths:
                if p.exists():
                    shutil.copy2(p, outdir / p.name)
            LATEST[gid] = outdir

            files = build_files(outdir / MAP_NAME, [outdir / n for n in CSV_FILES if (outdir / n).exists()])
            msg = "Processed your save. Map + CSVs below."
            try:
                if interaction.response.is_done():
                    if files:
                        await interaction.followup.send(msg, files=files)
                    else:
                        await interaction.followup.send(msg)
                else:
                    if files:
                        await interaction.response.send_message(msg, files=files)
                    else:
                        await interaction.response.send_message(msg)
            finally:
                for f in files:
                    try:
                        fp = getattr(f, "fp", None)
                        if fp and not fp.closed:
                            fp.close()
                    except Exception:
                        pass
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

@bot.tree.command(name="map", description="Show the latest processed world map", guild=TEST_GUILD)
async def map_cmd(interaction: discord.Interaction):
    # No defer here → avoids 10062 edge cases for fast replies
    gid = interaction.guild_id or 0
    base = LATEST.get(gid) or latest_dir_for_guild(gid)
    img = base / MAP_NAME
    if not img.exists():
        return await interaction.response.send_message("No map yet. Use `/submit` first.")
    f = discord.File(str(img), filename=MAP_NAME)
    try:
        await interaction.response.send_message(files=[f])
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

@bot.tree.command(name="table", description="Show a stats table (default top 10, sorted desc)", guild=TEST_GUILD)
@app_commands.describe(
    category="Which table (overview/economy/military/development/technology/legitimacy)",
    limit="How many rows (1–100, default 10)",
    column="Column to sort by (desc). If omitted, first numeric column."
)
@app_commands.choices(category=[app_commands.Choice(name=k, value=k) for k in CATEGORIES.keys()])
async def table_cmd(interaction: discord.Interaction, category: app_commands.Choice[str], limit: Optional[int] = 10, column: Optional[str] = None):
    # No defer here (fast)
    gid = interaction.guild_id or 0
    base = LATEST.get(gid) or latest_dir_for_guild(gid)
    csv_path = base / CATEGORIES[category.value]
    if not csv_path.exists():
        return await interaction.response.send_message(f"No data for `{category.value}` yet. Use `/submit` first.")

    headers, rows = load_csv_rows(csv_path)
    if not rows:
        return await interaction.response.send_message("No rows in CSV.")

    sort_key = column if column in headers else find_first_numeric_header(headers, rows) or "province_count"
    rows_sorted = sort_rows(rows, sort_key)
    limit = max(1, min(int(limit or 10), 100))
    table_txt = format_table(headers, rows_sorted, limit)
    await interaction.response.send_message(f"**{category.value.capitalize()}** — sorted by **{sort_key}** (desc)\n{table_txt}")

# Force-sync command (owner/admin use)
@bot.tree.command(name="sync", description="Force resync slash commands (admin only)", guild=TEST_GUILD)
async def sync_cmd(interaction: discord.Interaction):
    if not interaction.user.guild_permissions.administrator:
        return await interaction.response.send_message("Admins only.", ephemeral=True)
    try:
        if TEST_GUILD:
            await bot.tree.sync(guild=TEST_GUILD)
        await bot.tree.sync()
        await interaction.response.send_message("Synced commands.")
    except Exception as e:
        await interaction.response.send_message(f"Sync failed: {e}")

# ---------- Entrypoint ----------
if __name__ == "__main__":
    if not TOKEN:
        print("Missing DISCORD_TOKEN env var", file=sys.stderr)
        sys.exit(1)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.ensure_future(bot.close()))
        except NotImplementedError:
            pass

    bot.run(TOKEN)
