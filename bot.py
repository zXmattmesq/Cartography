#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cartography — Discord bot wrapper for eu4_viewer.py (RAM-friendly)

Commands
- /submit  : upload a .eu4 save → generate & store map + CSVs (only the map is posted)
- /map     : post the latest map for this server
- /table   : render a stats table from the stored CSVs (auto-sort by first numeric col)
- /sync    : admin utility to force-sync slash commands

Hosting notes
- Works as a Web Service (tiny HTTP health server on :10000) or as a Worker.
- Uses a single asyncio.Semaphore to serialize heavy jobs under low CPU/RAM.
"""

from __future__ import annotations
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

# ========= Env / Config =========
load_dotenv(override=False)

TOKEN = os.environ.get("DISCORD_TOKEN")
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", ".")).resolve()
WORK_DIR = Path(os.environ.get("WORK_DIR", "data")).resolve()

# Map renderer knobs (keep tiny for low memory plans)
VIEWER_CHUNK = os.environ.get("VIEWER_CHUNK", "64")
VIEWER_SCALE = os.environ.get("VIEWER_SCALE", "0.75")

# Fast guild sync (optional)
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

# ========= Bot =========
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# Per-guild cache of latest artifacts + global work semaphore
LATEST: Dict[int, Path] = {}
WORK_SEM = asyncio.Semaphore(1)


# ========= Health server (port binding for web services) =========
async def _health_server():
    from aiohttp import web

    async def ok(_):
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", ok)
    app.router.add_get("/healthz", ok)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 10000)
    await site.start()


# ========= Helpers =========
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def latest_dir_for_guild(guild_id: int) -> Path:
    p = WORK_DIR / str(guild_id) / "latest"
    ensure_dir(p)
    return p


def load_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames or []
        return headers, [row for row in rdr]


def first_numeric_header(headers: List[str], rows: List[Dict[str, str]]) -> Optional[str]:
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


def sort_rows_desc(rows: List[Dict[str, str]], key: str) -> List[Dict[str, str]]:
    def to_num(r: Dict[str, str]) -> float:
        try:
            return float(r.get(key, ""))
        except Exception:
            return float("-inf")

    return sorted(rows, key=to_num, reverse=True)


def fmt_table(headers: List[str], rows: List[Dict[str, str]], limit: int) -> str:
    headers = [h for h in headers if h != "tag"]
    headers = [h for h in headers if any((r.get(h, "") != "") for r in rows)]
    if not headers:
        return "```(no columns to display)```"
    widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows[:limit])) for h in headers}
    head = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    body = [" | ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers) for r in rows[:limit]]
    return "```\n" + head + "\n" + sep + "\n" + "\n".join(body) + "\n```"


async def run_viewer(save_path: Path, work_dir: Path) -> Tuple[Optional[Path], List[Path]]:
    """
    Execute eu4_viewer.py with low-RAM params.
    Returns (map_path_if_exists, [csv_paths_found_in_assets_dir])
    """
    ensure_dir(work_dir)
    out_map = work_dir / MAP_NAME

    cmd = [
        sys.executable,
        "eu4_viewer.py",
        str(save_path),
        "--assets",
        str(ASSETS_DIR),
        "--out",
        str(out_map),
        "--chunk",
        str(VIEWER_CHUNK),
        "--scale",
        str(VIEWER_SCALE),
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


# ========= Commands =========
@bot.event
async def on_ready():
    # health HTTP
    bot.loop.create_task(_health_server())

    try:
        if TEST_GUILD:
            await bot.tree.sync(guild=TEST_GUILD)  # instant for this guild
            bot.tree.copy_global_to(guild=TEST_GUILD)
        await bot.tree.sync()  # global (may take minutes to propagate)
        print("Slash commands synced.")
    except Exception as e:
        print(f"Slash sync failed: {e}", file=sys.stderr)

    print(f"Logged in as {bot.user} (id={bot.user.id})")


@bot.tree.command(name="submit", description="Upload a .eu4 save to generate a map (stats are stored for /table).", guild=TEST_GUILD)
@app_commands.describe(save="Attach your .eu4 save file")
async def submit(interaction: discord.Interaction, save: discord.Attachment):
    # Long job → try to defer, but handle the rare race where the token expires
    try:
        await interaction.response.defer(ephemeral=False, thinking=True)
    except discord.NotFound:
        pass

    if not save.filename.lower().endswith(".eu4"):
        if interaction.response.is_done():
            return await interaction.followup.send("Please upload a `.eu4` file.")
        return await interaction.response.send_message("Please upload a `.eu4` file.")

    async with WORK_SEM:
        tmpdir = Path(tempfile.mkdtemp(prefix="eu4job_"))
        try:
            local = tmpdir / save.filename
            await save.save(str(local))

            map_path, csv_paths = await run_viewer(local, tmpdir)

            gid = interaction.guild_id or 0
            outdir = latest_dir_for_guild(gid)
            shutil.rmtree(outdir, ignore_errors=True)
            outdir.mkdir(parents=True, exist_ok=True)

            # keep results for this guild
            if map_path and map_path.exists():
                shutil.copy2(map_path, outdir / MAP_NAME)
            for p in csv_paths:
                if p.exists():
                    shutil.copy2(p, outdir / p.name)

            LATEST[gid] = outdir

            # Only send the map now
            map_out = outdir / MAP_NAME
            if map_out.exists():
                f = discord.File(str(map_out), filename=MAP_NAME)
                try:
                    msg = "Processed your save. **Map below.** Use `/table` to view stats."
                    if interaction.response.is_done():
                        await interaction.followup.send(msg, files=[f])
                    else:
                        await interaction.response.send_message(msg, files=[f])
                finally:
                    fp = getattr(f, "fp", None)
                    if fp and not fp.closed:
                        fp.close()
            else:
                if interaction.response.is_done():
                    await interaction.followup.send("Processed, but no map was produced.")
                else:
                    await interaction.response.send_message("Processed, but no map was produced.")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


@bot.tree.command(name="map", description="Show the latest processed world map for this server.", guild=TEST_GUILD)
async def map_cmd(interaction: discord.Interaction):
    gid = interaction.guild_id or 0
    outdir = LATEST.get(gid) or latest_dir_for_guild(gid)
    img = outdir / MAP_NAME
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


@bot.tree.command(name="table", description="Show a stats table (default top 10, sorted by first numeric column).", guild=TEST_GUILD)
@app_commands.describe(
    category="overview/economy/military/development/technology/legitimacy",
    limit="How many rows (1–100, default 10)",
    column="Sort by this column (desc); if omitted, first numeric column."
)
@app_commands.choices(category=[app_commands.Choice(name=k, value=k) for k in CATEGORIES.keys()])
async def table_cmd(
    interaction: discord.Interaction,
    category: app_commands.Choice[str],
    limit: Optional[int] = 10,
    column: Optional[str] = None,
):
    gid = interaction.guild_id or 0
    outdir = LATEST.get(gid) or latest_dir_for_guild(gid)
    csv_path = outdir / CATEGORIES[category.value]
    if not csv_path.exists():
        return await interaction.response.send_message(f"No `{category.value}` data yet. Use `/submit` first.")

    headers, rows = load_csv_rows(csv_path)
    if not rows:
        return await interaction.response.send_message("No rows available.")

    sort_key = column if (column and column in headers) else first_numeric_header(headers, rows) or "province_count"
    rows_sorted = sort_rows_desc(rows, sort_key)
    limit = max(1, min(int(limit or 10), 100))
    table = fmt_table(headers, rows_sorted, limit)
    await interaction.response.send_message(f"**{category.value.capitalize()}** — sorted by **{sort_key}** (desc)\n{table}")


@bot.tree.command(name="sync", description="Force-resync slash commands (admins only).", guild=TEST_GUILD)
async def sync_cmd(interaction: discord.Interaction):
    if not interaction.user.guild_permissions.administrator:
        return await interaction.response.send_message("Admins only.", ephemeral=True)
    try:
        if TEST_GUILD:
            await bot.tree.sync(guild=TEST_GUILD)
        await bot.tree.sync()
        await interaction.response.send_message("Synced.")
    except Exception as e:
        await interaction.response.send_message(f"Sync failed: {e}")


# ========= Entrypoint =========
if __name__ == "__main__":
    if not TOKEN:
        print("Missing DISCORD_TOKEN env var", file=sys.stderr)
        sys.exit(1)

    loop = asyncio.get_event_loop()
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig:
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.ensure_future(bot.close()))
            except Exception:
                pass

    bot.run(TOKEN)
