#!/usr/bin/env python3
import os
import csv
import math
import shutil
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands, tasks
from aiohttp import web
from concurrent.futures import ProcessPoolExecutor
import uuid
import sys
import subprocess

# ======================== CONFIG ========================
ASSETS_DIR = Path(__file__).parent.resolve()
EU4_SCRIPT = ASSETS_DIR / "eu4_viewer.py"

CATEGORIES = {
    "overview":     {"file": "overview.csv"},
    "economy":      {"file": "economy.csv"},
    "military":     {"file": "military.csv"},
    "development":  {"file": "development.csv"},
    "technology":   {"file": "technology.csv"},
    "legitimacy":   {"file": "legitimacy.csv"},
}

COUNTRY_NAMES_FILE = ASSETS_DIR / "country_names.csv"

DEFAULT_TABLE_LIMIT = 10
RETENTION_HOURS = int(os.getenv("RETENTION_HOURS", "8"))
MAX_TOTAL_STORAGE_MB = int(os.getenv("MAX_TOTAL_STORAGE_MB", "300"))
WORKDIR_ROOT = Path(os.getenv("WORKDIR_ROOT", tempfile.gettempdir())) / "eu4bot_runs"
MAX_CONCURRENCY = max(1, int(os.getenv("MAX_CONCURRENCY", "1")))
MAX_SAVE_MB = int(os.getenv("MAX_SAVE_MB", "80"))

# Discord (no privileged intents needed for slash commands)
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# Concurrency + simple queue metrics
EXEC = ProcessPoolExecutor(max_workers=1)     # one worker process keeps RAM stable
JOB_SEM = asyncio.Semaphore(MAX_CONCURRENCY)  # gate how many run at once
RUNNING_JOBS = 0
PENDING_JOBS = 0

# Last processed workspace per guild
LATEST_WORKDIR: Dict[int, Path] = {}

# ======================== STORAGE MANAGER ========================
def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try: total += p.stat().st_size
            except Exception: pass
    return total

def human_bytes(n: int) -> str:
    if n < 1024: return f"{n} B"
    units = ["KB","MB","GB","TB"]
    i = int(math.log(n, 1024))
    return f"{n / (1024**i):.1f} {units[i-1]}"

def workdir_for_guild(guild_id: int) -> Path:
    d = WORKDIR_ROOT / str(guild_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

def all_workdirs() -> List[Path]:
    if not WORKDIR_ROOT.exists(): return []
    return [p for p in WORKDIR_ROOT.iterdir() if p.is_dir()]

def mark_touch(path: Path) -> None:
    now = time.time()
    try: os.utime(path, (now, now))
    except Exception: pass
    for p in path.glob("*"):
        try: os.utime(p, (now, now))
        except Exception: pass

def purge_dir(path: Path) -> None:
    try: shutil.rmtree(path, ignore_errors=True)
    except Exception: pass

def current_total_bytes() -> int:
    return sum(dir_size_bytes(p) for p in all_workdirs())

def auto_purge() -> None:
    cutoff = time.time() - RETENTION_HOURS * 3600
    for wd in all_workdirs():
        try:
            if wd.stat().st_mtime < cutoff:
                purge_dir(wd)
        except Exception:
            purge_dir(wd)

    max_bytes = MAX_TOTAL_STORAGE_MB * 1024 * 1024
    total = current_total_bytes()
    if total <= max_bytes:
        return

    remaining = []
    for p in all_workdirs():
        try: remaining.append((p.stat().st_mtime, p))
        except Exception: purge_dir(p)
    remaining.sort(key=lambda x: x[0])  # oldest first
    for _, p in remaining:
        if current_total_bytes() <= max_bytes:
            break
        purge_dir(p)

@tasks.loop(minutes=15)
async def janitor():
    auto_purge()

# ======================== COUNTRY NAMES (optional) ========================
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

def best_country_name(row: dict) -> str:
    tag = (row.get("tag") or "").upper()
    if tag in TAG_TO_NAME:
        return TAG_TO_NAME[tag]
    for key in ("country_name", "localized_name", "long_name"):
        if row.get(key):
            return str(row[key])[:40]
    n = (row.get("name") or "").strip()
    if n and len(n) <= 24 and not any(ch.isdigit() for ch in n):
        return n
    return tag or "Unknown"

# ======================== CSV HELPERS ========================
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
    if not rows: return []
    cols = list(rows[0].keys())
    nums = [c for c in cols if c not in ("tag","name") and any(is_number(r.get(c)) for r in rows)]
    ordered = [c for c in preferred_order if c in nums]
    for c in nums:
        if c not in ordered: ordered.append(c)
    return ordered

def sort_rows_by(rows: List[dict], key: str) -> List[dict]:
    def keyfn(r):
        v = r.get(key)
        try: return float(v) if v not in (None, "") else float("-inf")
        except Exception: return float("-inf")
    return sorted(rows, key=keyfn, reverse=True)

def coerce_display_rows(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        rr = dict(r)
        rr["name"] = best_country_name(r)
        rr.pop("tag", None)
        out.append(rr)
    return out

def default_cols_for(category: str, rows: List[dict]) -> List[str]:
    base = {
        "overview":     ["name","province_count","total_development","avg_development","income","manpower","army_quality_score"],
        "economy":      ["name","income","treasury","inflation","loans","interest","war_exhaustion","corruption"],
        "military":     ["name","army_quality_score","manpower","max_manpower","land_forcelimit",
                         "army_tradition","army_professionalism","discipline","land_morale"],
        "development":  ["name","total_development","avg_development","province_count"],
        "technology":   ["name","mil_tech","adm_tech","dip_tech"],
        "legitimacy":   ["name","absolutism","legitimacy","republican_tradition","devotion",
                         "horde_unity","meritocracy","government_reform_progress","prestige","stability"],
    }.get(category)
    if not base:
        return [c for c in rows[0].keys() if c != "tag"]
    have = rows[0].keys()
    base = [c for c in base if c in have]
    if "name" not in base and "name" in have:
        base = ["name"] + base
    return base

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

def monospace_table(rows: List[dict], cols: List[str], limit: int) -> str:
    rows = rows[:limit]
    def fmt(x: Optional[str]) -> str:
        if x is None or x == "": return "-"
        try:
            fx = float(x)
            if abs(fx - int(fx)) < 1e-9: return f"{int(fx)}"
            return f"{fx:.2f}"
        except Exception:
            return str(x)

    for r in rows:
        if "name" in r and r["name"]:
            r["name"] = str(r["name"])[:22]

    widths = {}
    for c in cols:
        head = c.upper()
        maxw = len(head)
        for r in rows:
            v = fmt(r.get(c))
            if len(v) > maxw: maxw = len(v)
        widths[c] = max(4, min(maxw, 22))

    header = " ".join(f"{c.upper():<{widths[c]}}" for c in cols)
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(" ".join(f"{fmt(r.get(c)):<{widths[c]}}" for c in cols))
    return "```\n" + "\n".join(lines) + "\n```"

# ============== Column picker UI ==============
class ColumnPicker(discord.ui.Select):
    def __init__(self, category: str, rows_original: List[dict], default_key: Optional[str]):
        options = []
        nums = numeric_columns(rows_original, [])
        for c in nums[:25]:
            options.append(discord.SelectOption(label=c, value=c, default=(c == default_key)))
        super().__init__(placeholder="Sort by…", min_values=1, max_values=1, options=options)
        self.category = category
        self.rows_original = rows_original
        self.default_key = default_key

    async def callback(self, interaction: discord.Interaction):
        key = self.values[0]
        disp = coerce_display_rows(self.rows_original)
        cols = default_cols_for(self.category, disp)
        disp_sorted = sort_rows_by(disp, key)
        table_txt = monospace_table(disp_sorted, cols, DEFAULT_TABLE_LIMIT)
        await interaction.response.edit_message(
            content=f"**{self.category.capitalize()} — Top {DEFAULT_TABLE_LIMIT} (sorted by {key})**\n{table_txt}",
            view=self.view
        )

class ColumnPickerView(discord.ui.View):
    def __init__(self, category: str, rows_original: List[dict], default_key: Optional[str]):
        super().__init__(timeout=180)
        self.add_item(ColumnPicker(category, rows_original, default_key))

# ======================== HEAVY WORK (separate process) ========================
def build_outputs(save_path: str, workdir: str) -> dict:
    """
    Runs eu4_viewer.py on save_path, writes outputs into workdir, returns paths.
    This runs inside a separate process (ProcessPoolExecutor) to keep bot RAM/CPU low.
    """
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    out_png = wd / "world_map.png"

    script = EU4_SCRIPT
    assets = ASSETS_DIR

    log_path = wd / "build.log"
    with log_path.open("w") as log:
        subprocess.run(
            [sys.executable, str(script), save_path, "--assets", str(assets), "--out", str(out_png)],
            stdout=log, stderr=log, check=True
        )

    csvs = {}
    for name in CATEGORIES.keys():
        p = assets / f"{name}.csv"
        if p.exists():
            # copy CSVs into workdir, so each guild has its own snapshot
            dest = wd / p.name
            shutil.copy2(p, dest)
            csvs[name] = str(dest)

    # delete the uploaded save to save space
    try: Path(save_path).unlink(missing_ok=True)
    except Exception: pass

    return {"map": str(out_png), "csvs": csvs, "log": str(log_path)}

# ======================== COMMANDS ========================
@bot.event
async def on_ready():
    try:
        await bot.tree.sync()
    except Exception as e:
        print("Slash sync failed:", e)
    WORKDIR_ROOT.mkdir(parents=True, exist_ok=True)
    janitor.start()
    print(f"Logged in as {bot.user} (id={bot.user.id})")

@bot.tree.command(name="submit_save", description="Upload a EU4 save; I’ll render the map + stats tables.")
@app_commands.describe(savefile="Attach a .eu4 save file")
async def submit_save_cmd(interaction: discord.Interaction, savefile: discord.Attachment):
    global PENDING_JOBS
    await interaction.response.defer(thinking=True, ephemeral=False)

    if savefile.size > MAX_SAVE_MB * 1024 * 1024:
        return await interaction.edit_original_response(
            content=f"❌ Save is too large for this free plan (>{MAX_SAVE_MB} MB).")

    guild_id = interaction.guild_id or interaction.user.id
    workdir = workdir_for_guild(guild_id)
    LATEST_WORKDIR[guild_id] = workdir

    job_id = uuid.uuid4().hex[:8]
    local_path = workdir / f"{job_id}_{savefile.filename}"
    await interaction.edit_original_response(content=f"📥 Downloading `{savefile.filename}` …")
    await savefile.save(local_path)

    # Queue note
    queued_note = ""
    if JOB_SEM.locked() or RUNNING_JOBS >= MAX_CONCURRENCY:
        PENDING_JOBS += 1
        queued_note = f"\n⏳ Server is busy. Your job is **queued** (Position ~{PENDING_JOBS})."

    await interaction.edit_original_response(content=f"🧮 Parsing & rendering…{queued_note}")

    async def _do_job():
        global RUNNING_JOBS, PENDING_JOBS
        async with JOB_SEM:
            if PENDING_JOBS > 0:
                PENDING_JOBS -= 1
            RUNNING_JOBS += 1
            try:
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(EXEC, build_outputs, str(local_path), str(workdir)),
                    timeout=900  # 15 minutes hard cap
                )
            except asyncio.TimeoutError:
                await interaction.followup.send("⏱️ Timed out building this save (15 min). Try a smaller file.")
                return
            except subprocess.CalledProcessError:
                await interaction.followup.send("❌ Failed while generating outputs. Check your assets or try another save.")
                return
            except Exception as e:
                await interaction.followup.send(f"❌ Unexpected error: {type(e).__name__}: {e}")
                return
            finally:
                RUNNING_JOBS -= 1
                mark_touch(workdir)
                auto_purge()

            # Send map
            files = []
            map_file = Path(result["map"])
            if map_file.exists():
                files.append(discord.File(str(map_file), filename="world_map.png"))
            msg = "✅ Processed your save."
            await interaction.followup.send(msg, files=files if files else None)

            # Notify tables ready
            await interaction.followup.send(
                "📊 Tables updated. Try `/table category:overview` (or economy, military, development, technology, legitimacy).",
                ephemeral=False
            )

            # Keep only latest 3 job artifacts per guild
            try:
                all_items = sorted(workdir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                job_dirs = [p for p in all_items if p.is_dir()]
                for d in job_dirs[3:]:
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

    # Kick background task so the event loop stays responsive
    asyncio.create_task(_do_job())

@bot.tree.command(name="map", description="Send the last rendered world map for this server.")
async def map_cmd(interaction: discord.Interaction):
    guild_id = interaction.guild_id or interaction.user.id
    workdir = LATEST_WORKDIR.get(guild_id)
    if not workdir:
        return await interaction.response.send_message("No processed save yet. Use **/submit_save** first.")
    path = workdir / "world_map.png"
    if not path.exists():
        # fallback: find any map in workdir
        maps = list(workdir.glob("**/world_map.png"))
        if maps:
            path = maps[0]
        else:
            return await interaction.response.send_message("I couldn't find a map for the last run.")
    mark_touch(workdir)
    await interaction.response.send_message(file=discord.File(str(path), filename="world_map.png"))

@bot.tree.command(name="categories", description="List all table categories you can show.")
async def categories_cmd(interaction: discord.Interaction):
    cats = ", ".join(CATEGORIES.keys())
    await interaction.response.send_message(f"Available categories: **{cats}**")

@bot.tree.command(name="table", description="Show a top-N table; sorted by the first numeric column (dropdown to change).")
@app_commands.describe(
    category="Which table (overview, economy, military, development, technology, legitimacy)",
    limit=f"How many rows (default {DEFAULT_TABLE_LIMIT}, max 100)"
)
@app_commands.choices(category=[app_commands.Choice(name=n, value=n) for n in CATEGORIES.keys()])
async def table_cmd(interaction: discord.Interaction, category: app_commands.Choice[str], limit: Optional[int] = None):
    await interaction.response.defer(thinking=True, ephemeral=False)
    guild_id = interaction.guild_id or interaction.user.id
    workdir = LATEST_WORKDIR.get(guild_id)
    if not workdir:
        return await interaction.followup.send("No processed save yet. Use **/submit_save** first.")

    # Prefer CSVs copied into workdir by build_outputs; else fallback to assets
    csv_path = workdir / CATEGORIES[category.value]["file"]
    if not csv_path.exists():
        csv_path = ASSETS_DIR / CATEGORIES[category.value]["file"]
        if not csv_path.exists():
            return await interaction.followup.send(f"I couldn't find `{CATEGORIES[category.value]['file']}`. Please run **/submit_save** again.")

    rows = read_csv_rows(csv_path)
    if not rows:
        return await interaction.followup.send("That table is empty.")

    disp = coerce_display_rows(rows)
    cols = default_cols_for(category.value, disp)
    sort_key = auto_pick_sort_column(category.value, disp)
    disp_sorted = sort_rows_by(disp, sort_key) if sort_key else disp
    n = max(1, min(limit or DEFAULT_TABLE_LIMIT, 100))
    table_txt = monospace_table(disp_sorted, cols, n)

    view = ColumnPickerView(category.value, rows, sort_key)
    mark_touch(workdir)
    await interaction.followup.send(f"**{category.value.capitalize()} — Top {n} (sorted by {sort_key or '—'})**\n{table_txt}", view=view)

@bot.tree.command(name="status", description="Show storage usage, retention, and concurrency.")
async def status_cmd(interaction: discord.Interaction):
    total = current_total_bytes()
    await interaction.response.send_message(
        f"Storage: **{human_bytes(total)}** across {len(all_workdirs())} servers.\n"
        f"Retention: **{RETENTION_HOURS}h**, Cap: **{MAX_TOTAL_STORAGE_MB} MB**.\n"
        f"Concurrency: **{MAX_CONCURRENCY}** (change via env var). Current: running={RUNNING_JOBS}, queued={PENDING_JOBS}."
    )

@bot.tree.command(name="queue", description="Show how many jobs are running/queued.")
async def queue_cmd(interaction: discord.Interaction):
    await interaction.response.send_message(f"Jobs — running: **{RUNNING_JOBS}**, queued: **{PENDING_JOBS}**.")

@bot.tree.command(name="purge", description="Delete the last processed outputs for this server.")
async def purge_cmd(interaction: discord.Interaction):
    guild_id = interaction.guild_id or interaction.user.id
    wd = workdir_for_guild(guild_id)
    purge_dir(wd)
    LATEST_WORKDIR.pop(guild_id, None)
    await interaction.response.send_message("🧹 Purged this server's outputs.")

# ======================== HTTP HEALTH (Render Web Service) ========================
async def health(_request):
    return web.Response(text="OK")

async def run_web_server():
    app = web.Application()
    app.add_routes([web.get("/", health), web.get("/healthz", health)])
    port = int(os.getenv("PORT", "10000"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    print(f"HTTP health server running on :{port}")

# ======================== ENTRYPOINT ========================
async def main_async():
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Set DISCORD_TOKEN environment variable.")
    WORKDIR_ROOT.mkdir(parents=True, exist_ok=True)
    http_task = asyncio.create_task(run_web_server())
    bot_task = asyncio.create_task(bot.start(token))
    await asyncio.gather(http_task, bot_task)

if __name__ == "__main__":
    asyncio.run(main_async())
