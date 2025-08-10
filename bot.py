# bot.py
import asyncio
import csv
import io
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands

import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# -----------------------------
# Config / paths
# -----------------------------
TOKEN = os.environ.get("DISCORD_TOKEN")
if not TOKEN:
    print("Missing DISCORD_TOKEN env var")
    raise SystemExit(1)

# Where our static assets live (provinces.bmp, definition.csv, default.map, 00_country_colors.txt)
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", ".")).resolve()
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Default rendering knobs (safe for low memory)
CHUNK = int(os.environ.get("RENDER_CHUNK", "64"))
SCALE = float(os.environ.get("RENDER_SCALE", "0.75"))

# Map output filename (we overwrite this on every submit)
LAST_MAP = ASSETS_DIR / "world_map.png"

# Small job queue to avoid multiple heavy renders at once
JOB_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT_JOBS", "1")))
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "420"))  # 7 minutes

# -----------------------------
# Minimal web health server (for PaaS keepalive)
# -----------------------------
class _Health(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"discord bot running")

def _run_health():
    port = int(os.environ.get("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), _Health)
    print(f"HTTP health server running on :{port}")
    server.serve_forever()

threading.Thread(target=_run_health, daemon=True).start()

# -----------------------------
# Discord setup
# -----------------------------
intents = discord.Intents.none()  # we only need interactions
client = commands.Bot(command_prefix="!", intents=intents)
tree = client.tree

# -----------------------------
# Utility
# -----------------------------
def _is_num(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def _first_numeric_col(columns: List[str], rows: List[Dict[str, str]]) -> str:
    for c in columns:
        if any(_is_num(r.get(c, "")) for r in rows):
            return c
    return columns[0] if columns else "name"

def _render_table(rows: List[Dict[str, str]], columns: List[str], limit: int) -> str:
    rows = rows[:limit]
    widths = [len(c.upper()) for c in columns]
    for r in rows:
        for i, c in enumerate(columns):
            widths[i] = max(widths[i], len(str(r.get(c, ""))))
    header = "  ".join(h.upper().ljust(widths[i]) for i, h in enumerate(columns))
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        line = "  ".join(str(r.get(c, "")).ljust(widths[i]) for i, c in enumerate(columns))
        lines.append(line)
    return "\n".join(lines)

def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        rows = [r for r in rdr]
    return cols, rows

def _sort_rows(rows: List[Dict[str, str]], sort_by: str) -> List[Dict[str, str]]:
    def keyfun(r):
        v = r.get(sort_by, "")
        try:
            return float(v)
        except Exception:
            return -float("inf")
    return sorted(rows, key=keyfun, reverse=True)

# -----------------------------
# Views (column sorting)
# -----------------------------
class TableView(discord.ui.View):
    def __init__(
        self,
        category: str,
        columns: List[str],
        rows: List[Dict[str, str]],
        top: int,
        current_sort: str,
        assets_dir: Path,
    ):
        super().__init__(timeout=300)
        self.category = category
        self.columns = columns
        self.rows = rows
        self.top = max(1, min(int(top), 200))
        self.sort_by = current_sort
        self.assets_dir = assets_dir

        # Build select with display columns (hide 'tag')
        display_cols = [c for c in self.columns if c != "tag"]
        if "name" in display_cols:
            display_cols.remove("name")
            display_cols = ["name"] + display_cols

        options = [
            discord.SelectOption(label=c, value=c, default=(c == self.sort_by))
            for c in display_cols
        ]
        # Discord limit: max 25 options; trim if necessary (rare)
        options = options[:25]

        self.select = discord.ui.Select(placeholder="Sort by column…", options=options, min_values=1, max_values=1)
        self.select.callback = self._on_select  # type: ignore
        self.add_item(self.select)

    async def _on_select(self, interaction: discord.Interaction):
        self.sort_by = self.select.values[0]
        await self._refresh(interaction)

    async def _refresh(self, interaction: discord.Interaction):
        # sort + render preview
        rows_sorted = _sort_rows(self.rows, self.sort_by)
        display_cols = [c for c in self.columns if c != "tag"]
        if "name" in display_cols:
            display_cols.remove("name")
            display_cols = ["name"] + display_cols

        preview_rows = rows_sorted[: min(self.top, 15)]
        preview_text = _render_table(preview_rows, display_cols, limit=len(preview_rows))
        title = f"{self.category.capitalize()} — top {self.top} (sorted by {self.sort_by}, desc)"
        content = f"**{title}**\n```{preview_text}```"

        # Attach full table + CSV every time (keeps message short, sortable stays snappy)
        full_table_text = _render_table(rows_sorted, display_cols, limit=self.top)
        buf_txt = io.BytesIO(full_table_text.encode("utf-8"))
        buf_txt.name = "table.txt"

        csv_bytes = (self.assets_dir / f"{self.category}.csv").read_bytes()
        buf_csv = io.BytesIO(csv_bytes)
        buf_csv.name = f"{self.category}.csv"

        await interaction.response.edit_message(content=content, attachments=[discord.File(buf_txt), discord.File(buf_csv)], view=self)

# -----------------------------
# Slash commands
# -----------------------------
@tree.command(name="ping", description="Check if the bot is alive.")
async def ping_cmd(interaction: discord.Interaction):
    await interaction.response.send_message("Pong!")

@tree.command(name="map", description="Show the last generated world map.")
async def map_cmd(interaction: discord.Interaction):
    if not LAST_MAP.exists():
        await interaction.response.send_message("No map yet. Upload a save with `/submit` first.")
        return
    await interaction.response.send_message(file=discord.File(str(LAST_MAP)))

@tree.command(name="submit", description="Upload a .eu4 save and generate map + stats.")
@app_commands.describe(
    save="Attach a .eu4 save file (zipped saves are OK).",
    chunk="Advanced: row chunk height (default 64)",
    scale="Advanced: downscale factor (default 0.75)"
)
async def submit_save_cmd(
    interaction: discord.Interaction,
    save: discord.Attachment,
    chunk: Optional[int] = None,
    scale: Optional[float] = None,
):
    await interaction.response.defer(ephemeral=False, thinking=True)

    if not save.filename.lower().endswith(".eu4"):
        await interaction.followup.send("Please attach a `.eu4` save file.")
        return

    # Only one heavy render at a time
    async with JOB_SEMAPHORE:
        job_id = f"eu4job_{uuid.uuid4().hex[:8]}"
        tmpdir = Path(tempfile.mkdtemp(prefix=job_id + "_", dir="/tmp"))
        try:
            # download to tmp
            save_path = tmpdir / save.filename
            data = await save.read()
            save_path.write_bytes(data)

            # run eu4_viewer
            cmd = [
                "python", str(Path(__file__).with_name("eu4_viewer.py")),
                str(save_path),
                "--assets", str(ASSETS_DIR),
                "--out", str(LAST_MAP),
                "--chunk", str(chunk or CHUNK),
                "--scale", str(scale or SCALE),
            ]
            start = time.time()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=JOB_TIMEOUT_S)
            except asyncio.TimeoutError:
                proc.kill()
                await interaction.followup.send("⏱️ Processing timed out. Try a smaller map (lower `--scale`) or a shorter save.")
                return

            # log (for troubleshooting)
            if stdout:
                print(stdout.decode("utf-8", errors="ignore"))
            if stderr:
                print(stderr.decode("utf-8", errors="ignore"))

            if proc.returncode != 0:
                await interaction.followup.send("❌ Failed to process the save. Check your assets files and try again.")
                return

            # Success: post a simple message + the map
            if LAST_MAP.exists():
                await interaction.followup.send(
                    "Processed your save. **Map below.** Use **/table** to view stats.",
                    file=discord.File(str(LAST_MAP)),
                )
            else:
                await interaction.followup.send("Processed your save. Use **/map** to view the world map and **/table** to view stats.")

            dur = time.time() - start
            print(f"Completed job in {dur:.1f}s → {LAST_MAP}")

        finally:
            # best effort cleanup
            try:
                for p in tmpdir.glob("*"):
                    p.unlink(missing_ok=True)
                tmpdir.rmdir()
            except Exception:
                pass

# Table categories available (must match CSV filenames written by eu4_viewer.py)
TABLE_CHOICES = [
    app_commands.Choice(name="Overview", value="overview"),
    app_commands.Choice(name="Economy", value="economy"),
    app_commands.Choice(name="Military", value="military"),
    app_commands.Choice(name="Development", value="development"),
    app_commands.Choice(name="Technology", value="technology"),
    app_commands.Choice(name="Legitimacy", value="legitimacy"),
]

@tree.command(name="table", description="Show a stats table (sortable).")
@app_commands.describe(
    category="Which table to show",
    top="How many rows (default 10, up to 200)",
    sort_by="Sort column (optional; you can change later with the dropdown)"
)
@app_commands.choices(category=TABLE_CHOICES)
async def table_cmd(
    interaction: discord.Interaction,
    category: app_commands.Choice[str],
    top: Optional[int] = 10,
    sort_by: Optional[str] = None,
):
    await interaction.response.defer(ephemeral=False, thinking=True)

    csv_path = ASSETS_DIR / f"{category.value}.csv"
    if not csv_path.exists():
        await interaction.followup.send(f"Couldn't find `{category.value}.csv`. Upload a save first with `/submit`.")
        return

    columns, rows = _read_csv(csv_path)
    if not rows:
        await interaction.followup.send("No data yet. Did the save parse correctly?")
        return

    top = max(1, min(int(top or 10), 200))

    # Choose default sort
    if sort_by is None or sort_by not in columns:
        sort_by = _first_numeric_col(columns, rows)

    rows_sorted = _sort_rows(rows, sort_by)

    # Choose display columns (hide 'tag', put 'name' first if present)
    display_cols = [c for c in columns if c != "tag"]
    if "name" in display_cols:
        display_cols.remove("name")
        display_cols = ["name"] + display_cols

    # Build preview
    preview_rows = rows_sorted[: min(top, 15)]
    preview = _render_table(preview_rows, display_cols, limit=len(preview_rows))
    title = f"{category.value.capitalize()} — top {top} (sorted by {sort_by}, desc)"
    content = f"**{title}**\n```{preview}```"

    # Attach full table and the raw CSV
    full_table_text = _render_table(rows_sorted, display_cols, limit=top)
    buf_txt = io.BytesIO(full_table_text.encode("utf-8"))
    buf_txt.name = "table.txt"

    csv_bytes = csv_path.read_bytes()
    buf_csv = io.BytesIO(csv_bytes)
    buf_csv.name = f"{category.value}.csv"

    view = TableView(
        category=category.value,
        columns=columns,
        rows=rows,
        top=top,
        current_sort=sort_by,
        assets_dir=ASSETS_DIR,
    )
    await interaction.followup.send(content=content, files=[discord.File(buf_txt), discord.File(buf_csv)], view=view)

# -----------------------------
# Startup / sync
# -----------------------------
@client.event
async def on_ready():
    try:
        await tree.sync()
        print("Slash commands synced.")
    except Exception as e:
        print("Failed to sync commands:", e)
    print(f"Logged in as {client.user} (id={client.user.id})")

client.run(TOKEN)
