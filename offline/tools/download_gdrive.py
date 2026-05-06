"""Download a file or folder from Google Drive into offline/data/.

Uses `gdown` so public Drive links with confirmation tokens (large files)
and shared folders both work without OAuth.

Usage:
    python -m tools.download_gdrive <url-or-id>
    python -m tools.download_gdrive <url-or-id> --out my_file.zip
    python -m tools.download_gdrive <folder-url> --folder
    python -m tools.download_gdrive <url-or-id> --unzip

Accepts any of:
    https://drive.google.com/file/d/<ID>/view?usp=sharing
    https://drive.google.com/open?id=<ID>
    https://drive.google.com/drive/folders/<ID>
    <ID>                     (raw 33-char Drive ID)
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    sys.exit("gdown not installed. Run: pip install gdown")

import typer

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer(add_completion=False, help=__doc__)


def _extract_id(url_or_id: str) -> str | None:
    """Pull the Drive file/folder ID out of common URL shapes."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/folders/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", url_or_id):
        return url_or_id
    return None


@app.command()
def main(
    source: str = typer.Argument(..., help="Drive URL or raw file/folder ID"),
    out: str | None = typer.Option(
        None, "--out", "-o", help="Output filename (single file) or dir name (folder). Defaults to Drive's name."
    ),
    folder: bool = typer.Option(False, "--folder", help="Treat source as a Drive folder."),
    unzip: bool = typer.Option(False, "--unzip", help="Extract a downloaded .zip and delete the archive."),
):
    file_id = _extract_id(source)
    if file_id is None:
        typer.echo(f"Could not parse a Drive ID from: {source!r}", err=True)
        raise typer.Exit(2)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if folder:
        target_dir = DATA_DIR / (out or file_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Downloading folder {file_id} -> {target_dir}")
        gdown.download_folder(id=file_id, output=str(target_dir), quiet=False, use_cookies=False)
        typer.echo(f"Done. Saved into {target_dir}")
        return

    output_path = str(DATA_DIR / out) if out else None
    typer.echo(f"Downloading file {file_id} -> {output_path or DATA_DIR}/")
    saved = gdown.download(
        id=file_id,
        output=output_path or str(DATA_DIR) + "/",
        quiet=False,
    )
    if not saved:
        typer.echo("Download failed (gdown returned None).", err=True)
        raise typer.Exit(1)

    saved_path = Path(saved)
    typer.echo(f"Saved: {saved_path}")

    if unzip:
        if saved_path.suffix.lower() != ".zip":
            typer.echo(f"--unzip requested but {saved_path.name} is not a .zip; skipping.", err=True)
            return
        extract_dir = saved_path.with_suffix("")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        typer.echo(f"Extracting -> {extract_dir}")
        with zipfile.ZipFile(saved_path) as zf:
            zf.extractall(extract_dir)
        saved_path.unlink()
        typer.echo(f"Extracted and removed archive. Contents at {extract_dir}")


if __name__ == "__main__":
    app()
