#####################################################################
#       Developed by @moscow_professor                              # 
#       t.me/moscow_professor                                       #
#       If you publish or use my code,                              #
#       please at least leave a link to my profile in Telegram.     #
#####################################################################

#!/usr/bin/env python3

import asyncio
import argparse
import sqlite3
import re
import sys
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from typing import List, Optional, Set, Dict, Any

import httpx
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ====================== CONFIGURATION ======================
DB_FILE = "processed_links.db"
REQUEST_TIMEOUT = 30
MAX_CONCURRENT = 5
BASE_URL = "https://telegra.ph/"

TELEGRAPH_REGEX = re.compile(r"https?://telegra\.ph/[\w-]+")
GRAPH_REGEX = re.compile(r"https?://graph\.org/[\w-]+")
TELEGCRACK_SEARCH_REGEX = re.compile(r"https?://telegcrack\.com/select\.php\?.*q=.*")

# Allowed file types
ALLOWED_IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.gif'}
ALLOWED_VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.m4v'}
ALLOWED_EXT = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT
MAX_VIDEO_SIZE_MB = 50   # default
MAX_IMAGE_SIZE_MB = None # no limit

# NSFW model
MODEL_PATH = "./nsfw_mobilenet2.224x224.h5"
THRESHOLD = 0.45
FILTER_NOT_GIRLS_DIR = "./filtered_not_girls"

# Link generation
MAX_CONCURRENT_CHECKS = 20

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ====================== SHARED HTTP CLIENT SETTINGS ======================
CLIENT_KWARGS = {"verify": False, "follow_redirects": True, "timeout": REQUEST_TIMEOUT}

# ====================== TRANSLITERATION ======================
def translit_ru_to_en(text: str) -> str:
    """Informal phonetic transliteration (e.g., мелкая -> melkaya)"""
    mapping = {
        'щ': 'shch', 'ш': 'sh', 'ч': 'ch', 'ж': 'zh', 'ц': 'ts',
        'ё': 'yo', 'ю': 'yu', 'я': 'ya',
        'ъ': '', 'ь': '',
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
        'е': 'e', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k',
        'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p',
        'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f',
        'х': 'kh', 'э': 'e',
        'Щ': 'Shch', 'Ш': 'Sh', 'Ч': 'Ch', 'Ж': 'Zh', 'Ц': 'Ts',
        'Ё': 'Yo', 'Ю': 'Yu', 'Я': 'Ya',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D',
        'Е': 'E', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K',
        'Л': 'L', 'М': 'M', 'Н': 'N', 'О': 'O', 'П': 'P',
        'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F',
        'Х': 'Kh', 'Э': 'E'
    }
    for rus, lat in sorted(mapping.items(), key=lambda x: -len(x[0])):
        text = text.replace(rus, lat)
    return text

# ====================== DATABASE ======================
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_links(
                link TEXT PRIMARY KEY,
                has_images INTEGER,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur = conn.execute("PRAGMA table_info(processed_links)")
        cols = [row[1] for row in cur.fetchall()]
        if "has_images" not in cols:
            conn.execute("ALTER TABLE processed_links ADD COLUMN has_images INTEGER")
        if "checked_at" not in cols:
            conn.execute("ALTER TABLE processed_links ADD COLUMN checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

def is_checked(link: str) -> Optional[bool]:
    with sqlite3.connect(DB_FILE) as conn:
        row = conn.execute("SELECT has_images FROM processed_links WHERE link = ?", (link,)).fetchone()
        return bool(row[0]) if row else None

def mark_checked(link: str, has_images: bool):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO processed_links(link, has_images) VALUES (?, ?)",
            (link, 1 if has_images else 0)
        )

# ====================== URL EXTRACTION ======================
def extract_real_url(link: str) -> Optional[str]:
    link = link.strip()
    if TELEGRAPH_REGEX.match(link) or GRAPH_REGEX.match(link):
        return link
    if "telegcrack.com/out.php" in link:
        parsed = urlparse(link)
        qs = parse_qs(parsed.query)
        if "to" in qs:
            decoded = unquote(qs["to"][0])
            if TELEGRAPH_REGEX.match(decoded) or GRAPH_REGEX.match(decoded):
                return decoded
    return None

# ====================== HTML AND SEARCH PAGE PARSING ======================
async def parse_html_file(file_path: str) -> List[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    soup = BeautifulSoup(content, "html.parser")
    found_links: Set[str] = set()
    for a in soup.select(".part ol li a"):
        href = a.get("href")
        if href and href.startswith("/out.php?to="):
            full_url = f"https://telegcrack.com{href}" if href.startswith("/") else href
            real = extract_real_url(full_url)
            if real:
                found_links.add(real)
    for a in soup.find_all("a", href=True):
        real = extract_real_url(a["href"])
        if real:
            found_links.add(real)
    print(f"✅ Extracted {len(found_links)} unique articles from {file_path}")
    return list(found_links)

async def parse_telegcrack_search_page(url: str) -> List[str]:
    async with httpx.AsyncClient(**CLIENT_KWARGS) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except Exception as e:
            print(f"Error loading search page {url}: {e}")
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        found_links: Set[str] = set()
        for a in soup.select(".part ol li a"):
            href = a.get("href")
            if href and href.startswith("/out.php?to="):
                full_url = f"https://telegcrack.com{href}" if href.startswith("/") else href
                real = extract_real_url(full_url)
                if real:
                    found_links.add(real)
        for a in soup.find_all("a", href=True):
            real = extract_real_url(a["href"])
            if real:
                found_links.add(real)
        print(f"✅ Found {len(found_links)} unique articles")
        return list(found_links)

# ====================== LINK GENERATION BY KEYWORD ======================
async def check_sequence(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, base_url: str) -> List[str]:
    """Check existence of pages with indices (word-month-day, word-month-day-2, ...)"""
    async with semaphore:
        valid = []
        index = 1
        while True:
            url = base_url if index == 1 else f'{base_url}-{index}'
            try:
                resp = await client.get(url, follow_redirects=True)
                if resp.status_code == 200:
                    valid.append(url)
                    index += 1
                else:
                    break
            except Exception:
                break
        return valid

async def generate_links(word: str) -> List[str]:
    """
    Generate all possible links like https://telegra.ph/word-month-day
    and check if they exist.
    """
    days = [f'{i:02d}' for i in range(1, 32)]
    months = [f'{i:02d}' for i in range(1, 13)]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHECKS)
    valid_urls = []
    async with httpx.AsyncClient(**CLIENT_KWARGS) as client:
        tasks = []
        for month in months:
            for day in days:
                url = f'{BASE_URL}{word}-{month}-{day}'
                tasks.append(check_sequence(client, semaphore, url))
        results = await asyncio.gather(*tasks)
        for sublist in results:
            valid_urls.extend(sublist)
    return valid_urls

async def expand_input_items(items: List[str]) -> List[str]:
    result: List[str] = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if Path(item).exists():
            if item.lower().endswith(".html"):
                links = await parse_html_file(item)
                result.extend(links)
            else:
                with open(item, encoding="utf-8") as f:
                    for line in f:
                        sub = await expand_input_items([line.strip()])
                        result.extend(sub)
            continue
        if TELEGCRACK_SEARCH_REGEX.match(item):
            print(f"🔍 Parsing results page: {item}")
            links = await parse_telegcrack_search_page(item)
            result.extend(links)
            continue
        real = extract_real_url(item)
        if real:
            result.append(real)
        else:
            print(f"⚠️ Unknown format, skipping: {item}")
    return result

# ====================== MEDIA CHECK & DOWNLOAD ======================
async def check_page_has_media(url: str) -> bool:
    async with httpx.AsyncClient(**CLIENT_KWARGS) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Look for any tag with src (img, video, source)
            media_tags = soup.find_all(["img", "video", "source"])
            return len(media_tags) > 0
        except Exception as e:
            print(f"  Error checking {url}: {e}")
            return False

async def download_media_from_page(url: str, root_dir: Path,
                                   max_video_size_mb: int = MAX_VIDEO_SIZE_MB,
                                   max_image_size_mb: Optional[int] = MAX_IMAGE_SIZE_MB) -> bool:
    async with httpx.AsyncClient(**CLIENT_KWARGS) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Failed to load {url}: {e}")
            return False

        soup = BeautifulSoup(resp.text, "html.parser")
        # Collect all tags with src
        media_tags = soup.find_all(["img", "video", "source"])
        if not media_tags:
            return False

        page_name = Path(url.rstrip("/")).name
        page_name = translit_ru_to_en(page_name)
        folder = root_dir / "telegraph" / page_name
        folder.mkdir(parents=True, exist_ok=True)

        print(f"  📸 Downloading {len(media_tags)} media files from {url}")
        for tag in media_tags:
            src = tag.get("src")
            if not src:
                continue
            if src.startswith("/"):
                src = f"https://telegra.ph{src}" if "telegra.ph" in url else f"https://graph.org{src}"

            # Check extension
            ext = Path(src).suffix.lower()
            if ext not in ALLOWED_EXT:
                print(f"     ⏭️ Skipping unknown format: {src} (extension {ext})")
                continue

            # Determine type and size limit
            is_video = ext in ALLOWED_VIDEO_EXT
            max_size = max_video_size_mb if is_video else max_image_size_mb

            # Check size via HEAD request
            if max_size is not None:
                try:
                    head_resp = await client.head(src)
                    content_length = head_resp.headers.get("Content-Length")
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        if size_mb > max_size:
                            print(f"     ⏭️ {'Video' if is_video else 'Image'} too large ({size_mb:.1f} MB > {max_size} MB), skipping")
                            continue
                except Exception as e:
                    print(f"     ⚠️ Could not check size for {src}: {e}, will download anyway")

            # Download
            try:
                fname = folder / Path(src).name
                resp_media = await client.get(src)
                resp_media.raise_for_status()
                fname.write_bytes(resp_media.content)
                print(f"     ✓ {fname.name}")
            except Exception as e:
                print(f"     ✗ Error {src}: {e}")
        return True

# ====================== NSFW FILTERING ======================
def load_nsfw_model() -> Any:
    """Load the NSFW model. If the file is missing, return None."""
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model {MODEL_PATH} not found. Filtering will be skipped.")
        return None
    print("Loading NSFW model...")
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def is_girl(image_path: Path, model: Any, threshold: float = THRESHOLD) -> bool:
    if model is None:
        return False  # no model → no filtering
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        x = np.array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x, verbose=0)[0]
        porn_sexy_sum = preds[3] + preds[4]
        return porn_sexy_sum > threshold
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return False

def run_filter_on_directory(root_dir: Path, filter_not_girls_dir: Path, model: Any):
    if model is None:
        print("⚠️ NSFW filtering skipped (model not loaded).")
        return

    print("\n" + "="*60)
    print(" STARTING NSFW FILTERING")
    print("="*60)

    filter_not_girls_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    exclude_dirs = {filter_not_girls_dir.resolve()}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_path = Path(dirpath).resolve()
        if current_path in exclude_dirs:
            continue
        dirnames[:] = [d for d in dirnames if (current_path / d) not in exclude_dirs]

        for f in filenames:
            if f.lower().endswith(tuple(ALLOWED_IMAGE_EXT)):
                all_images.append(Path(dirpath) / f)

    print(f"Found {len(all_images)} images to filter...")

    girls_count = 0
    not_girls_count = 0

    for img_path in all_images:
        if is_girl(img_path, model):
            print(f"[GIRL] {img_path}")
            girls_count += 1
        else:
            new_name = f"{img_path.parent.name}_{img_path.name}"
            dst_path = filter_not_girls_dir / new_name
            counter = 1
            while dst_path.exists():
                base, ext = os.path.splitext(new_name)
                dst_path = filter_not_girls_dir / f"{base}_{counter}{ext}"
                counter += 1
            shutil.move(str(img_path), str(dst_path))
            print(f"[NOT GIRL] {img_path} -> {dst_path}")
            not_girls_count += 1

    print(f"\n📊 Filtering completed:")
    print(f"   Girls (kept in place): {girls_count}")
    print(f"   Not girls (moved to {filter_not_girls_dir}): {not_girls_count}")

# ====================== POST-PROCESSING ======================
def cleanup_empty_and_low_quality_folders(root_dir: Path, mixed_folder_name: str = "_mixed_low_quality"):
    print("\n" + "="*60)
    print(" STARTING POST-PROCESSING")
    print("="*60)

    if not root_dir.exists():
        print(f"Folder {root_dir} does not exist")
        return

    mixed_dir = root_dir / mixed_folder_name
    mixed_dir.mkdir(parents=True, exist_ok=True)

    telegraph_dir = root_dir / "telegraph"
    if telegraph_dir.exists():
        subfolders = [f for f in telegraph_dir.iterdir() if f.is_dir()]
    else:
        subfolders = [f for f in root_dir.iterdir() if f.is_dir() and f.name != mixed_folder_name]

    deleted_empty = 0
    moved_low_quality_files = 0
    deleted_low_quality_folders = 0

    for folder in subfolders:
        # Count all media files (images + videos)
        media_extensions = ALLOWED_EXT
        media_files = [f for f in folder.rglob('*')
                       if f.is_file() and f.suffix.lower() in media_extensions]
        media_count = len(media_files)

        if media_count == 0:
            try:
                folder.rmdir()
                print(f"🗑️ Removed empty folder: {folder}")
                deleted_empty += 1
            except OSError:
                print(f"⚠️ Could not remove {folder} (might not be empty)")
        elif media_count < 3:
            for media_file in media_files:
                new_name = f"{folder.name}_{media_file.name}"
                dst_path = mixed_dir / new_name
                counter = 1
                while dst_path.exists():
                    base, ext = os.path.splitext(new_name)
                    dst_path = mixed_dir / f"{base}_{counter}{ext}"
                    counter += 1
                shutil.move(str(media_file), str(dst_path))
                print(f"📦 Moved {media_file.name} -> {dst_path}")
                moved_low_quality_files += 1

            try:
                folder.rmdir()
                print(f"🗑️ Removed low-content folder: {folder}")
                deleted_low_quality_folders += 1
            except OSError:
                shutil.rmtree(folder)
                print(f"🗑️ Removed (recursively) low-content folder: {folder}")
                deleted_low_quality_folders += 1

    print(f"\n📊 Post-processing finished:")
    print(f"   Empty folders deleted: {deleted_empty}")
    print(f"   Files moved from folders with <3 media: {moved_low_quality_files}")
    print(f"   Low-content folders deleted: {deleted_low_quality_folders}")

# ====================== SINGLE LINK PROCESSING ======================
async def process_single_link(
    real_url: str,
    check_only: bool,
    download_dir: Optional[Path],
    force_recheck: bool = False,
    max_video_size_mb: int = MAX_VIDEO_SIZE_MB,
    max_image_size_mb: Optional[int] = MAX_IMAGE_SIZE_MB
) -> Optional[str]:
    if not force_recheck:
        cached = is_checked(real_url)
        if cached is not None:
            if not cached:
                print(f"⏭️ Skipped (no media, already checked): {real_url}")
                return None
            if check_only:
                return real_url

    has_media = await check_page_has_media(real_url)
    mark_checked(real_url, has_media)

    if not has_media:
        print(f"⛔ No media: {real_url}")
        return None

    print(f"✅ Has media: {real_url}")

    if not check_only and download_dir:
        await download_media_from_page(real_url, download_dir,
                                       max_video_size_mb, max_image_size_mb)

    return real_url

# ====================== MAIN ======================
async def main():
    parser = argparse.ArgumentParser(
        description="Parse media (images and videos) from Telegra.ph with NSFW filtering and cleanup"
    )
    parser.add_argument("--check-only", "-c", action="store_true", help="Only check, do not download")
    parser.add_argument("--download", "-d", metavar="DIR", default="./downloads", help="Download directory (default ./downloads)")
    parser.add_argument("--input", "-i", nargs="+", help="File, telegcrack page URL, or direct link")
    parser.add_argument("--word", "-w", help="Keyword to generate links by date iteration")
    parser.add_argument("--output", "-o", metavar="FILE", help="Save good links list to a file")
    parser.add_argument("--force", "-f", action="store_true", help="Ignore DB cache, recheck everything")
    parser.add_argument("--threshold", "-t", type=float, default=0.45, help="NSFW filter threshold (0-1), default 0.45")
    parser.add_argument("--skip-filter", action="store_true", help="Skip NSFW filtering after download")
    parser.add_argument("--final_clean", action="store_true", help="Run post‑processing (empty and <3 media folders)")
    parser.add_argument("--max-video-size", type=int, default=50, help="Maximum video size in MB (default 50)")
    parser.add_argument("--max-image-size", type=int, default=None, help="Maximum image size in MB (default no limit)")
    args = parser.parse_args()

    if not args.word and not args.input:
        parser.error("Either --word or --input must be provided")

    global THRESHOLD
    THRESHOLD = args.threshold

    filter_not_girls_dir = Path(FILTER_NOT_GIRLS_DIR)
    filter_not_girls_dir.mkdir(parents=True, exist_ok=True)

    init_db()
    download_dir = Path(args.download) if args.download else None
    if download_dir:
        download_dir.mkdir(parents=True, exist_ok=True)

    # Get list of links
    if args.word:
        word_latin = translit_ru_to_en(args.word)
        print(f"🔍 Generating links for keyword '{args.word}' -> '{word_latin}'...")
        all_real_links = await generate_links(word_latin)
        print(f"✅ Found {len(all_real_links)} valid pages.")
        (download_dir.parent / "valid_urls.txt").write_text("\n".join(all_real_links), encoding="utf-8")
    else:
        all_real_links = await expand_input_items(args.input)

    if not all_real_links:
        print("No valid Telegra.ph links found.")
        return

    unique_links = list(dict.fromkeys(all_real_links))
    print(f"\n🔍 Got {len(unique_links)} unique articles to check.\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async def limited_process(link):
        async with semaphore:
            return await process_single_link(
                link,
                args.check_only,
                download_dir,
                args.force,
                max_video_size_mb=args.max_video_size,
                max_image_size_mb=args.max_image_size
            )

    tasks = [limited_process(link) for link in unique_links]
    results = await asyncio.gather(*tasks)
    good_links = [res for res in results if res is not None]

    print(f"\n📊 Parsing summary: {len(good_links)} out of {len(unique_links)} pages contain media.")
    if good_links and args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("\n".join(good_links))
        print(f"💾 List saved to {args.output}")
    elif good_links and not args.output:
        print("\nLinks with media:")
        for link in good_links:
            print(f"  {link}")

    # ========== RUN FILTERING ==========
    if not args.check_only and download_dir and not args.skip_filter:
        model = load_nsfw_model()
        run_filter_on_directory(download_dir, filter_not_girls_dir, model)

    # ========== POST-PROCESSING ==========
    if args.final_clean:
        cleanup_empty_and_low_quality_folders(download_dir)

if __name__ == "__main__":
    asyncio.run(main())