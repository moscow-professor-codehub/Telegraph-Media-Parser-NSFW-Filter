

#  Telegraph Media Harvester

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Telegram](https://img.shields.io/badge/Telegram-@adapter_professor-26A5E4?logo=telegram&logoColor=white)](https://t.me/adapter_professor)

**Telegraph Media Harvester** is an asynchronous Python tool for discovering, downloading, and filtering media files (images and videos) from Telegra.ph pages.

It supports keyword-based link generation (with automatic transliteration from Russian), parsing telegcrack.com search pages, batch processing, size limitations, and **NSFW filtering** using a MobileNetV2 model.

---

##  Features

- 🔍 **Link generation** by keyword via date iteration (supports Russian words with automatic transliteration to English).
- 📄 **Source parsing**: HTML files, direct Telegra.ph links, and telegcrack.com search result pages.
- 🚀 **Asynchronous downloading** with format validation (`.jpg`, `.png`, `.mp4`, etc.) and configurable size limits for videos/images.
- 🚫 **NSFW filtering** using a pre-trained MobileNetV2 model (with an adjustable confidence threshold).
- 🧹 **Post-processing**: automatically removes empty folders and folders containing fewer than 3 media files (low-content folders).
- 💾 **SQLite caching** – skips previously checked links that had no media.



##  Installation

### 1. Clone the repository
```bash
git clone https://github.com/moscow-professor-codehub/telegraph-media-harvester.git
cd telegraph-media-harvester
```

### 2. Install dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Note**: `tensorflow` is quite a heavy library. If you don't have a dedicated GPU, install the CPU version. Here is the `requirements.txt` content:
```
httpx>=0.24.0
beautifulsoup4>=4.12.0
numpy>=1.24.0
Pillow>=10.0.0
tensorflow>=2.13.0
```

### 3. Download the NSFW model
The script relies on the **`nsfw_mobilenet2.224x224.h5`** model. This model file is **not included** in the repository due to its large size.
Place the model file in the project root folder (next to `main.py`).

> 💡 *If the model file is missing, the script will simply skip the filtering step and continue downloading.*

---

##  Usage

### Basic examples

**1. Check a page (without downloading)**
```bash
python main.py --word "girl" --check-only
```

**2. Download media from a file containing a list of links**
```bash
python main.py --input links.txt --download ./my_downloads
```

**3. Full cycle (download + NSFW filtering + cleanup)**
```bash
python main.py --word "cute" --download ./downloads --final_clean --max-video-size 100
```

---

##  Command-line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-c`, `--check-only` | Flag | `False` | Only check for media, **do not download** files. |
| `-d`, `--download` | Path | `./downloads` | Directory where media files will be saved. |
| `-i`, `--input` | List | `None` | An `.html` file, a telegcrack link, or a direct Telegra.ph link. |
| `-w`, `--word` | String | `None` | Keyword to generate links via date iteration. |
| `-o`, `--output` | File | `None` | Save the list of valid links (with media) to a text file. |
| `-f`, `--force` | Flag | `False` | Ignore the SQLite cache and re-check all links. |
| `-t`, `--threshold` | Float | `0.45` | NSFW filtering sensitivity threshold (from 0.0 to 1.0). |
| `--skip-filter` | Flag | `False` | Skip the NSFW filtering stage after downloading. |
| `--final_clean` | Flag | `False` | Run post-processing: delete empty folders and folders with < 3 files. |
| `--max-video-size` | Integer | `50` | Maximum video file size in MB. |
| `--max-image-size` | Integer | `None` | Maximum image file size in MB (no limit by default). |

---

##  Download Structure
Files are organized by page. Example:
```
downloads/
└── telegraph/
    └── [page-slug]/
        ├── image1.jpg
        ├── image2.png
        └── video.mp4
```
If a folder contains fewer than 3 files, when `--final_clean` is used, the media is moved to `downloads/_mixed_low_quality`, and the original folder is removed.

---

##  License and Attribution

This project is distributed under the **MIT License**.

**Important usage condition**: When copying, modifying, or distributing this code, you **must** retain the original copyright notice and explicitly mention the original author (Your Name) in your project or documentation.

For more details, see the [`LICENSE`](LICENSE)  file.

---

##  Important Notes

- **Transliteration**: The function translates Russian words into English phonetically (e.g., *девушка* → *devushka*). This simplifies searching for pages on Telegra.ph.
- **Database**: The `processed_links.db` file is automatically created in the project folder. If you delete it, the script will start checking all links from scratch (even without the `--force` flag).
- **telegcrack.com handling**: The script correctly processes redirect links (`/out.php?to=`), extracting the real Telegra.ph URLs from them.
- **Performance**: By default, the number of concurrent requests is limited to `5` (the `MAX_CONCURRENT` parameter inside the code). You can adjust this value for faster processing, but be careful to avoid being blocked by the servers.
- **Model advice**: If you have limited RAM, loading the NSFW model might take a few seconds. Please be patient on the first run.

---

##  Connect with the author

For updates, support, or feedback, join our Telegram channel: [t.me/adapter_professor](https://t.me/adapter_professor)

---

If you encounter any issues or have questions, please open an **Issue** in this repository. Happy scraping! 🚀
