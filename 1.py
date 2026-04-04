"""
Nepal Vehicle Classification - Image Downloader v2
===================================================
Downloads 1000 images per vehicle class using:
  1. duckduckgo-search library (primary)
  2. Bing image scraper (fallback)

Usage:
    pip install ddgs requests pillow tqdm
    python download_images.py
    python download_images.py --classes bus truck hiace
    python download_images.py --target 500           # faster test run
    python download_images.py --skip-download        # just re-split
"""

import os
import time
import shutil
import random
import hashlib
import argparse
import requests
import re
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

# ── pip guard (updated for new package name) ──
try:
    from ddgs import DDGS
except ImportError:
    import subprocess
    import sys
    print("📦 Installing ddgs (DuckDuckGo Search)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ddgs", "-q"])
    from ddgs import DDGS

# ─────────────────────────────────────────────
# Vehicle Classes — Nepal specific
# Multiple queries per class = more diversity
# ─────────────────────────────────────────────
VEHICLE_CLASSES = {
    # ── Cars ──────────────────────────────────
    "car_suzuki": [
        "Suzuki Swift car Nepal road",
        "Suzuki Dzire sedan Nepal street",
        "Suzuki Alto Nepal traffic",
        "Suzuki Ciaz Nepal",
        "Suzuki Brezza Nepal",
        "Suzuki car Kathmandu",
    ],
    "car_hyundai": [
        "Hyundai i20 Nepal road",
        "Hyundai Creta Nepal street",
        "Hyundai Verna Nepal traffic",
        "Hyundai Tucson Nepal",
        "Hyundai Grand i10 Nepal",
        "Hyundai car Kathmandu",
    ],
    "car_toyota": [
        "Toyota Corolla Nepal road",
        "Toyota Yaris Nepal street",
        "Toyota Fortuner Nepal traffic",
        "Toyota Hilux Nepal",
        "Toyota Land Cruiser Nepal",
        "Toyota car Kathmandu",
    ],
    "car_honda": [
        "Honda City sedan Nepal road",
        "Honda WRV Nepal street",
        "Honda Jazz Nepal traffic",
        "Honda HR-V Nepal",
        "Honda Amaze Nepal",
        "Honda car Kathmandu",
    ],
    "car_kia": [
        "Kia Seltos Nepal road",
        "Kia Sonet Nepal street",
        "Kia Stonic Nepal traffic",
        "Kia Carnival Nepal",
        "Kia car Kathmandu",
        "Kia EV6 Nepal",
    ],
    "car_tata": [
        "Tata Nexon Nepal road",
        "Tata Altroz Nepal street",
        "Tata Tigor Nepal traffic",
        "Tata Harrier Nepal",
        "Tata Punch Nepal",
        "Tata car Kathmandu",
    ],
    "car_volkswagen": [
        "Volkswagen Polo Nepal road",
        "Volkswagen Vento Nepal street",
        "VW Taigun Nepal traffic",
        "Volkswagen Tiguan Nepal",
        "VW car Kathmandu",
        "Volkswagen car Nepal",
    ],
    "car_ford": [
        "Ford Figo Nepal road",
        "Ford EcoSport Nepal street",
        "Ford Aspire Nepal traffic",
        "Ford Endeavour Nepal",
        "Ford car Kathmandu",
        "Ford Nepal vehicle",
    ],
    "car_nissan": [
        "Nissan Sunny Nepal road",
        "Nissan Kicks Nepal street",
        "Nissan Magnite Nepal traffic",
        "Nissan Terrano Nepal",
        "Nissan car Kathmandu",
        "Nissan X-Trail Nepal",
    ],
    "car_mg": [
        "MG Hector Nepal road",
        "MG ZS EV Nepal street",
        "MG Astor Nepal traffic",
        "MG Gloster Nepal",
        "MG car Kathmandu",
        "MG Nepal vehicle",
    ],
    
    # ── Bikes ─────────────────────────────────
    "bike_bajaj": [
        "Bajaj Pulsar 150 Nepal road",
        "Bajaj Dominar Nepal street",
        "Bajaj Avenger Nepal traffic",
        "Bajaj NS200 Nepal",
        "Bajaj CT100 Nepal",
        "Bajaj motorcycle Kathmandu",
    ],
    "bike_hero": [
        "Hero Splendor Nepal road",
        "Hero HF Deluxe Nepal street",
        "Hero Xtreme 160R Nepal traffic",
        "Hero Glamour Nepal",
        "Hero Passion Nepal",
        "Hero motorcycle Kathmandu",
    ],
    "bike_honda": [
        "Honda CB Shine Nepal road",
        "Honda CB Hornet Nepal street",
        "Honda CB150R Nepal traffic",
        "Honda XBlade Nepal",
        "Honda Unicorn Nepal",
        "Honda motorcycle Kathmandu",
    ],
    "bike_tvs": [
        "TVS Apache 160 Nepal road",
        "TVS Raider 125 Nepal street",
        "TVS Star City Nepal traffic",
        "TVS Apache 200 Nepal",
        "TVS Sport Nepal",
        "TVS motorcycle Kathmandu",
    ],
    "bike_yamaha": [
        "Yamaha FZS Nepal road",
        "Yamaha MT-15 Nepal street",
        "Yamaha R15 Nepal traffic",
        "Yamaha FZ25 Nepal",
        "Yamaha Saluto Nepal",
        "Yamaha motorcycle Kathmandu",
    ],
    "bike_royal_enfield": [
        "Royal Enfield Meteor 350 Nepal road",
        "Royal Enfield Classic 350 Nepal street",
        "Royal Enfield Himalayan Nepal traffic",
        "Royal Enfield Thunderbird Nepal",
        "Royal Enfield Bullet Nepal",
        "Royal Enfield Kathmandu",
    ],
    "bike_ktm": [
        "KTM Duke 200 Nepal road",
        "KTM RC 200 Nepal street",
        "KTM 390 Duke Nepal traffic",
        "KTM Duke 125 Nepal",
        "KTM Adventure Nepal",
        "KTM motorcycle Kathmandu",
    ],
    "bike_suzuki": [
        "Suzuki Gixxer Nepal road",
        "Suzuki Intruder Nepal street",
        "Suzuki Hayate Nepal traffic",
        "Suzuki GSX Nepal",
        "Suzuki Access motorcycle Nepal",
        "Suzuki bike Kathmandu",
    ],
    "bike_jawa": [
        "Jawa 42 Nepal road",
        "Jawa Perak Nepal street",
        "Jawa Yezdi Nepal traffic",
        "Jawa motorcycle Nepal",
        "Jawa bike Kathmandu",
        "Jawa classic motorcycle Nepal",
    ],
    "bike_mahindra": [
        "Mahindra Mojo Nepal road",
        "Mahindra Centuro Nepal street",
        "Mahindra motorcycle Nepal traffic",
        "Mahindra bike Nepal",
        "Mahindra two wheeler Nepal",
        "Mahindra bike Kathmandu",
    ],
    
    # ── Scooters ──────────────────────────────
    "scooter_honda_activa": [
        "Honda Activa scooter Nepal road",
        "Honda Activa 6G Nepal street",
        "Honda Activa 125 Nepal traffic",
        "Honda Activa Nepal Kathmandu",
        "Honda Activa scooter side view",
        "Honda Activa Nepal riding",
    ],
    "scooter_tvs_jupiter": [
        "TVS Jupiter scooter Nepal road",
        "TVS Jupiter ZX Nepal street",
        "TVS Jupiter 125 Nepal traffic",
        "TVS Jupiter scooter Kathmandu",
        "TVS Jupiter side view Nepal",
        "TVS Jupiter scooter Nepal",
    ],
    "scooter_yamaha_ray": [
        "Yamaha Ray ZR scooter Nepal road",
        "Yamaha Fascino Nepal street",
        "Yamaha Ray scooter Nepal traffic",
        "Yamaha scooter Kathmandu",
        "Yamaha Ray ZR 125 Nepal",
        "Yamaha scooter Nepal side view",
    ],
    "scooter_aprilia": [
        "Aprilia SR 160 scooter Nepal road",
        "Aprilia Storm 125 Nepal street",
        "Aprilia scooter Nepal traffic",
        "Aprilia SR scooter Kathmandu",
        "Aprilia scooter Nepal side view",
        "Aprilia scooter Nepal",
    ],
    "scooter_vespa": [
        "Vespa SXL scooter Nepal road",
        "Vespa VXL Nepal street",
        "Vespa scooter Nepal traffic",
        "Vespa scooter Kathmandu",
        "Vespa 125 Nepal",
        "Vespa scooter Nepal side view",
    ],
    "scooter_suzuki_access": [
        "Suzuki Access 125 scooter Nepal road",
        "Suzuki Burgman Street Nepal",
        "Suzuki scooter Nepal traffic",
        "Suzuki Access scooter Kathmandu",
        "Suzuki scooter Nepal side view",
        "Suzuki scooter Nepal",
    ],
    "scooter_hero_destini": [
        "Hero Destini 125 scooter Nepal road",
        "Hero Maestro Edge Nepal street",
        "Hero Pleasure scooter Nepal traffic",
        "Hero scooter Kathmandu",
        "Hero Destini scooter Nepal side view",
        "Hero scooter Nepal",
    ],
    "scooter_honda_dio": [
        "Honda Dio scooter Nepal road",
        "Honda Dio 110 Nepal street",
        "Honda Dio scooter Nepal traffic",
        "Honda Dio scooter Kathmandu",
        "Honda Dio Nepal side view",
        "Honda Dio scooter Nepal",
    ],
    "scooter_tvs_ntorq": [
        "TVS Ntorq 125 scooter Nepal road",
        "TVS Ntorq Race Edition Nepal street",
        "TVS Ntorq scooter Nepal traffic",
        "TVS Ntorq scooter Kathmandu",
        "TVS Ntorq Nepal side view",
        "TVS Ntorq scooter Nepal",
    ],
    "scooter_yadea": [
        "Yadea electric scooter Nepal road",
        "Yadea G5 scooter Nepal street",
        "Yadea EV scooter Nepal traffic",
        "Yadea scooter Kathmandu",
        "Yadea electric Nepal side view",
        "Yadea scooter Nepal",
    ],
    
    # ── Heavy / Public ─────────────────────────
    "bus": [
        "public bus Nepal road Kathmandu",
        "Nepal local bus street traffic",
        "intercity bus Nepal highway",
        "Sajha Yatayat bus Nepal",
        "Nepal bus side view road",
        "Nepal bus front view road",
        "Nepal bus Pokhara highway",
        "Nepal public transport bus",
        "Nepal bus rear view road",
        "Nepal tourist bus highway",
    ],
    "truck": [
        "truck Nepal highway road",
        "Nepal cargo truck street",
        "heavy truck Nepal Kathmandu traffic",
        "Nepal truck side view road",
        "Nepal truck front view highway",
        "Nepal tipper truck road",
        "Nepal tanker truck highway",
        "Nepal delivery truck road",
        "Nepal truck rear view road",
        "Nepal goods truck highway",
    ],
    "hiace": [
        "Toyota HiAce microbus Nepal road",
        "HiAce van Nepal street traffic",
        "Nepal microbus Kathmandu road",
        "Toyota HiAce Nepal side view",
        "Nepal microbus front view road",
        "HiAce Nepal public transport",
        "Nepal tempo microbus road",
        "Toyota HiAce Nepal highway",
        "Nepal minibus road traffic",
        "HiAce van Nepal Pokhara road",
    ],
}

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

TARGET_PER_CLASS  = 1000
IMAGES_PER_QUERY  = 100      # DDG returns max ~100 per query
TRAIN_SPLIT       = 0.80
DOWNLOAD_WORKERS  = 8
IMG_TIMEOUT       = 12
MIN_IMG_SIZE      = (80, 80)

RAW_DIR   = Path("data/raw")
TRAIN_DIR = Path("data/train")
VAL_DIR   = Path("data/val")

# NOTE: default paths can be overridden via command-line flags below.
# Use --raw-dir=dataset/raw etc if your dataset directories already exist.

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ─────────────────────────────────────────────
# URL Collection
# ─────────────────────────────────────────────

def get_urls_ddg(query: str, max_results: int = 100) -> list:
    """Fetch image URLs using DuckDuckGo."""
    urls = []
    try:
        with DDGS() as ddgs:
            # Note: The new ddgs package uses a different API
            results = list(ddgs.images(
                query,
                max_results=max_results,
            ))
            urls = [r["image"] for r in results if "image" in r]
    except Exception as e:
        # Silently fail - we'll try fallback
        pass
    return urls


def get_urls_bing(query: str, max_results: int = 50) -> list:
    """Bing image scraper fallback."""
    urls = []
    try:
        encoded = requests.utils.quote(query)
        # Use mobile version for simpler HTML
        url = f"https://www.bing.com/images/search?q={encoded}&count={max_results}&form=HDRSC2&first=0"
        r = requests.get(url, headers=HEADERS, timeout=10)
        
        # Try multiple patterns to extract image URLs
        patterns = [
            r'"murl":"(https?://[^"]+)"',
            r'mediaurl":"(https?://[^"]+)"',
            r'src="(https?://[^"]+\.(?:jpg|jpeg|png|webp|bmp))"',
        ]
        
        for pattern in patterns:
            found = re.findall(pattern, r.text)
            urls.extend(found)
            
        # Remove duplicates while preserving order
        seen = set()
        urls = [u for u in urls if not (u in seen or seen.add(u))]
        
    except Exception as e:
        print(f"      ⚠ Bing error for '{query}': {e}")
    
    return urls[:max_results]


def collect_urls(class_name: str, queries: list, target: int) -> list:
    """Collect unique image URLs from multiple queries."""
    all_urls = []
    seen = set()
    
    # Round 1: all provided queries
    for q in queries:
        if len(all_urls) >= target + 200:
            break
            
        # Try DDG first, then Bing
        urls = get_urls_ddg(q, IMAGES_PER_QUERY)
        if not urls or len(urls) < 20:  # If DDG returned too few, try Bing
            bing_urls = get_urls_bing(q, IMAGES_PER_QUERY)
            if bing_urls:
                urls = bing_urls
        
        # Add new URLs
        new_urls = [u for u in urls if u not in seen]
        seen.update(new_urls)
        all_urls.extend(new_urls)
        
        # Add delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 1.0))
    
    # Round 2: augmented queries if still short
    if len(all_urls) < target:
        extra_views = ["side view", "front view", "rear view", "on road", "parking", "driving", "close up"]
        base_query = queries[0].split()[0]  # Get first word as base
        
        for kw in extra_views:
            if len(all_urls) >= target + 200:
                break
                
            q = f"{base_query} {kw} Nepal"
            urls = get_urls_ddg(q, IMAGES_PER_QUERY)
            if not urls or len(urls) < 10:
                urls = get_urls_bing(q, IMAGES_PER_QUERY)
            
            new_urls = [u for u in urls if u not in seen]
            seen.update(new_urls)
            all_urls.extend(new_urls)
            time.sleep(random.uniform(0.5, 1.0))
    
    # Remove any obviously invalid URLs
    valid_urls = []
    for url in all_urls:
        # Skip URLs with common issues
        if any(bad in url.lower() for bad in ['data:image', 'base64', 'placeholder', 'blank']):
            continue
        if not url.startswith(('http://', 'https://')):
            continue
        valid_urls.append(url)
    
    return valid_urls[:target + 300]  # buffer above target


# ─────────────────────────────────────────────
# Image Download
# ─────────────────────────────────────────────

def is_valid_image(data: bytes) -> bool:
    """Check if image data is valid and meets size requirements."""
    try:
        img = Image.open(BytesIO(data))
        w, h = img.size
        return w >= MIN_IMG_SIZE[0] and h >= MIN_IMG_SIZE[1]
    except Exception:
        return False


def download_one(url: str, dest_dir: Path) -> bool:
    """Download a single image with validation."""
    try:
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(0.1, 0.3))
        
        r = requests.get(url, headers=HEADERS, timeout=IMG_TIMEOUT, stream=True)
        if r.status_code != 200:
            return False
            
        # Check content type
        ct = r.headers.get("content-type", "").lower()
        if "image" not in ct and not any(url.lower().endswith(ext) for ext in VALID_EXTS):
            return False
            
        # Get image data
        data = r.content
        if len(data) < 2048:  # Minimum 2KB
            return False
            
        # Validate image
        if not is_valid_image(data):
            return False
            
        # Determine extension
        ext = ".jpg"
        if "png" in ct or url.lower().endswith(".png"):
            ext = ".png"
        elif "webp" in ct or url.lower().endswith(".webp"):
            ext = ".webp"
        elif "jpeg" in ct or url.lower().endswith(".jpeg"):
            ext = ".jpeg"
            
        # Save image
        h = hashlib.md5(data).hexdigest()[:16]
        fname = dest_dir / f"{h}{ext}"
        
        # Avoid duplicate saves
        if not fname.exists():
            fname.write_bytes(data)
            return True
            
    except Exception:
        pass
    return False


def download_class(class_name: str, queries: list, raw_dir: Path, target: int = 1000) -> int:
    """Download images for a single class."""
    dest = raw_dir / class_name
    dest.mkdir(parents=True, exist_ok=True)
    
    # Count existing valid images
    existing = len([f for f in dest.iterdir() if f.suffix.lower() in VALID_EXTS])
    
    if existing >= target:
        print(f"  ✅ {class_name}: already has {existing} images")
        return existing
    
    remaining = target - existing
    print(f"  🔍 {class_name}: collecting URLs (need {remaining} more, have {existing})...")
    
    # Collect URLs
    urls = collect_urls(class_name, queries, remaining + 200)
    print(f"      → {len(urls)} unique URLs found, downloading...")
    
    if not urls:
        print(f"      ⚠ No URLs found for {class_name}")
        return existing
    
    # Download images
    downloaded = existing
    with tqdm(total=min(len(urls), remaining + 100), desc=f"      ↓ {class_name}", unit="img", leave=False) as pbar:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
            # Submit download tasks
            futures = {ex.submit(download_one, url, dest): url for url in urls}
            
            # Process completed downloads
            for future in as_completed(futures):
                pbar.update(1)
                if future.result():
                    downloaded += 1
                if downloaded >= target:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
    
    # Final count
    final = len([f for f in dest.iterdir() if f.suffix.lower() in VALID_EXTS])
    status = "✅" if final >= target else f"⚠  only {final}/{target}"
    print(f"  {status} {class_name}: {final} images total")
    
    return final


# ─────────────────────────────────────────────
# Split Dataset
# ─────────────────────────────────────────────

def split_dataset(raw_dir: Path, train_dir: Path, val_dir: Path):
    """Split raw images into train and validation sets."""
    print("\n📂 Splitting dataset into train / validation sets...")
    
    total_train = 0
    total_val = 0
    
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
        
        if not images:
            print(f"    ⚠ {class_name}: no images found")
            continue
        
        # Shuffle and split
        random.shuffle(images)
        n_train = max(1, int(len(images) * TRAIN_SPLIT))
        train_imgs = images[:n_train]
        val_imgs = images[n_train:]
        
        # Copy to train directory
        train_out = train_dir / class_name
        train_out.mkdir(parents=True, exist_ok=True)
        for img in train_imgs:
            dst = train_out / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
        
        # Copy to validation directory
        val_out = val_dir / class_name
        val_out.mkdir(parents=True, exist_ok=True)
        for img in val_imgs:
            dst = val_out / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
        
        total_train += len(train_imgs)
        total_val += len(val_imgs)
        
        print(f"    {class_name:<35} train={len(train_imgs):4d}  val={len(val_imgs):4d}")
    
    print(f"\n  Total: train={total_train}, val={total_val}, overall={total_train+total_val}")


def print_summary(classes: dict):
    """Print dataset summary statistics."""
    print("\n📊 Dataset Summary:")
    print(f"{'Class':<35} {'Train':>6} {'Val':>6} {'Total':>7}")
    print("─" * 56)
    
    grand_train = grand_val = 0
    low_count_classes = []
    
    for cls in sorted(classes.keys()):
        t = len(list((TRAIN_DIR / cls).glob("*"))) if (TRAIN_DIR / cls).exists() else 0
        v = len(list((VAL_DIR / cls).glob("*"))) if (VAL_DIR / cls).exists() else 0
        grand_train += t
        grand_val += v
        total = t + v
        
        # Flag classes with insufficient images
        if total < 500:
            low_count_classes.append(f"{cls} ({total} images)")
            flag = "  ⚠ LOW"
        else:
            flag = ""
        
        print(f"{cls:<35} {t:>6} {v:>6} {total:>7}{flag}")
    
    print("─" * 56)
    print(f"{'TOTAL':<35} {grand_train:>6} {grand_val:>6} {grand_train+grand_val:>7}")
    
    if low_count_classes:
        print(f"\n⚠ Warning: {len(low_count_classes)} classes have fewer than 500 images:")
        for cls in low_count_classes:
            print(f"   - {cls}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    global DOWNLOAD_WORKERS, RAW_DIR, TRAIN_DIR, VAL_DIR  # Declare globals at the beginning
    
    parser = argparse.ArgumentParser(description="Nepal Vehicle Image Downloader v2")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Download only specific classes (partial name match)")
    parser.add_argument("--target", type=int, default=TARGET_PER_CLASS,
                        help=f"Target images per class (default: {TARGET_PER_CLASS})")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only re-split existing images")
    parser.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS,
                        help=f"Number of download workers (default: {DOWNLOAD_WORKERS})")
    parser.add_argument("--raw-dir", type=str, default="dataset/raw",
                        help="Raw images directory (default: dataset/raw)")
    parser.add_argument("--train-dir", type=str, default="dataset/train",
                        help="Train split directory (default: dataset/train)")
    parser.add_argument("--val-dir", type=str, default="dataset/val",
                        help="Val split directory (default: dataset/val)")
    args = parser.parse_args()
    
    # Set global worker count and directories
    DOWNLOAD_WORKERS = args.workers
    RAW_DIR = Path(args.raw_dir)
    TRAIN_DIR = Path(args.train_dir)
    VAL_DIR = Path(args.val_dir)
    
    random.seed(42)
    
    # Filter classes if specified
    classes = VEHICLE_CLASSES
    if args.classes:
        classes = {k: v for k, v in VEHICLE_CLASSES.items()
                   if any(a.lower() in k.lower() for a in args.classes)}
        if not classes:
            print(f"❌ No classes matched: {args.classes}")
            return
        print(f"🎯 Filtered to {len(classes)} classes: {list(classes.keys())}\n")
    
    # Download images if not skipped
    if not args.skip_download:
        total_target = args.target * len(classes)
        print(f"🚗 Target: {args.target:,} images × {len(classes)} classes = ~{total_target:,} images total")
        print(f"   Using {DOWNLOAD_WORKERS} download workers\n")
        
        for i, (cls, queries) in enumerate(classes.items(), 1):
            print(f"\n[{i}/{len(classes)}] {cls}")
            download_class(cls, queries, RAW_DIR, target=args.target)
        
        print("\n✅ Downloads complete.")
    
    # Split dataset
    split_dataset(RAW_DIR, TRAIN_DIR, VAL_DIR)
    
    # Print summary
    print_summary(classes)


if __name__ == "__main__":
    main()