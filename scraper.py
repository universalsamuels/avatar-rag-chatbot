"""
scraper.py

Fetches Avatar Wiki articles using the MediaWiki API (revisions/wikitext),
strips wiki markup, and saves clean .txt files to data/ folder.

Run with: python scraper.py
"""

import os
import re
import time
import requests

# ---------- CONFIG ----------
DATA_DIR = "data"
API_URL = "https://avatar.fandom.com/api.php"
DELAY = 1.0

# ---------- ARTICLES TO FETCH ----------
ARTICLES = [
    # Main Characters ATLA
    "Aang", "Katara", "Sokka", "Toph Beifong", "Zuko", "Azula",
    "Iroh", "Ozai", "Mai", "Ty Lee", "Suki",

    # Main Characters Korra
    "Korra", "Mako", "Bolin", "Asami Sato", "Tenzin",
    "Lin Beifong", "Suyin Beifong", "Varrick", "Zhu Li", "Jinora",

    # Korra Antagonists
    "Amon", "Unalaq", "Zaheer", "Kuvira", "Tarrlok", "Red Lotus",

    # Avatar Lore
    "Wan", "Avatar cycle", "Avatar State", "Lion turtle",
    "Raava", "Vaatu", "Harmonic Convergence",
    "Hundred Year War", "Genocide of the Air Nomads",

    # Bending
    "Airbending", "Waterbending", "Earthbending", "Firebending",
    "Metalbending", "Lavabending", "Bloodbending",
    "Lightning generation and redirection", "Energybending",
    "Combustion bending", "Seismic sense",

    # Nations & Places
    "Air Nomads", "Water Tribe", "Earth Kingdom", "Fire Nation",
    "Republic City", "Ba Sing Se", "Northern Water Tribe",
    "Southern Water Tribe", "Zaofu", "Omashu",
    "Spirit World", "Southern Air Temple", "Northern Air Temple",

    # Animals
    "Appa", "Momo", "Naga", "Pabu", "Flying bison",
    "Dragon", "Badgermole",

    # Organizations
    "Order of the White Lotus", "Dai Li", "Kyoshi Warriors",
    "Equalists", "Earth Empire", "United Forces",

    # Past Avatars
    "Kyoshi", "Roku", "Kuruk", "Yangchen",

    # Supporting ATLA
    "Bumi (king)", "Pakku", "Jeong Jeong", "Piandao", "Ursa",
    "Hakoda", "Jet", "Hama", "Combustion Man", "June",
    "Cabbage merchant",

    # Supporting Korra
    "Hiroshi Sato", "Pema", "Meelo", "Ikki",
    "Bumi (Air Nomad)", "Kya (daughter of Aang)",
    "Tonraq", "Desna and Eska", "Opal", "Baatar Jr.",

    # Spirits
    "Wan Shi Tong", "Koh", "Yue", "La and Tui",

    # Production
    "Avatar: The Last Airbender (series)",
    "The Legend of Korra",
    "Michael DiMartino and Bryan Konietzko",
]


def fetch_wikitext(title):
    """Fetch raw wikitext for an article using the revisions API."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "redirects": True,
        "format": "json",
    }
    headers = {"User-Agent": "AvatarMind-RAG-Bot/1.0 (educational project)"}
    try:
        r = requests.get(API_URL, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                return None
            revisions = page.get("revisions", [])
            if revisions:
                slots = revisions[0].get("slots", {})
                if slots:
                    return slots.get("main", {}).get("*", "")
                # fallback for older API format
                return revisions[0].get("*", "")
        return None
    except requests.RequestException as e:
        print(f"\n    ERROR: {e}")
        return None


def clean_wikitext(text):
    """Strip wiki markup and return readable plain text."""
    # Remove templates like {{...}}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # Remove file/image links
    text = re.sub(r'\[\[(?:File|Image):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    # Convert [[link|display]] to just display text
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Remove remaining [[ ]] links
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    # Remove external links [url text] -> text
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    # Remove bold/italic markers
    text = re.sub(r"'{2,3}", '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Convert section headers == Header == to plain text
    text = re.sub(r'={2,4}\s*(.+?)\s*={2,4}', r'\n\1\n', text)
    # Remove table markup
    text = re.sub(r'^\s*[|!{].*$', '', text, flags=re.MULTILINE)
    # Remove category/reference lines
    text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text)
    text = re.sub(r'<ref[^/]*/>', '', text)
    text = re.sub(r'<ref.*?</ref>', '', text, flags=re.DOTALL)
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def make_filename(title):
    return title.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(":", "") + ".txt"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Fetching {len(ARTICLES)} articles from Avatar Wiki...\n")

    success = 0
    failed = []

    for title in ARTICLES:
        filename = make_filename(title)
        filepath = os.path.join(DATA_DIR, filename)

        if os.path.exists(filepath):
            print(f"  SKIP (exists): {filename}")
            success += 1
            continue

        print(f"  Fetching: {title}...", end=" ", flush=True)
        wikitext = fetch_wikitext(title)

        if wikitext and len(wikitext) > 200:
            cleaned = clean_wikitext(wikitext)
            if len(cleaned) > 150:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"{title}\n\n{cleaned}")
                print(f"✓ ({len(cleaned):,} chars)")
                success += 1
            else:
                print("✗ (too short after cleaning)")
                failed.append(title)
        else:
            print("✗ (not found or empty)")
            failed.append(title)

        time.sleep(DELAY)

    print(f"\n{'='*50}")
    print(f"Done! {success}/{len(ARTICLES)} articles saved.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    print(f"\nNext step: run  python ingest.py  to embed everything.")


if __name__ == "__main__":
    main()