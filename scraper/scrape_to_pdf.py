"""Standalone website-to-PDF scraper (independent of the RAG app).

Give it a URL; it scrapes the page — or, with --crawl, every internal page on
the same site — and writes everything into one clean, readable PDF.

Examples:
    python scrape_to_pdf.py https://example.com
    python scrape_to_pdf.py https://soe.cusat.ac.in/index.php --crawl -o soe.pdf
    python scrape_to_pdf.py https://site.com --crawl --max-pages 200 --images

Dependencies:  pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

# ---------------------------------------------------------------------------
# Fonts (Unicode-capable so non-Latin text / em-dashes render correctly)
# ---------------------------------------------------------------------------
_FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT_REGULAR = f"{_FONT_DIR}/DejaVuSans.ttf"
FONT_BOLD = f"{_FONT_DIR}/DejaVuSans-Bold.ttf"
FONT_ITALIC = f"{_FONT_DIR}/DejaVuSans-Oblique.ttf"
HAS_UNICODE_FONT = Path(FONT_REGULAR).exists()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

SKIP_EXTENSIONS = {
    ".pdf", ".zip", ".rar", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".css", ".js",
    ".mp3", ".mp4", ".avi", ".mov", ".woff", ".woff2", ".ttf", ".eot",
}
# URL path fragments that are auth/portal traps — no useful content, and they
# tend to redirect to slow endpoints that time out.
SKIP_PATH_HINTS = ("login", "logout", "myprofile", "signin", "signup",
                   "register", "/admin")
CONTENT_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre",
                "blockquote", "table", "img"]
STRIP_TAGS = ["script", "style", "noscript", "svg", "form", "iframe", "head"]
BOILERPLATE_TAGS = ["nav", "footer", "header", "aside"]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch(session: requests.Session, url: str, timeout: int = 12):
    try:
        resp = session.get(url, headers=HEADERS, timeout=timeout,
                           verify=session.verify)
        resp.raise_for_status()
    except requests.exceptions.SSLError:
        # Many edu/gov sites have broken cert chains. Retry without verification.
        try:
            print(f"  ~ SSL verify failed, retrying insecurely: {url}")
            resp = session.get(url, headers=HEADERS, timeout=timeout, verify=False)
            resp.raise_for_status()
            session.verify = False  # remember for the rest of the crawl
        except requests.RequestException as exc:
            print(f"  ! failed: {url} ({exc})")
            return None
    except requests.RequestException as exc:
        print(f"  ! failed: {url} ({exc})")
        return None
    ctype = resp.headers.get("Content-Type", "")
    if "html" not in ctype.lower():
        return None
    resp.encoding = resp.apparent_encoding or resp.encoding
    return resp.text


# ---------------------------------------------------------------------------
# Link discovery (for crawl mode)
# ---------------------------------------------------------------------------
def normalize(url: str) -> str:
    url, _ = urldefrag(url)            # drop #fragments
    return url.rstrip("/") or url


def same_site(a: str, b: str) -> bool:
    na, nb = urlparse(a).netloc.lower(), urlparse(b).netloc.lower()
    return na.replace("www.", "") == nb.replace("www.", "")


def extract_links(base_url: str, soup: BeautifulSoup) -> list[str]:
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        full = normalize(urljoin(base_url, href))
        path = urlparse(full).path.lower()
        if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
            continue
        if any(h in full.lower() for h in SKIP_PATH_HINTS):
            continue
        if urlparse(full).scheme not in ("http", "https"):
            continue
        links.append(full)
    return links


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------
def pick_root(soup: BeautifulSoup):
    for sel in ["main", "article", "#content", "#main", ".content",
                "#main-content", "[role=main]"]:
        el = soup.select_one(sel)
        if el and len(el.get_text(strip=True)) > 150:
            return el
    return soup.body or soup


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def extract_blocks(soup: BeautifulSoup, base_url: str) -> tuple[str, list]:
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    # Remove obvious boilerplate, but keep them if the page is basically nav-only.
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()

    title = clean_text(soup.title.get_text()) if soup.title else base_url
    root = pick_root(soup)

    blocks = []
    seen_text = set()
    for el in root.find_all(CONTENT_TAGS):
        name = el.name
        if name == "img":
            src = el.get("src") or el.get("data-src")
            if src:
                blocks.append(("img", urljoin(base_url, src)))
            continue
        if name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [clean_text(td.get_text(" "))
                         for td in tr.find_all(["td", "th"])]
                if any(cells):
                    rows.append(cells)
            if rows:
                blocks.append(("table", rows))
            continue
        text = clean_text(el.get_text(" "))
        if not text or len(text) < 2:
            continue
        # Skip list items whose text is already captured by a parent paragraph.
        key = (name, text)
        if key in seen_text:
            continue
        seen_text.add(key)
        blocks.append((name, text))
    return title, blocks


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------
class PDF(FPDF):
    def __init__(self, source_url: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.source_url = source_url
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)
        if HAS_UNICODE_FONT:
            self.add_font("DejaVu", "", FONT_REGULAR)
            self.add_font("DejaVu", "B", FONT_BOLD)
            self.add_font("DejaVu", "I", FONT_ITALIC)
            self.base_font = "DejaVu"
        else:
            self.base_font = "Helvetica"

    def footer(self):
        self.set_y(-14)
        self.set_font(self.base_font, "I", 8)
        self.set_text_color(140)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)

    def _safe(self, text: str) -> str:
        if HAS_UNICODE_FONT:
            return text
        return text.encode("latin-1", "replace").decode("latin-1")


SIZES = {"h1": 17, "h2": 14, "h3": 12.5, "h4": 11.5, "h5": 11, "h6": 11}


def render_pdf(pages: list[dict], source_url: str, out_path: Path,
               session: requests.Session, embed_images: bool) -> None:
    pdf = PDF(source_url)
    epw = pdf.w - pdf.l_margin - pdf.r_margin

    # ---- cover page ----
    pdf.add_page()
    pdf.set_font(pdf.base_font, "B", 22)
    pdf.ln(40)
    pdf.multi_cell(epw, 12, pdf._safe("Website Scrape"), align="C")
    pdf.ln(4)
    pdf.set_font(pdf.base_font, "", 12)
    pdf.set_text_color(80)
    pdf.multi_cell(epw, 8, pdf._safe(source_url), align="C")
    pdf.ln(2)
    pdf.multi_cell(epw, 8, pdf._safe(f"{len(pages)} page(s) scraped"), align="C")
    pdf.set_text_color(0)

    # ---- table of contents ----
    pdf.add_page()
    pdf.set_font(pdf.base_font, "B", 15)
    pdf.multi_cell(epw, 9, pdf._safe("Contents"))
    pdf.ln(2)
    pdf.set_font(pdf.base_font, "", 10)
    for i, page in enumerate(pages, 1):
        pdf.multi_cell(epw, 6, pdf._safe(f"{i}. {page['title']}"))

    # ---- one section per scraped page ----
    for i, page in enumerate(pages, 1):
        pdf.add_page()
        pdf.set_font(pdf.base_font, "B", 16)
        pdf.set_text_color(20, 40, 90)
        pdf.multi_cell(epw, 9, pdf._safe(f"{i}. {page['title']}"))
        pdf.set_font(pdf.base_font, "I", 8)
        pdf.set_text_color(120)
        pdf.multi_cell(epw, 5, pdf._safe(page["url"]))
        pdf.set_text_color(0)
        pdf.ln(3)

        for kind, value in page["blocks"]:
            if kind in SIZES:
                pdf.ln(2)
                pdf.set_font(pdf.base_font, "B", SIZES[kind])
                pdf.multi_cell(epw, SIZES[kind] * 0.5, pdf._safe(value))
                pdf.ln(1)
            elif kind == "p" or kind == "blockquote":
                pdf.set_font(pdf.base_font, "I" if kind == "blockquote" else "", 10.5)
                pdf.multi_cell(epw, 5.6, pdf._safe(value))
                pdf.ln(1.5)
            elif kind == "li":
                pdf.set_font(pdf.base_font, "", 10.5)
                pdf.multi_cell(epw, 5.6, pdf._safe(f"  •  {value}"))
            elif kind == "pre":
                pdf.set_font(pdf.base_font, "", 9)
                pdf.set_fill_color(244, 244, 246)
                pdf.multi_cell(epw, 5, pdf._safe(value), fill=True)
                pdf.ln(1.5)
            elif kind == "table":
                _render_table(pdf, value, epw)
            elif kind == "img" and embed_images:
                _render_image(pdf, session, value, epw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))


def _render_table(pdf: "PDF", rows: list[list[str]], epw: float) -> None:
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    col_w = epw / ncols
    pdf.ln(1)
    pdf.set_font(pdf.base_font, "", 8.5)
    line_h = 5
    for ri, row in enumerate(rows):
        row = row + [""] * (ncols - len(row))
        # height = tallest cell in this row
        heights = []
        for cell in row:
            n = max(1, pdf.multi_cell(col_w, line_h, pdf._safe(cell),
                                      split_only=True).__len__())
            heights.append(n * line_h)
        h = max(heights)
        if pdf.get_y() + h > pdf.h - pdf.b_margin:
            pdf.add_page()
        x0, y0 = pdf.get_x(), pdf.get_y()
        bold = "B" if ri == 0 else ""
        pdf.set_font(pdf.base_font, bold, 8.5)
        for ci, cell in enumerate(row):
            x = x0 + ci * col_w
            pdf.set_xy(x, y0)
            if ri == 0:
                pdf.set_fill_color(235, 238, 245)
            pdf.multi_cell(col_w, line_h, pdf._safe(cell), border=1,
                           fill=(ri == 0), align="L", max_line_height=line_h)
        pdf.set_xy(x0, y0 + h)
    pdf.ln(2)


def _render_image(pdf: "PDF", session: requests.Session, url: str,
                  epw: float) -> None:
    try:
        r = session.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "").lower()
        if not any(t in ctype for t in ("png", "jpeg", "jpg", "gif")):
            return
        img = BytesIO(r.content)
        w = min(epw, 120)
        if pdf.get_y() + 60 > pdf.h - pdf.b_margin:
            pdf.add_page()
        pdf.image(img, w=w)
        pdf.ln(3)
    except Exception:
        return  # images are best-effort; never fail the whole scrape


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def crawl(start_url: str, max_pages: int, delay: float,
          session: requests.Session) -> list[dict]:
    start = normalize(start_url)
    queue = deque([start])
    visited = set()
    pages = []
    while queue and len(pages) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        print(f"[{len(pages) + 1}/{max_pages}] {url}")
        html = fetch(session, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "lxml")
        link_soup = BeautifulSoup(html, "lxml")  # links before tags are stripped
        title, blocks = extract_blocks(soup, url)
        if blocks:
            pages.append({"url": url, "title": title, "blocks": blocks})
        for link in extract_links(url, link_soup):
            if same_site(start, link) and link not in visited:
                queue.append(link)
        if delay:
            time.sleep(delay)
    return pages


def scrape_single(url: str, session: requests.Session) -> list[dict]:
    print(f"[1/1] {url}")
    html = fetch(session, url)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    title, blocks = extract_blocks(soup, url)
    return [{"url": url, "title": title, "blocks": blocks}]


def main() -> int:
    ap = argparse.ArgumentParser(description="Scrape a website into a PDF.")
    ap.add_argument("url", help="Starting URL (include http:// or https://)")
    ap.add_argument("-o", "--output", help="Output PDF path")
    ap.add_argument("--crawl", action="store_true",
                    help="Follow internal links and scrape the whole site")
    ap.add_argument("--max-pages", type=int, default=150,
                    help="Max pages to scrape in crawl mode (default 150)")
    ap.add_argument("--delay", type=float, default=0.4,
                    help="Politeness delay between requests, seconds")
    ap.add_argument("--images", action="store_true",
                    help="Download and embed images (slower, bigger PDF)")
    ap.add_argument("--insecure", action="store_true",
                    help="Skip TLS certificate verification from the start")
    args = ap.parse_args()

    url = args.url
    if not urlparse(url).scheme:
        url = "https://" + url

    out = Path(args.output) if args.output else Path(
        urlparse(url).netloc.replace(":", "_") + ".pdf")

    session = requests.Session()
    if args.insecure:
        session.verify = False
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print(f"Scraping {'site' if args.crawl else 'page'}: {url}")
    pages = (crawl(url, args.max_pages, args.delay, session)
             if args.crawl else scrape_single(url, session))

    if not pages:
        print("No content scraped. Exiting.")
        return 1

    total_blocks = sum(len(p["blocks"]) for p in pages)
    print(f"\nScraped {len(pages)} page(s), {total_blocks} content blocks.")
    print("Rendering PDF...")
    render_pdf(pages, url, out, session, args.images)
    print(f"Saved: {out.resolve()}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
