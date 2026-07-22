# -*- coding: utf-8 -*-
"""
================================================================================
 0_crawling_newspaper.py  :  [Step 0] Build the input dataset by crawling news
================================================================================
 Run order : (0) crawling  ->  (1) preprocessing  ->  (2) topic selection  ->  (3) LDA analysis

 This script merges two crawlers and applies the article-selection procedure
 described in the paper:

   "Articles published between 1997 and 2021 were collected. Articles related to
    the study area were selected using the location-specific keyword '경안천'
    (Kyung-An Stream). Duplicate and non-environmental articles were subsequently
    removed. Exact duplicates were identified using canonical URLs and article
    metadata, including the news outlet, publication date, and title, and
    near-duplicate articles were identified based on normalized-title similarity.
    Non-environmental articles were excluded through keyword-based relevance
    screening."

 PIPELINE
   [1] Collect EVERY article in the period. No keyword filter is applied while
       crawling -- the whole archive is downloaded first, and the '경안천'
       selection happens afterwards in step [3], exactly as the paper describes.
       - BigKindsCrawler : Selenium (headless Chrome), reads a news-ID list
                           exported from BigKinds  -> primary source (1997-2021)
       - NaverNewsCrawler: requests + BeautifulSoup, walks the news list day by
                           day and downloads every article (original approach)
   [2] Merge into one common schema
       source | outlet | date | title | url | canonical_url | content
   [3] Study-area selection      : keep articles containing "경안천"
   [4] Exact-duplicate removal   : canonical URL, then (outlet, date, title)
   [5] Near-duplicate removal    : normalized-title similarity
   [6] Relevance screening       : environmental keyword screening
   [7] Export                    : Date, content  -> input for 1_preprocessing.py

 Every stage prints the remaining article count, so the numbers can be reported
 directly in the paper (collected -> selected -> deduplicated -> screened).

 NOTE
   - BigKinds requires a news-ID list file (one ID per line), exported from a
     BigKinds keyword search. This mirrors the original notebook's approach.
   - The Naver crawler follows the original 2019 script's structure. Naver has
     since changed its page layout, so the CSS selectors in NAVER_SELECTORS may
     need to be updated before use. Set SOURCES = ("bigkinds",) to skip Naver.
   - Crawling delays are intentional. Do not remove them.
================================================================================
"""

import os
import re
import time
import random
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ============================================================================
#  CONFIG  -- edit for your environment
# ============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "news_data")

# --- collection period (paper: 1997-2021) ---
START_DATE = "1997-01-01"
END_DATE   = "2021-12-31"

# --- study-area keyword (paper: location-specific keyword) ---
AREA_KEYWORD = "경안천"

# --- which sources to use ---
SOURCES = ("bigkinds",)          # e.g. ("bigkinds", "naver")

# --- Naver news list (crawls EVERY article in the category, then filter later) ---
NAVER_SID1     = "102"   # section: 102 = society (사회)   -- original script value
NAVER_SID2     = "252"   # sub-section                     -- original script value
NAVER_MAX_PAGE = 100     # max list pages per day

# --- BigKinds ---
BIGKINDS_ID_FILE = os.path.join(OUTPUT_DIR, "bigkinds_news_ids.txt")  # one news ID per line
BIGKINDS_URL     = "https://www.bigkinds.or.kr/v2/news/newsDetailView.do?newsId="
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# --- crawling politeness (do not lower these) ---
DELAY_MIN, DELAY_MAX = 5, 8      # seconds between article requests

# --- de-duplication / screening ---
NEAR_DUP_THRESHOLD  = 0.85       # normalized-title similarity threshold
NEAR_DUP_WINDOW_DAY = True       # compare near-duplicates within the same date
MIN_ENV_HITS        = 1          # minimum environmental keyword hits to keep

# Environmental keyword list for relevance screening.
# (derived from the study's own topic results; edit to match your criteria)
ENV_KEYWORDS = [
    "수질", "오염", "하천", "환경", "생태", "폐수", "상수원", "팔당호", "팔당",
    "수변구역", "환경부", "하수", "쓰레기", "정화", "복원", "습지", "녹지",
    "공원", "자연", "방류", "배출", "준설", "수질개선", "오염원", "생태계",
    "상수도", "취수", "정수", "수생", "어류", "녹조", "부영양화", "생물",
    "침전", "총인", "BOD", "COD", "수계", "유역", "물환경",
]

# --- output (input file for 1_preprocessing.py) ---
OUTPUT_CSV      = os.path.join(OUTPUT_DIR, "NewsResult_kyan_crawled.csv")
RAW_CSV         = os.path.join(OUTPUT_DIR, "crawled_raw.csv")        # all articles, before filtering
NAVER_INDEX_CSV = os.path.join(OUTPUT_DIR, "naver_index.csv")        # article index (resume point)
OUTPUT_ENCODING = "utf-8-sig"    # set to "cp949" to match 1_preprocessing.py as-is

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLUMNS = ["source", "outlet", "date", "title", "url", "canonical_url", "content"]


# ============================================================================
#  Utilities
# ============================================================================
def canonicalize_url(url):
    """Reduce a URL to a canonical form so the same article maps to one key."""
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    try:
        p = urlparse(url)
    except ValueError:
        return url

    host = (p.netloc or "").lower().replace("www.", "")
    path = (p.path or "").rstrip("/")
    q = parse_qs(p.query or "")

    # Naver articles are uniquely identified by oid (outlet) + aid (article id).
    # Both URL forms must collapse to the same key:
    #   old : news.naver.com/main/read.nhn?oid=025&aid=0003123456
    #   new : n.news.naver.com/article/025/0003123456
    if "naver" in host:
        if "oid" in q and "aid" in q:
            return "naver:%s/%s" % (q["oid"][0], q["aid"][0])
        m = re.search(r"/article(?:/comment)?/(\d+)/(\d+)", path)
        if m:
            return "naver:%s/%s" % (m.group(1), m.group(2))
    # BigKinds articles are identified by newsId
    if "bigkinds" in host and "newsId" in q:
        return "bigkinds:%s" % q["newsId"][0]

    # generic: drop tracking/query/fragment
    return "%s%s" % (host, path)


def normalize_title(title):
    """Normalize a title for similarity comparison."""
    if not title or not isinstance(title, str):
        return ""
    t = unicodedata.normalize("NFKC", title)
    t = t.lower()
    t = re.sub(r"\[[^\]]*\]", " ", t)      # [단독], [속보] ...
    t = re.sub(r"\([^)]*\)", " ", t)       # (사진), (종합) ...
    t = re.sub(r"<[^>]*>", " ", t)
    t = re.sub(r"[^0-9a-z가-힣]", "", t)   # keep letters/digits only
    return t.strip()


def count_env_hits(text):
    """Count how many distinct environmental keywords appear in the text."""
    if not text or not isinstance(text, str):
        return 0
    return sum(1 for kw in ENV_KEYWORDS if kw.lower() in text.lower())


def _sleep():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


# ============================================================================
#  [1-A] BigKinds crawler  (Selenium, from crawing_bigkind.ipynb)
# ============================================================================
class BigKindsCrawler:
    """Fetch article bodies from BigKinds for a list of news IDs.

    The news-ID list is exported from a BigKinds keyword search
    (keyword: AREA_KEYWORD, period: START_DATE ~ END_DATE).
    """

    def __init__(self, id_file=BIGKINDS_ID_FILE):
        self.id_file = id_file
        self.driver = None

    def _start_driver(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        try:
            import chromedriver_autoinstaller as chromedriver
            chromedriver.install()
        except Exception:
            pass

        options = Options()
        options.add_argument("user-agent=" + USER_AGENT)
        options.add_argument("headless")
        self.driver = webdriver.Chrome(options=options)

    def crawl(self):
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        if not os.path.exists(self.id_file):
            print("    [BigKinds] ID file not found, skipping:", self.id_file)
            return pd.DataFrame(columns=COLUMNS)

        with open(self.id_file, "r", encoding="utf-8") as f:
            news_ids = [ln.strip() for ln in f if ln.strip()]
        print("    [BigKinds] %d news IDs loaded" % len(news_ids))

        self._start_driver()
        rows = []
        try:
            for n, news_id in enumerate(news_ids, 1):
                url = BIGKINDS_URL + news_id
                try:
                    self.driver.get(url)
                    _sleep()
                    wait = WebDriverWait(self.driver, 10)
                    element = wait.until(EC.presence_of_element_located(("id", "content")))
                    content = re.sub(r"\s+", " ", element.text).strip()

                    # metadata (best-effort; layout may vary)
                    def _text(css):
                        try:
                            return self.driver.find_element("css selector", css).text.strip()
                        except Exception:
                            return ""

                    title = _text("h1.title") or _text(".news-title") or ""
                    outlet = _text(".provider") or _text(".news-provider") or ""
                    date_txt = _text(".date") or _text(".news-date") or ""

                    rows.append({
                        "source": "bigkinds",
                        "outlet": outlet,
                        "date": date_txt,
                        "title": title,
                        "url": url,
                        "canonical_url": canonicalize_url(url),
                        "content": content,
                    })
                except Exception as e:
                    print("    [BigKinds] failed (%s): %s" % (news_id, e))
                if n % 50 == 0:
                    print("    [BigKinds] %d/%d" % (n, len(news_ids)))
        finally:
            if self.driver:
                self.driver.quit()

        df = pd.DataFrame(rows, columns=COLUMNS)
        print("    [BigKinds] collected %d articles" % len(df))
        return df


# ============================================================================
#  [1-B] Naver news crawler  (requests + BeautifulSoup, from navernews_cralwer.py)
# ============================================================================
NAVER_LIST_URL = "https://news.naver.com/main/list.naver"
NAVER_SELECTORS = {           # update these if Naver changes its layout
    "list_link": "dt > a",            # article links on the day's list page
    "list_outlet": "span.writing",    # outlet name on the list page
    "article_title": "#title_area, #articleTitle, h2.media_end_head_headline",
    "article_date": ".media_end_head_info_datestamp_time, span.t11",
    "article_body": "#dic_area, #articleBodyContents, #newsct_article",
}


class NaverNewsCrawler:
    """Download EVERY article in a Naver news category over the date range.

    Follows the original script: walk the day-by-day news list and take all
    articles. No keyword filter is applied here -- the whole archive is
    collected first and the '경안천' selection is done later in
    select_study_area(), as described in the paper.

    Two-stage structure kept from the original:
      indexing()          -> collect article links / titles / outlets / dates
      crawling_contents() -> fetch the body of each indexed article
    """

    def __init__(self, start_date=START_DATE, end_date=END_DATE):
        self.start_date = start_date
        self.end_date = end_date
        self.index_df = pd.DataFrame()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def indexing(self):
        dt_list = pd.date_range(start=self.start_date, end=self.end_date).strftime("%Y%m%d").tolist()
        rows = []
        print("    [Naver] indexing %d days (sid1=%s, sid2=%s) ..."
              % (len(dt_list), NAVER_SID1, NAVER_SID2))

        for d in dt_list:
            dt = datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d")
            prev_html = ""
            for page in range(1, NAVER_MAX_PAGE + 1):
                params = {"mode": "LS2D", "mid": "shm", "sid1": NAVER_SID1,
                          "sid2": NAVER_SID2, "date": d, "page": page}
                try:
                    html = self.session.get(NAVER_LIST_URL, params=params, timeout=15).text
                except requests.RequestException as e:
                    print("    [Naver] list failed %s p%d: %s" % (d, page, e))
                    break

                # the last page repeats itself -> stop
                if html == prev_html:
                    break
                prev_html = html

                soup = BeautifulSoup(html, "lxml")
                links = soup.select(NAVER_SELECTORS["list_link"])
                if not links:
                    break
                outlets = [s.get_text(strip=True)
                           for s in soup.select(NAVER_SELECTORS["list_outlet"])]

                idx = 0
                for tag in links:
                    link = tag.get("href", "")
                    title = tag.get_text(strip=True)
                    if not title or title == "동영상기사" or "javascript" in link:
                        continue
                    outlet = outlets[idx] if idx < len(outlets) else ""
                    idx += 1
                    rows.append({
                        "source": "naver",
                        "outlet": outlet,
                        "date": dt,
                        "title": title,
                        "url": link,
                        "canonical_url": canonicalize_url(link),
                        "content": "",
                    })
                _sleep()
            print("    [Naver] %s -> %d articles indexed so far" % (dt, len(rows)))

        self.index_df = pd.DataFrame(rows, columns=COLUMNS)
        # a day's list can repeat an article across pages
        self.index_df = (self.index_df
                         .drop_duplicates(subset=["canonical_url"], keep="first")
                         .reset_index(drop=True))
        self.index_df.to_csv(NAVER_INDEX_CSV, index=False, encoding=OUTPUT_ENCODING)
        print("    [Naver] indexed %d articles -> %s"
              % (len(self.index_df), os.path.basename(NAVER_INDEX_CSV)))
        return self.index_df

    def crawling_contents(self):
        if self.index_df.empty:
            return self.index_df

        for i, row in self.index_df.iterrows():
            try:
                html = self.session.get(row["url"], timeout=15).text
                soup = BeautifulSoup(html, "lxml")

                def _sel(css):
                    el = soup.select_one(css)
                    return el.get_text(" ", strip=True) if el else ""

                body = _sel(NAVER_SELECTORS["article_body"])
                self.index_df.at[i, "content"] = re.sub(r"\s+", " ", body).strip()
                if not row["title"]:
                    self.index_df.at[i, "title"] = _sel(NAVER_SELECTORS["article_title"])
                if not row["date"]:
                    self.index_df.at[i, "date"] = _sel(NAVER_SELECTORS["article_date"])
            except Exception as e:
                print("    [Naver] body failed (%s): %s" % (row["url"], e))
            if i and i % 20 == 0:
                print("    [Naver] %d/%d" % (i, len(self.index_df)))
            _sleep()

        print("    [Naver] collected %d article bodies" % len(self.index_df))
        return self.index_df


# ============================================================================
#  [3-6] Selection / de-duplication / screening
# ============================================================================
def select_study_area(df, keyword=AREA_KEYWORD):
    """[3] Keep only articles related to the study area."""
    hay = (df["title"].fillna("") + " " + df["content"].fillna(""))
    return df[hay.str.contains(keyword, na=False)].copy()


def drop_exact_duplicates(df):
    """[4] Exact duplicates: canonical URL, then (outlet, publication date, title)."""
    before = len(df)
    df = df[df["canonical_url"].notna()].copy()

    # (a) canonical URL
    has_url = df["canonical_url"].astype(str).str.len() > 0
    df = pd.concat([
        df[has_url].drop_duplicates(subset=["canonical_url"], keep="first"),
        df[~has_url],
    ]).sort_index()

    # (b) article metadata: outlet + publication date + title
    df["_ntitle"] = df["title"].map(normalize_title)
    df = df.drop_duplicates(subset=["outlet", "date", "_ntitle"], keep="first")
    print("    [4] exact duplicates removed : %d -> %d" % (before, len(df)))
    return df


def drop_near_duplicates(df, threshold=NEAR_DUP_THRESHOLD, same_day=NEAR_DUP_WINDOW_DAY):
    """[5] Near-duplicates: normalized-title similarity."""
    before = len(df)
    if "_ntitle" not in df.columns:
        df["_ntitle"] = df["title"].map(normalize_title)

    # (a) identical normalized titles -> duplicates regardless of date
    df = df[(df["_ntitle"] == "") | ~df.duplicated(subset=["_ntitle"], keep="first")].copy()

    # (b) fuzzy similarity, compared within the same publication date
    drop_idx = set()
    groups = df.groupby("date") if same_day else [("all", df)]
    for _, g in groups:
        idxs = [i for i in g.index if g.at[i, "_ntitle"]]
        for a_pos in range(len(idxs)):
            ia = idxs[a_pos]
            if ia in drop_idx:
                continue
            ta = df.at[ia, "_ntitle"]
            for b_pos in range(a_pos + 1, len(idxs)):
                ib = idxs[b_pos]
                if ib in drop_idx:
                    continue
                if SequenceMatcher(None, ta, df.at[ib, "_ntitle"]).ratio() >= threshold:
                    drop_idx.add(ib)
    df = df.drop(index=list(drop_idx))
    print("    [5] near duplicates removed  : %d -> %d" % (before, len(df)))
    return df


def screen_environmental(df, min_hits=MIN_ENV_HITS):
    """[6] Exclude non-environmental articles via keyword-based relevance screening."""
    before = len(df)
    hay = (df["title"].fillna("") + " " + df["content"].fillna(""))
    df = df.assign(env_hits=hay.map(count_env_hits))
    df = df[df["env_hits"] >= min_hits].copy()
    print("    [6] non-environmental removed: %d -> %d" % (before, len(df)))
    return df


# ============================================================================
#  MAIN
# ============================================================================
def main():
    print("=" * 70)
    print(" Step 0 : crawling newspapers  (%s ~ %s, keyword='%s')"
          % (START_DATE, END_DATE, AREA_KEYWORD))
    print("=" * 70)

    # ---- [1] collect -----------------------------------------------------
    frames = []
    if "bigkinds" in SOURCES:
        print("\n[1] Collecting from BigKinds ...")
        frames.append(BigKindsCrawler().crawl())
    if "naver" in SOURCES:
        print("\n[1] Collecting from Naver ...")
        nc = NaverNewsCrawler()
        nc.indexing()
        frames.append(nc.crawling_contents())

    if not frames:
        print("No source selected. Set SOURCES and run again.")
        return

    # ---- [2] merge -------------------------------------------------------
    df = pd.concat(frames, ignore_index=True)[COLUMNS]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["date"] >= START_DATE) | df["date"].isna()]
    df = df[(df["date"] <= END_DATE) | df["date"].isna()]
    df.to_csv(RAW_CSV, index=False, encoding=OUTPUT_ENCODING)
    print("\n[2] collected (merged)          : %d articles  -> %s"
          % (len(df), os.path.basename(RAW_CSV)))

    # ---- [3]~[6] selection pipeline --------------------------------------
    df = select_study_area(df)
    print("    [3] study-area selected      : %d" % len(df))
    df = drop_exact_duplicates(df)
    df = drop_near_duplicates(df)
    df = screen_environmental(df)

    # ---- [7] export : Date, content  (input for 1_preprocessing.py) ------
    out = pd.DataFrame({
        "Date": df["date"].dt.strftime("%Y-%m-%d"),
        "content": df["content"].fillna("").str.strip(),
    })
    out = out[out["content"].str.len() > 0]
    out.to_csv(OUTPUT_CSV, index=False, encoding=OUTPUT_ENCODING)

    print("\n" + "=" * 70)
    print(" FINAL input dataset : %d articles" % len(out))
    print(" saved -> %s" % OUTPUT_CSV)
    if not out.empty:
        print(" period: %s ~ %s" % (out["Date"].min(), out["Date"].max()))
    print(" next  : set INPUT_CSV in 1_preprocessing.py to this file, then run it")
    print("=" * 70)


if __name__ == "__main__":
    main()
