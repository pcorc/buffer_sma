"""
Latest-Day ECR Rankings Exporter — pure-Python port of cost-ratio-enhanced
===========================================================================

Reproduces, in Python, exactly what the ECR web tool (ecr.vestfin.com) renders
on its front page: the Enhanced Cost Ratio ranking of all First Trust buffer
ETFs as of the most recent data.

Where this logic lives in the TypeScript repo (cost-ratio-enhanced):

  ECR methodology (the formulas)
      src/lib/calculator/enhanced-cost-ratio.ts
  Applied at scrape time (components stored per snapshot)
      src/lib/scraper/orchestrator.ts -> runFullScrape(), via computeAllComponents()
  Main-table Excel download (ASP.NET postback)
      src/lib/scraper/main-table.ts -> scrapeMainTable()
  Latest-day selection + final ECR + sort that the front end renders
      src/lib/data/etfs.ts -> fetchLatestEtfs()  (served by /api/etfs)

Two data sources:

  --source ft   (default) Scrape First Trust live, exactly like the website's
                3x-daily scraper: ASP.NET postback on the buffer-strategy fund
                list -> 28-column Excel -> compute components -> ECR -> rank.
                Works anywhere with internet; no credentials.

  --source api  Pull the website's own JSON (GET <url>/api/etfs), which
                contains the ECR values the front end is showing right now.
                Use this on the office network if you want the site's numbers
                verbatim: python export_latest_ecr.py --source api

UNITS NOTE: the fund-list Excel reports values as DECIMALS (0.1431 = 14.31%),
unlike the per-fund historical Excel behind input_data/data.csv, which uses
percent units. The DBB Score formula exp(DBB * 5) assumes decimals.

Usage:
    python export_latest_ecr.py                   # scrape FT, rank, export xlsx
    python export_latest_ecr.py --top 15          # also print top 15 to console
    python export_latest_ecr.py --include-quarterly
    python export_latest_ecr.py --source api --url https://ecr.vestfin.com

Output: output/ecr_rankings_<YYYY-MM-DD_HHMM>.xlsx

Requires: pandas, openpyxl. Network access to www.ftportfolios.com.
"""

import argparse
import http.cookiejar
import io
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Same strategy-filtered URL as FT_FUND_LIST_URL in src/lib/utils/constants.ts:
# the four buffer strategy types (Buffer, Cap Buffer, Deep Buffer, Max Buffer).
FUND_LIST_URL = (
    "https://www.ftportfolios.com/retail/etf/targetoutcomefundlist.aspx"
    "?Strategy=BUF&Strategy=CBUF&Strategy=DBUF&Strategy=MBUF"
)
DOWNLOAD_CONTROL = "ctl00$ContentPlaceHolder1$targetoutcomeretail$lnkDownloadToExcel"


# ============================================================================
# ECR methodology — 1:1 port of src/lib/calculator/enhanced-cost-ratio.ts
# All inputs are DECIMALS (0.0915 = 9.15%), None = value not published.
# ============================================================================

def calc_dbb_score(downside_before_buffer):
    """DBB Score = exp(Downside Before Buffer x 5). Higher = less downside."""
    if downside_before_buffer is None:
        return None
    return math.exp(downside_before_buffer * 5)


def calc_buffer_integrity(remaining_buffer, downside_before_buffer, initial_buffer):
    """Buffer Integrity = max(0, Remaining Buffer + DBB) / Initial Buffer."""
    if (remaining_buffer is None or downside_before_buffer is None
            or initial_buffer is None or initial_buffer == 0):
        return None
    return max(0.0, remaining_buffer + downside_before_buffer) / initial_buffer


def calc_cap_integrity(remaining_cap, initial_cap):
    """Cap Integrity = min(Remaining Cap / Initial Cap, 1). None for capless ETFs."""
    if remaining_cap is None or initial_cap is None or initial_cap == 0:
        return None
    return min(remaining_cap / initial_cap, 1.0)


def calc_time_scaling(days_remaining):
    """Time Scaling = 1 - ln(Days Remaining / 365)."""
    if days_remaining is None or days_remaining <= 0:
        return 1.0
    return 1.0 - math.log(days_remaining / 365)


def calc_enhanced_cost_ratio(dbb_score, buffer_integrity, cap_integrity, time_scaling):
    """
    ECR = (DBB Score + Buffer Integrity + Cap Integrity) / Time Scaling.
    When Cap Integrity is None (no cap), average DBB and Buffer instead
    so the score stays comparable.
    """
    if dbb_score is None or buffer_integrity is None:
        return None
    if time_scaling == 0:
        return None
    if cap_integrity is None:
        numerator = (dbb_score + buffer_integrity) / 2
    else:
        numerator = dbb_score + buffer_integrity + cap_integrity
    return numerator / time_scaling


def compute_all_components(row):
    """Port of computeAllComponents(): raw snapshot fields -> the four scores."""
    return {
        "dbb_score": calc_dbb_score(row["downside_before_buffer_net"]),
        "buffer_integrity": calc_buffer_integrity(
            row["remaining_buffer_net"],
            row["downside_before_buffer_net"],
            row["buffer_net"],
        ),
        "cap_integrity": calc_cap_integrity(
            row["remaining_cap_net"], row["fund_cap_net"]
        ),
        "time_scaling": calc_time_scaling(row["remaining_outcome_days"]),
    }


# ============================================================================
# Source 1: live First Trust scrape — port of src/lib/scraper/main-table.ts
# ============================================================================

def fetch_main_table_excel(retries=3):
    """ASP.NET postback on the fund list page -> 28-column Excel bytes.

    FT occasionally answers the postback with the HTML page instead of the
    file (rate limiting); a fresh attempt after a short pause resolves it.
    """
    last_err = None
    for attempt in range(retries):
        if attempt > 0:
            time.sleep(10)
        try:
            return _fetch_main_table_excel_once()
        except RuntimeError as err:
            last_err = err
    raise RuntimeError(f"Fund-list Excel download failed after {retries} "
                       f"attempts: {last_err}")


def _fetch_main_table_excel_once():
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    headers = {"User-Agent": USER_AGENT}

    req = urllib.request.Request(FUND_LIST_URL, headers=headers)
    with opener.open(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    # Harvest form state (__VIEWSTATE etc.) the way cheerio does in TS
    form = {}
    for tag in re.findall(r"<input[^>]*>", html):
        name = re.search(r'name="([^"]*)"', tag)
        if not name:
            continue
        input_type = (re.search(r'type="([^"]*)"', tag) or (None, "text"))[1].lower()
        if input_type in ("submit", "image", "button"):
            continue
        if input_type == "checkbox" and "checked" not in tag:
            continue
        value = re.search(r'value="([^"]*)"', tag)
        form[name.group(1)] = value.group(1) if value else ""
    if "__VIEWSTATE" not in form:
        raise RuntimeError("No __VIEWSTATE on fund list page — FT layout changed?")

    # Trigger the "Download to Excel" link's postback
    form["__EVENTTARGET"] = DOWNLOAD_CONTROL
    form["__EVENTARGUMENT"] = ""

    req = urllib.request.Request(
        FUND_LIST_URL,
        data=urllib.parse.urlencode(form).encode(),
        headers={**headers,
                 "Content-Type": "application/x-www-form-urlencoded",
                 "Referer": FUND_LIST_URL},
    )
    with opener.open(req, timeout=120) as resp:
        content_type = resp.headers.get("Content-Type", "")
        data = resp.read()
    if "spreadsheet" not in content_type and "excel" not in content_type:
        raise RuntimeError(f"Expected an Excel file, got {content_type} — "
                           f"FT may have changed the download control id")
    return data


def parse_main_table(excel_bytes):
    """Parse the fund-list Excel into one row per fund (values are decimals)."""
    raw = pd.read_excel(io.BytesIO(excel_bytes), header=None)

    header_idx = None
    for i in range(min(len(raw), 10)):
        if str(raw.iloc[i, 0]).strip() == "Ticker":
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("No 'Ticker' header row found in fund-list Excel")

    df = raw.iloc[header_idx + 1:].copy()
    df.columns = [str(h).strip() for h in raw.iloc[header_idx]]
    df = df[df["Ticker"].astype(str).str.match(r"^[A-Z]{2,10}$", na=False)].copy()

    # Same field mapping as scrapeMainTable() in main-table.ts
    column_map = {
        "Ticker": "ticker",
        "Strategy Type": "strategy_type",
        "Series": "series",
        "Reference Asset": "reference_asset",
        "Outcome Period Start Date": "outcome_period_start",
        "Outcome Period End Date": "outcome_period_end",
        "Fund Cap Net": "fund_cap_net",
        "Buffer Net": "buffer_net",
        "Fund Value (USD)": "fund_value",
        "Fund Return": "fund_return",
        "Reference Asset Value (USD)": "reference_asset_value",
        "Reference Asset Return": "reference_asset_return",
        "Remaining Cap Net": "remaining_cap_net",
        "Remaining Buffer Net": "remaining_buffer_net",
        "Downside Before Buffer Net": "downside_before_buffer_net",
        "Remaining Outcome Period (days)": "remaining_outcome_days",
    }
    missing = [c for c in column_map if c not in df.columns]
    if missing:
        raise RuntimeError(f"Fund-list Excel is missing columns {missing} — "
                           f"FT may have renamed headers")
    df = df[list(column_map)].rename(columns=column_map)

    numeric_cols = [c for c in df.columns if c not in
                    ("ticker", "strategy_type", "series", "reference_asset",
                     "outcome_period_start", "outcome_period_end")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("outcome_period_start", "outcome_period_end"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df.reset_index(drop=True)


def rankings_from_ft():
    """Scrape FT now and build the rankings — the website's full pipeline."""
    df = parse_main_table(fetch_main_table_excel())
    print(f"  fetched {len(df)} funds from First Trust")

    records = df.to_dict("records")
    for rec in records:
        # pandas NaN -> None so the calculator's null handling matches TS
        for key, value in rec.items():
            if isinstance(value, float) and math.isnan(value):
                rec[key] = None
        rec.update(compute_all_components(rec))
        rec["ecr"] = calc_enhanced_cost_ratio(
            rec["dbb_score"], rec["buffer_integrity"],
            rec["cap_integrity"], rec["time_scaling"],
        )
    out = pd.DataFrame(records)
    out["as_of"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return out


# ============================================================================
# Source 2: the website's own JSON — what the front end is showing right now
# ============================================================================

def rankings_from_api(base_url):
    """GET <base_url>/api/etfs and normalize to the same output shape."""
    url = base_url.rstrip("/") + "/api/etfs"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)

    out = pd.DataFrame(payload["etfs"])
    out = out.rename(columns={
        "strategyType": "strategy_type",
        "referenceAsset": "reference_asset",
        "fundValue": "fund_value",
        "fundReturn": "fund_return",
        "referenceAssetValue": "reference_asset_value",
        "referenceAssetReturn": "reference_asset_return",
        "remainingCapNet": "remaining_cap_net",
        "remainingBufferNet": "remaining_buffer_net",
        "downsideBeforeBufferNet": "downside_before_buffer_net",
        "remainingOutcomePeriod": "remaining_outcome_days",
        "outcomePeriodStart": "outcome_period_start",
        "outcomePeriodEnd": "outcome_period_end",
        "fundCapNet": "fund_cap_net",
        "bufferNet": "buffer_net",
        "dbbScore": "dbb_score",
        "bufferIntegrity": "buffer_integrity",
        "capIntegrity": "cap_integrity",
        "timeScaling": "time_scaling",
    })
    out["as_of"] = payload.get("lastUpdated")
    print(f"  fetched {len(out)} funds from {url} (as of {payload.get('lastUpdated')})")
    return out


# ============================================================================
# Main
# ============================================================================

EXPORT_COLUMNS = [
    "ticker", "ecr", "series", "strategy_type", "reference_asset",
    "fund_value", "fund_return", "reference_asset_return",
    "remaining_cap_net", "remaining_buffer_net", "downside_before_buffer_net",
    "remaining_outcome_days", "outcome_period_start", "outcome_period_end",
    "fund_cap_net", "buffer_net",
    "dbb_score", "buffer_integrity", "cap_integrity", "time_scaling", "as_of",
]


def main():
    parser = argparse.ArgumentParser(
        description="Export the latest-day ECR rankings (what the web tool shows).")
    parser.add_argument("--source", choices=["ft", "api"], default="ft",
                        help="ft = scrape First Trust live (default); "
                             "api = pull the website's own /api/etfs JSON")
    parser.add_argument("--url", default="https://ecr.vestfin.com",
                        help="website base URL for --source api")
    parser.add_argument("--include-quarterly", action="store_true",
                        help="keep Quarterly-series funds (the homepage hides them)")
    parser.add_argument("--top", type=int, default=10,
                        help="how many rows to print to the console")
    args = parser.parse_args()

    print("=" * 80)
    print("LATEST-DAY ECR RANKINGS")
    print("=" * 80)

    out = rankings_from_api(args.url) if args.source == "api" else rankings_from_ft()

    # Mirror fetchLatestEtfs() in src/lib/data/etfs.ts: drop Quarterly,
    # sort by ECR descending with null ECRs last
    if not args.include_quarterly:
        out = out[out["series"] != "Quarterly"].copy()
    out = out.sort_values("ecr", ascending=False, na_position="last").reset_index(drop=True)
    out.index += 1
    out.index.name = "rank"

    export = out[[c for c in EXPORT_COLUMNS if c in out.columns]]
    print(f"\nTop {args.top} of {len(export)} funds:")
    print(export.head(args.top)[
        ["ticker", "ecr", "series", "dbb_score", "buffer_integrity",
         "cap_integrity", "time_scaling"]
    ].round(4).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = OUTPUT_DIR / f"ecr_rankings_{stamp}.xlsx"
    export.to_excel(out_path, sheet_name="ECR Rankings")
    print(f"\n✅ Exported {len(export)} funds → {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
