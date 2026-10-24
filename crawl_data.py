"""
crawl_openalex.py
Thu thập metadata bài báo (paper) Khoa học Máy tính của 20 trường từ OpenAlex API.
TOÀN BỘ số liệu lấy trực tiếp từ API, không sinh / không random / không nội suy.

Chuẩn bị:
    1. Tạo tài khoản miễn phí tại https://openalex.org, lấy key ở https://openalex.org/settings/api
       (từ 2026 OpenAlex yêu cầu API key; key miễn phí có 1 USD/ngày, script này chỉ tốn vài cent).
    2. Đặt key vào biến môi trường (KHÔNG viết thẳng key vào code):
         Windows PowerShell:  $env:OPENALEX_API_KEY="xxxx"
         Windows CMD:         set OPENALEX_API_KEY=xxxx
         Linux/macOS:         export OPENALEX_API_KEY=xxxx
    3. pip install requests pandas

Chạy:
    python crawl_openalex.py --check     # chỉ đếm số paper theo từng trường/năm (gần như miễn phí)
    python crawl_openalex.py             # crawl thật -> openalex_papers.csv
"""
import argparse
import difflib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ------------------------------------------------------------------ CẤU HÌNH
API = "https://api.openalex.org"
API_KEY = os.environ.get("OPENALEX_API_KEY", "")
MAILTO = os.environ.get("OPENALEX_MAILTO", "")  # tuỳ chọn: email liên hệ

YEAR_FROM, YEAR_TO = 2019, 2024      # khoảng thời gian crawl (đủ thời gian để paper có trích dẫn)
MAX_PER_INST_YEAR = 80               # tối đa số paper / trường / năm (giữ cân bằng giữa các trường)
TARGET_ROWS = 5000                   # KPI số dòng
CS_FIELD_ID = 17                     # OpenAlex field 17 = Computer Science
WORK_TYPES = "article|preprint"      # loại tài liệu

OUT_CSV = "openalex_papers.csv"
INST_CSV = "institutions_resolved.csv"
CACHE_DIR = Path("cache_openalex")

# (nhãn hiển thị, tên để tìm trên OpenAlex, mã quốc gia ISO-2, nhóm)
UNIVERSITIES = [
    # ---- 10 trường công nghệ hàng đầu Việt Nam
    ("HUST", "Hanoi University of Science and Technology", "VN", "Vietnam"),
    ("VNU Hanoi (UET)", "Vietnam National University, Hanoi", "VN", "Vietnam"),
    ("PTIT", "Posts and Telecommunications Institute of Technology", "VN", "Vietnam"),
    ("UIT", "University of Information Technology", "VN", "Vietnam"),
    ("HCMUT", "Ho Chi Minh City University of Technology", "VN", "Vietnam"),
    ("FPT University", "FPT University", "VN", "Vietnam"),
    ("Ton Duc Thang University", "Ton Duc Thang University", "VN", "Vietnam"),
    ("Duy Tan University", "Duy Tan University", "VN", "Vietnam"),
    ("Phenikaa University", "Phenikaa University", "VN", "Vietnam"),
    ("VinUniversity", "VinUniversity", "VN", "Vietnam"),
    # ---- 10 trường quốc tế hàng đầu về AI
    ("CMU", "Carnegie Mellon University", "US", "Global"),
    ("MIT", "Massachusetts Institute of Technology", "US", "Global"),
    ("Stanford", "Stanford University", "US", "Global"),
    ("UC Berkeley", "University of California, Berkeley", "US", "Global"),
    ("Tsinghua", "Tsinghua University", "CN", "Global"),
    ("Peking University", "Peking University", "CN", "Global"),
    ("Oxford", "University of Oxford", "GB", "Global"),
    ("ETH Zurich", "ETH Zurich", "CH", "Global"),
    ("Toronto", "University of Toronto", "CA", "Global"),
    ("NUS", "National University of Singapore", "SG", "Global"),
]

# Nếu tìm tự động ra sai trường, điền ID thủ công ở đây, ví dụ {"UIT": "I1234567890"}
INSTITUTION_ID_OVERRIDE = {}

WORK_FIELDS = ",".join([
    "id", "doi", "title", "publication_year", "publication_date", "type", "language",
    "cited_by_count", "fwci", "citation_normalized_percentile", "referenced_works_count",
    "authorships", "countries_distinct_count", "institutions_distinct_count",
    "open_access", "primary_topic", "primary_location", "keywords",
])


# ------------------------------------------------------------------ HTTP
def get_json(session, path, params, retries=6):
    params = dict(params)
    if API_KEY:
        params["api_key"] = API_KEY
    if MAILTO:
        params["mailto"] = MAILTO
    for attempt in range(retries):
        try:
            r = session.get(f"{API}{path}", params=params, timeout=40)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"   ! lỗi mạng ({e.__class__.__name__}), thử lại sau {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403):
            sys.exit("API key sai hoặc thiếu. Kiểm tra biến môi trường OPENALEX_API_KEY.\n" + r.text[:300])
        if r.status_code == 429:
            wait = 5 * 2 ** attempt
            print(f"   ! 429 (hết hạn mức / quá nhanh), chờ {wait}s. Nếu lặp lại: hết ngân sách trong ngày.")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        sys.exit(f"Lỗi {r.status_code} từ OpenAlex cho {path}:\n{r.text[:500]}")
    sys.exit(f"Thất bại sau {retries} lần thử: {path} {params}")


# ------------------------------------------------------------------ TÌM ID TRƯỜNG
def resolve_institutions(session):
    rows = []
    for label, search_name, cc, region in UNIVERSITIES:
        if label in INSTITUTION_ID_OVERRIDE:
            rows.append(dict(label=label, region=region, country_code=cc, openalex_id=INSTITUTION_ID_OVERRIDE[label],
                             matched_name="(override)", works_count=None, similarity=1.0))
            continue
        data = get_json(session, "/institutions", {
            "search": search_name, "filter": f"country_code:{cc}", "per_page": 3,
            "select": "id,display_name,country_code,works_count",
        })
        results = data.get("results", [])
        if not results:
            print(f"  ✗ {label}: không tìm thấy trên OpenAlex -> điền tay vào INSTITUTION_ID_OVERRIDE")
            continue
        best = results[0]
        sim = difflib.SequenceMatcher(None, search_name.lower(), best["display_name"].lower()).ratio()
        flag = "" if sim >= 0.6 else "   <-- KHỚP YẾU, kiểm tra lại!"
        print(f"  {label:<26} -> {best['display_name']} ({best['id'].rsplit('/', 1)[-1]}, "
              f"{best.get('works_count', 0):,} works){flag}")
        rows.append(dict(label=label, region=region, country_code=cc, openalex_id=best["id"].rsplit("/", 1)[-1],
                         matched_name=best["display_name"], works_count=best.get("works_count"),
                         similarity=round(sim, 2)))
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df.to_csv(INST_CSV, index=False, encoding="utf-8-sig")
    return df


# ------------------------------------------------------------------ TRUY VẤN PAPER
def works_filter(inst_id, year):
    return (f"authorships.institutions.id:{inst_id},publication_year:{year},"
            f"primary_topic.field.id:{CS_FIELD_ID},type:{WORK_TYPES}")


def count_works(session, inst_id, year):
    data = get_json(session, "/works", {"filter": works_filter(inst_id, year), "per_page": 1, "select": "id"})
    return data["meta"]["count"]


def fetch_year(session, inst_id, year):
    """Lấy tối đa MAX_PER_INST_YEAR paper của 1 trường trong 1 năm (mới nhất trước). Có cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{inst_id}_{year}_{MAX_PER_INST_YEAR}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    works, cursor = [], "*"
    while len(works) < MAX_PER_INST_YEAR and cursor:
        data = get_json(session, "/works", {
            "filter": works_filter(inst_id, year),
            "sort": "publication_date:desc",
            "per_page": min(100, MAX_PER_INST_YEAR),
            "cursor": cursor,
            "select": WORK_FIELDS,
        })
        works.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not data.get("results"):
            break
        time.sleep(0.15)
    works = works[:MAX_PER_INST_YEAR]
    cache_file.write_text(json.dumps(works, ensure_ascii=False), encoding="utf-8")
    return works


def flatten(w, inst):
    """Chỉ trích xuất giá trị có sẵn trong JSON; thiếu thì để trống (None), không điền giả."""
    topic = w.get("primary_topic") or {}
    src = (w.get("primary_location") or {}).get("source") or {}
    oa = w.get("open_access") or {}
    kws = [k.get("display_name") for k in (w.get("keywords") or []) if k.get("display_name")]
    return {
        "work_id": (w.get("id") or "").rsplit("/", 1)[-1],
        "doi": w.get("doi"),
        "title": w.get("title"),
        "publication_date": w.get("publication_date"),
        "publication_year": w.get("publication_year"),
        "type": w.get("type"),
        "language": w.get("language"),
        "region": inst["region"],
        "institution": inst["label"],
        "institution_id": inst["openalex_id"],
        "country_code": inst["country_code"],
        "authors_count": len(w.get("authorships") or []),
        "institutions_distinct_count": w.get("institutions_distinct_count"),
        "countries_distinct_count": w.get("countries_distinct_count"),
        "referenced_works_count": w.get("referenced_works_count"),
        "cited_by_count": w.get("cited_by_count"),
        "fwci": w.get("fwci"),
        "citation_percentile": (w.get("citation_normalized_percentile") or {}).get("value"),
        "is_oa": oa.get("is_oa"),
        "oa_status": oa.get("oa_status"),
        "venue": src.get("display_name"),
        "venue_type": src.get("type"),
        "topic": topic.get("display_name"),
        "subfield": (topic.get("subfield") or {}).get("display_name"),
        "field": (topic.get("field") or {}).get("display_name"),
        "domain": (topic.get("domain") or {}).get("display_name"),
        "keywords": "; ".join(kws[:8]),
    }


# ------------------------------------------------------------------ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="chỉ đếm số paper có sẵn, không tải dữ liệu")
    args = ap.parse_args()

    if not API_KEY:
        print("⚠ Chưa có OPENALEX_API_KEY. Không có key chỉ được 0.1 USD/ngày, có thể không đủ.")

    session = requests.Session()
    session.headers.update({"User-Agent": "AI-learning-research-crawler/1.0"})
    years = list(range(YEAR_FROM, YEAR_TO + 1))

    print("🔎 Bước 1: xác định ID các trường trên OpenAlex")
    inst_df = resolve_institutions(session)
    print(f"   (đã lưu bảng đối chiếu vào {INST_CSV}: hãy mở kiểm tra các dòng 'KHỚP YẾU')\n")

    if args.check:
        print("📊 Số paper CS có sẵn (cap = %d / trường / năm)" % MAX_PER_INST_YEAR)
        total = 0
        print(f"{'Trường':<26}" + "".join(f"{y:>7}" for y in years) + f"{'Lấy được':>10}")
        for _, inst in inst_df.iterrows():
            counts = [count_works(session, inst["openalex_id"], y) for y in years]
            usable = sum(min(c, MAX_PER_INST_YEAR) for c in counts)
            total += usable
            print(f"{inst['label']:<26}" + "".join(f"{c:>7}" for c in counts) + f"{usable:>10}")
            time.sleep(0.1)
        print(f"\nƯớc tính tổng số dòng: {total:,} (KPI {TARGET_ROWS:,})")
        if total < TARGET_ROWS:
            print("→ Chưa đủ: tăng MAX_PER_INST_YEAR hoặc mở rộng YEAR_FROM/YEAR_TO.")
        return

    print("📥 Bước 2: tải paper theo từng trường × năm")
    rows = []
    for _, inst in inst_df.iterrows():
        inst = inst.to_dict()
        n_before = len(rows)
        for y in years:
            for w in fetch_year(session, inst["openalex_id"], y):
                rows.append(flatten(w, inst))
        print(f"  {inst['label']:<26} +{len(rows) - n_before} paper")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["title"]).drop_duplicates(subset=["work_id", "institution_id"])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✅ Đã lưu {len(df):,} dòng × {df.shape[1]} cột vào {OUT_CSV}")
    print("\nTỷ lệ thiếu dữ liệu từng cột (%):")
    print((df.isna().mean() * 100).round(1).sort_values(ascending=False).to_string())
    print("\nSố dòng theo nhóm:")
    print(df["region"].value_counts().to_string())
    if len(df) < TARGET_ROWS:
        print(f"\n⚠ Mới {len(df):,}/{TARGET_ROWS:,} dòng: tăng MAX_PER_INST_YEAR hoặc mở rộng khoảng năm rồi chạy lại.")


if __name__ == "__main__":
    main()