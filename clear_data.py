import re
import numpy as np
import pandas as pd

IN_CSV = "openalex_papers.csv"
OUT_CSV = "clean_openalex_papers.csv"


def infer_venue_and_type(row):
    """Suy diễn venue và venue_type trực tiếp từ tiền tố nhà xuất bản trong DOI và type."""
    doi_str = str(row["doi"] or "").lower()
    w_type = str(row["type"] or "").lower()

    if "techrxiv" in doi_str:
        return "TechRxiv (IEEE)", "repository"
    if "osf.io" in doi_str:
        return "OSF Preprints", "repository"
    if "essoar" in doi_str:
        return "Earth and Space Science Open Archive (ESSOAr)", "repository"
    if "22541/au." in doi_str:
        return "Authorea Preprints", "repository"
    if "10.31223/" in doi_str:
        return "EarthArXiv", "repository"
    if "10.32942/" in doi_str:
        return "EcoEvoRxiv", "repository"
    if "chemrxiv" in doi_str:
        return "ChemRxiv", "repository"
    if "10.33774/" in doi_str:
        return "Cambridge Open Engage", "repository"
    if "zenodo" in doi_str:
        return "Zenodo (CERN)", "repository"
    if "10.2196/preprints" in doi_str:
        return "JMIR Preprints", "repository"
    if "srxiv" in doi_str:
        return "SportRxiv", "repository"
    if "10.6703/ijase" in doi_str:
        return "International Journal of Applied Science and Engineering", "journal"
    if "10.4018/" in doi_str:
        return "IGI Global", "book series"
    if "10.21428/" in doi_str:
        return "PubPub", "journal"
    if "10.64206/" in doi_str:
        return "Open Access Science Journal", "journal"
    if w_type == "preprint":
        return "Open Preprint Repository", "repository"
    return "Academic Peer-Reviewed Venue", "journal"


def clean_only_missing_values(in_csv=IN_CSV, out_csv=OUT_CSV):
    print(f"📂 Đang đọc dữ liệu gốc từ '{in_csv}'...")
    df_raw = pd.read_csv(in_csv)
    df = df_raw.copy()

    na_before = df.isna().sum()
    print("\n📊 Số lượng ô NaN trước khi chuẩn hóa:")
    print(na_before[na_before > 0].to_string())

    # 1. CỘT 'language' (71 ô NaN): Nhận diện ngôn ngữ trực tiếp từ 'title'
    vn_pattern = re.compile(
        r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]",
        re.IGNORECASE,
    )
    mask_lang_na = df["language"].isna()
    df.loc[mask_lang_na, "language"] = df.loc[mask_lang_na, "title"].apply(
        lambda t: "vi" if bool(vn_pattern.search(str(t))) else "en"
    )

    # 2. CỘT 'venue' & 'venue_type' (101 ô NaN): Trích xuất từ tiền tố DOI & type
    mask_venue_na = df["venue"].isna() | df["venue_type"].isna()
    inferred_venues = df.loc[mask_venue_na].apply(infer_venue_and_type, axis=1)
    for idx, (v_name, v_type) in inferred_venues.items():
        if pd.isna(df.loc[idx, "venue"]):
            df.loc[idx, "venue"] = v_name
        if pd.isna(df.loc[idx, "venue_type"]):
            df.loc[idx, "venue_type"] = v_type

    # 3. CỘT 'keywords' (8 ô NaN): Tổng hợp từ 'topic', 'subfield', 'field' sẵn có
    mask_kw_na = df["keywords"].isna()
    df.loc[mask_kw_na, "keywords"] = df.loc[mask_kw_na].apply(
        lambda r: f"{str(r['topic']).lower()}; {str(r['subfield']).lower()}; {str(r['field']).lower()}",
        axis=1,
    )
    # 4. CỘT 'doi' (181 ô NaN): Chuẩn hóa bằng URI định danh gốc của OpenAlex
    mask_doi_na = df["doi"].isna()
    df.loc[mask_doi_na, "doi"] = "https://openalex.org/" + df.loc[mask_doi_na, "work_id"].astype(str)

    # 5. CỘT 'fwci' & 'citation_percentile' (2,097 ô NaN của nhóm preprint):
    #    Tính toán xác định dựa trên đường chuẩn trích dẫn của 5,501 bài 'article'
    #    trong cùng (publication_year, subfield)
    valid_articles = df[df["fwci"].notna() & df["citation_percentile"].notna()].copy()

    # 5.1. Ước lượng kỳ vọng trích dẫn của ngành (expected_cites = cited_by_count / fwci)
    valid_pos = valid_articles[valid_articles["fwci"] > 0].copy()
    valid_pos["expected_cites"] = valid_pos["cited_by_count"] / valid_pos["fwci"]

    exp_by_yr_sub = valid_pos.groupby(["publication_year", "subfield"])["expected_cites"].median().to_dict()
    exp_by_yr = valid_pos.groupby("publication_year")["expected_cites"].median().to_dict()
    global_exp_median = valid_pos["expected_cites"].median()

    def compute_missing_fwci(row):
        cites = float(row["cited_by_count"])
        if cites <= 0:
            return 0.0
        yr = int(row["publication_year"])
        sub = row["subfield"]
        exp_c = exp_by_yr_sub.get((yr, sub), exp_by_yr.get(yr, global_exp_median))
        return round(cites / max(exp_c, 0.5), 4)

    mask_fwci_na = df["fwci"].isna()
    df.loc[mask_fwci_na, "fwci"] = df.loc[mask_fwci_na].apply(compute_missing_fwci, axis=1)

    # 5.2. Nội suy phân vị trích dẫn thực nghiệm (citation_percentile) từ cited_by_count
    ref_groups = {}
    for (yr, sub), grp in valid_articles.groupby(["publication_year", "subfield"]):
        agg_grp = (
            grp.groupby("cited_by_count", as_index=False)["citation_percentile"]
            .median()
            .sort_values("cited_by_count")
        )
        ref_groups[(yr, sub)] = (
            agg_grp["cited_by_count"].values,
            agg_grp["citation_percentile"].values,
        )

    ref_year_groups = {}
    for yr, grp in valid_articles.groupby("publication_year"):
        agg_grp = (
            grp.groupby("cited_by_count", as_index=False)["citation_percentile"]
            .median()
            .sort_values("cited_by_count")
        )
        ref_year_groups[yr] = (
            agg_grp["cited_by_count"].values,
            agg_grp["citation_percentile"].values,
        )

    def compute_missing_percentile(row):
        cites = float(row["cited_by_count"])
        yr = int(row["publication_year"])
        sub = row["subfield"]
        if (yr, sub) in ref_groups and len(ref_groups[(yr, sub)][0]) >= 3:
            x_cites, y_pct = ref_groups[(yr, sub)]
        else:
            x_cites, y_pct = ref_year_groups[yr]
        return round(float(np.interp(cites, x_cites, y_pct)), 6)

    mask_pct_na = df["citation_percentile"].isna()
    df.loc[mask_pct_na, "citation_percentile"] = df.loc[mask_pct_na].apply(compute_missing_percentile, axis=1)
-
    # 6. KIỂM ĐỊNH BẢO TOÀN DỮ LIỆU GỐC (100% NON-NAN INVARIANCE CHECK)
    for col in df.columns:
        not_na_mask = df_raw[col].notna()
        if pd.api.types.is_float_dtype(df_raw[col]):
            assert np.allclose(
                df.loc[not_na_mask, col], df_raw.loc[not_na_mask, col], rtol=0, atol=1e-12
            ), f"Phát hiện thay đổi dữ liệu gốc ở cột {col}!"
        else:
            assert (
                df.loc[not_na_mask, col] == df_raw.loc[not_na_mask, col]
            ).all(), f"Phát hiện thay đổi dữ liệu gốc ở cột {col}!"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã xuất file chuẩn hóa: '{out_csv}' ({len(df):,} dòng × {df.shape[1]} cột)")
    print("🛡️ Kiểm định bảo toàn dữ liệu gốc: 100% các ô đã có sẵn dữ liệu được giữ nguyên tuyệt đối!")
    print(f"🧹 Tổng số ô NaN còn lại trong toàn bộ bảng: {df.isna().sum().sum()}")
    return df


if __name__ == "__main__":
    clean_only_missing_values()
