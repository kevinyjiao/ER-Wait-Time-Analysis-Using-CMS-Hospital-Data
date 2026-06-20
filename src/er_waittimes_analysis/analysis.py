from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

ED_MEASURE_PRIORITY = [
    "OP_18b",
    "OP_18B",
    "OP_18c",
    "OP_18C",
    "ED_1b",
    "ED_1B",
    "ED_2b",
    "ED_2B",
]


@dataclass
class AnalysisResult:
    dataset: pd.DataFrame
    measure_name: str
    wait_column: str
    size_note: str
    urban_rural_note: str
    tables: dict[str, pd.DataFrame]


def normalize_col(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("\ufeff", "")
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
    )


def find_column(columns: Iterable[str], candidates: Iterable[str], required: bool = True) -> str | None:
    lookup = {normalize_col(col): col for col in columns}
    for candidate in candidates:
        key = normalize_col(candidate)
        if key in lookup:
            return lookup[key]
    for candidate in candidates:
        key = normalize_col(candidate)
        for norm, original in lookup.items():
            if key in norm:
                return original
    if required:
        raise ValueError(f"Could not find any of these columns: {', '.join(candidates)}")
    return None


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, dtype=str, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, low_memory=False)


def discover_files(raw_dir: Path) -> tuple[Path, Path]:
    csvs = sorted(raw_dir.glob("*.csv"))
    if len(csvs) < 2:
        raise FileNotFoundError(
            f"Expected at least two CSV files in {raw_dir}. "
            "Put the CMS wait time file and hospital general information file there, "
            "or pass --wait-file and --info-file."
        )

    scored = []
    for path in csvs:
        sample = read_csv(path).head(25)
        cols = " ".join(normalize_col(c) for c in sample.columns)
        wait_score = sum(token in cols for token in ["measure id", "measure name", "score", "condition"])
        info_score = sum(token in cols for token in ["hospital type", "hospital ownership", "emergency services"])
        scored.append((path, wait_score, info_score))

    wait_file = max(scored, key=lambda item: item[1])[0]
    info_file = max((item for item in scored if item[0] != wait_file), key=lambda item: item[2])[0]
    return wait_file, info_file


def clean_score(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+\.?\d*)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def filter_ed_measure(wait_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    measure_id_col = find_column(wait_df.columns, ["Measure ID"], required=False)
    measure_name_col = find_column(wait_df.columns, ["Measure Name"], required=False)
    condition_col = find_column(wait_df.columns, ["Condition"], required=False)

    if measure_id_col:
        ids = wait_df[measure_id_col].astype(str).str.upper()
        for measure_id in ED_MEASURE_PRIORITY:
            match = wait_df.loc[ids == measure_id.upper()].copy()
            if not match.empty:
                name = (
                    match[measure_name_col].dropna().iloc[0]
                    if measure_name_col and match[measure_name_col].notna().any()
                    else measure_id
                )
                return match, str(name)

    text_parts = []
    for col in [measure_id_col, measure_name_col, condition_col]:
        if col:
            text_parts.append(wait_df[col].astype(str))
    if not text_parts:
        raise ValueError("Could not identify emergency department measures because no measure columns were found.")

    combined = text_parts[0]
    for part in text_parts[1:]:
        combined = combined + " " + part
    ed_mask = combined.str.contains(
        r"emergency|ed |ed-|er |OP_18|ED_1|ED_2|median time|left before being seen",
        case=False,
        regex=True,
        na=False,
    )
    filtered = wait_df.loc[ed_mask].copy()
    if filtered.empty:
        raise ValueError("No emergency department wait time rows were found.")

    if measure_id_col and measure_name_col:
        counts = filtered.groupby([measure_id_col, measure_name_col], dropna=False).size().sort_values(ascending=False)
        measure_id, measure_name = counts.index[0]
        filtered = filtered.loc[filtered[measure_id_col] == measure_id].copy()
        return filtered, str(measure_name)
    if measure_name_col:
        measure_name = filtered[measure_name_col].mode().iloc[0]
        return filtered.loc[filtered[measure_name_col] == measure_name].copy(), str(measure_name)
    return filtered, "Emergency department wait time measure"


def classify_size(df: pd.DataFrame) -> tuple[pd.Series, str]:
    bed_col = find_column(
        df.columns,
        ["Hospital Beds", "Total Beds", "Bed Count", "Number of Beds", "Beds"],
        required=False,
    )
    if bed_col:
        beds = clean_score(df[bed_col])
        labels = pd.cut(
            beds,
            bins=[-np.inf, 99, 299, np.inf],
            labels=["Small (<100 beds)", "Medium (100-299 beds)", "Large (300+ beds)"],
        )
        return labels.astype("object").fillna("Unknown"), f"Size based on `{bed_col}`."

    sample_col = find_column(df.columns, ["Sample", "Number of Patients", "Denominator"], required=False)
    if sample_col:
        sample = clean_score(df[sample_col])
        valid = sample.dropna()
        if valid.nunique() >= 3:
            buckets = pd.qcut(sample.rank(method="first"), q=3, labels=["Small volume", "Medium volume", "Large volume"])
            return buckets.astype("object").fillna("Unknown"), f"Size uses `{sample_col}` as a patient-volume proxy."

    return pd.Series("Unknown", index=df.index), "No bed-count or sample-volume column was available."


def classify_urban_rural(df: pd.DataFrame) -> tuple[pd.Series, str]:
    urban_col = find_column(
        df.columns,
        ["Urban/Rural", "Urban Rural", "Rural Urban", "CBSA Type", "Rural Versus Urban"],
        required=False,
    )
    if urban_col:
        status = df[urban_col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
        return status.fillna("Unknown"), f"Urban/rural status based on `{urban_col}`."

    type_col = find_column(df.columns, ["Hospital Type"], required=False)
    if type_col:
        status = np.where(
            df[type_col].astype(str).str.contains("critical access", case=False, na=False),
            "Rural proxy: Critical Access",
            "Urban/other proxy",
        )
        return pd.Series(status, index=df.index), "Urban/rural status uses hospital type as a proxy."

    return pd.Series("Unknown", index=df.index), "No urban/rural field or hospital type proxy was available."


def prepare_dataset(wait_file: Path, info_file: Path) -> AnalysisResult:
    wait_df = read_csv(wait_file)
    info_df = read_csv(info_file)

    facility_wait = find_column(wait_df.columns, ["Facility ID", "Provider ID", "CMS Certification Number", "CCN"])
    facility_info = find_column(info_df.columns, ["Facility ID", "Provider ID", "CMS Certification Number", "CCN"])
    score_col = find_column(wait_df.columns, ["Score", "Measure Value", "Value"])

    ed_df, measure_name = filter_ed_measure(wait_df)
    ed_df["wait_minutes"] = clean_score(ed_df[score_col])
    ed_df = ed_df.loc[ed_df["wait_minutes"].between(1, 1440, inclusive="both")].copy()

    merged = ed_df.merge(
        info_df,
        left_on=facility_wait,
        right_on=facility_info,
        how="left",
        suffixes=("", "_info"),
    )

    state_col = find_column(merged.columns, ["State"])
    name_col = find_column(merged.columns, ["Facility Name", "Hospital Name"], required=False)
    type_col = find_column(merged.columns, ["Hospital Type"], required=False)
    ownership_col = find_column(merged.columns, ["Hospital Ownership", "Ownership"], required=False)

    merged["state"] = merged[state_col].astype(str).str.upper().str.strip()
    merged["hospital_name"] = merged[name_col].astype(str).str.title() if name_col else "Unknown"
    merged["hospital_type"] = merged[type_col].astype(str).str.strip() if type_col else "Unknown"
    merged["ownership"] = merged[ownership_col].astype(str).str.strip() if ownership_col else "Unknown"
    merged["size_group"], size_note = classify_size(merged)
    merged["urban_rural"], urban_rural_note = classify_urban_rural(merged)

    keep = [
        "hospital_name",
        "state",
        "wait_minutes",
        "hospital_type",
        "ownership",
        "size_group",
        "urban_rural",
    ]
    dataset = merged[keep].dropna(subset=["state", "wait_minutes"]).copy()
    dataset = dataset.loc[dataset["state"].str.len().between(2, 3)]

    tables = build_summary_tables(dataset)
    return AnalysisResult(dataset, measure_name, "wait_minutes", size_note, urban_rural_note, tables)


def summarize_group(df: pd.DataFrame, column: str, min_hospitals: int = 5) -> pd.DataFrame:
    summary = (
        df.groupby(column, dropna=False)
        .agg(
            avg_wait_minutes=("wait_minutes", "mean"),
            median_wait_minutes=("wait_minutes", "median"),
            hospital_count=("wait_minutes", "size"),
        )
        .reset_index()
        .sort_values("avg_wait_minutes", ascending=False)
    )
    return summary.loc[summary["hospital_count"] >= min_hospitals].copy()


def build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "states": summarize_group(df, "state", min_hospitals=5),
        "hospital_type": summarize_group(df, "hospital_type", min_hospitals=5),
        "ownership": summarize_group(df, "ownership", min_hospitals=5),
        "size": summarize_group(df, "size_group", min_hospitals=5),
        "urban_rural": summarize_group(df, "urban_rural", min_hospitals=5),
    }


def save_tables(result: AnalysisResult) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.dataset.to_csv(PROCESSED_DIR / "analysis_dataset.csv", index=False)
    with pd.ExcelWriter(PROCESSED_DIR / "summary_tables.xlsx") as writer:
        for sheet, table in result.tables.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)


def chart_bar(
    data: pd.DataFrame,
    x_col: str,
    title: str,
    filename: str,
    top_n: int | None = None,
    color: str = "#2474A6",
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_data = data.copy()
    if top_n:
        plot_data = plot_data.head(top_n)
    plot_data = plot_data.sort_values("avg_wait_minutes", ascending=True)

    height = max(4.2, 0.35 * len(plot_data) + 1.8)
    fig, ax = plt.subplots(figsize=(9, height))
    sns.barplot(data=plot_data, x="avg_wait_minutes", y=x_col, color=color, ax=ax)
    ax.set_title(title, loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Average wait time (minutes)")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{value:.0f}" for value in container.datavalues], padding=4, fontsize=9)
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_charts(result: AnalysisResult) -> dict[str, Path]:
    charts = {
        "states": chart_bar(
            result.tables["states"],
            "state",
            "States with the longest average ER wait times",
            "state_wait_times.png",
            top_n=15,
            color="#D05A3F",
        ),
        "ownership": chart_bar(
            result.tables["ownership"],
            "ownership",
            "Average ER wait time by hospital ownership",
            "ownership_wait_times.png",
            color="#2474A6",
        ),
        "type": chart_bar(
            result.tables["hospital_type"],
            "hospital_type",
            "Average ER wait time by hospital type",
            "type_wait_times.png",
            color="#2A8C62",
        ),
        "urban_rural": chart_bar(
            result.tables["urban_rural"],
            "urban_rural",
            "Average ER wait time: urban vs rural",
            "urban_rural_wait_times.png",
            color="#7A5AA6",
        ),
    }

    size_table = result.tables["size"]
    if not size_table.empty:
        charts["size"] = chart_bar(
            size_table,
            "size_group",
            "Average ER wait time by hospital size",
            "size_wait_times.png",
            color="#B48725",
        )
    return charts


def run(wait_file: Path | None = None, info_file: Path | None = None) -> AnalysisResult:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if wait_file is None or info_file is None:
        wait_file, info_file = discover_files(RAW_DIR)

    result = prepare_dataset(Path(wait_file), Path(info_file))
    save_tables(result)
    charts = make_charts(result)

    print(f"Analyzed measure: {result.measure_name}")
    print(f"Hospitals in analysis: {len(result.dataset):,}")
    print(f"Saved cleaned dataset: {PROCESSED_DIR / 'analysis_dataset.csv'}")
    print(f"Saved summary tables: {PROCESSED_DIR / 'summary_tables.xlsx'}")
    print("Saved charts:")
    for chart_path in charts.values():
        print(f"- {chart_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CMS hospital emergency department wait times.")
    parser.add_argument("--wait-file", type=Path, default=None, help="Path to the CMS wait time CSV.")
    parser.add_argument("--info-file", type=Path, default=None, help="Path to the CMS hospital general information CSV.")
    args = parser.parse_args()
    try:
        run(args.wait_file, args.info_file)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
