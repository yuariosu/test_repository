"""
鉱工業指数（IIP）Excelファイル → CSV変換スクリプト
各シートを個別CSVとして保存し、分析用の結合データも作成する
"""
import pandas as pd
import os

EXCEL_FILE = "b2020_goq1j.xlsx"
OUTPUT_DIR = "csv_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# シート名と対応するウエイト列名のマッピング
SHEET_INFO = {
    "生産計": {"metric": "production", "weight_col": "付加生産ウエイト"},
    "出荷計": {"metric": "shipment",   "weight_col": "出荷ウエイト"},
    "在庫計（期末）": {"metric": "inventory_eop", "weight_col": "在庫ウエイト"},
    "在庫計（平均）": {"metric": "inventory_avg", "weight_col": "在庫ウエイト"},
    "在庫率計":  {"metric": "inventory_ratio", "weight_col": "在庫率ウエイト"},
}

# 時系列コード行（row index 1, 0-indexed）→ 列ラベル対応を構築するため
# 実際のヘッダーは row 2 (0-indexed), データは row 3 以降
dfs = {}

for sheet_name, info in SHEET_INFO.items():
    # header=2 → 3行目をヘッダーとして読み込む
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, header=2)

    # 最初の3列: 品目番号, 品目名称, ウエイト
    # 空列（None列）を除去
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # NaN行を除去
    df = df.dropna(subset=["品目番号"])
    df["品目番号"] = df["品目番号"].astype(int)

    metric = info["metric"]
    dfs[metric] = df

    # 個別CSV保存
    out_path = os.path.join(OUTPUT_DIR, f"{metric}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}  ({len(df)} rows x {len(df.columns)} cols)")

# ---- 分析用: 年次データ（CY）のみ抽出して結合 ----
CY_COLS = ["品目番号", "品目名称",
           "2018CY", "2019CY", "2020CY", "2021CY", "2022CY", "2023CY", "2024CY", "2025CY"]

combined_frames = []
for metric, df in dfs.items():
    available = [c for c in CY_COLS if c in df.columns]
    sub = df[available].copy()
    # wide → long
    id_cols = ["品目番号", "品目名称"]
    val_cols = [c for c in available if c not in id_cols]
    long = sub.melt(id_vars=id_cols, value_vars=val_cols, var_name="year_label", value_name="index_value")
    long["year"] = long["year_label"].str[:4].astype(int)
    long["metric"] = metric
    combined_frames.append(long)

combined = pd.concat(combined_frames, ignore_index=True)
combined_path = os.path.join(OUTPUT_DIR, "combined_long.csv")
combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
print(f"\nSaved combined long-format: {combined_path}  ({len(combined)} rows)")

# ---- 分析用: 四半期データ（Q）のみ抽出して結合 ----
Q_COLS = ["品目番号", "品目名称"] + \
    [f"{y}Q{q}" for y in range(2018, 2026) for q in range(1, 5)]

combined_q_frames = []
for metric, df in dfs.items():
    available = [c for c in Q_COLS if c in df.columns]
    sub = df[available].copy()
    id_cols = ["品目番号", "品目名称"]
    val_cols = [c for c in available if c not in id_cols]
    long = sub.melt(id_vars=id_cols, value_vars=val_cols, var_name="quarter_label", value_name="index_value")
    long["year"] = long["quarter_label"].str[:4].astype(int)
    long["quarter"] = long["quarter_label"].str[-1].astype(int)
    long["metric"] = metric
    combined_q_frames.append(long)

combined_q = pd.concat(combined_q_frames, ignore_index=True)
combined_q_path = os.path.join(OUTPUT_DIR, "combined_quarterly_long.csv")
combined_q.to_csv(combined_q_path, index=False, encoding="utf-8-sig")
print(f"Saved quarterly long-format: {combined_q_path}  ({len(combined_q)} rows)")

print("\n変換完了。csv_output/ ディレクトリに保存されました。")
