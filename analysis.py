"""
鉱工業指数（IIP）高度分析スクリプト
- 基本統計
- トレンド分析（コロナ前後比較）
- セクター別変化率ランキング
- 相関分析（生産・出荷・在庫）
- 四半期動態分析
- 在庫率と出荷の逆相関分析
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# 日本語フォント設定
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "csv_output"
FIG_DIR = "figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)

# =====================================================================
# 1. データ読み込み
# =====================================================================
prod = pd.read_csv(f"{OUTPUT_DIR}/production.csv", encoding="utf-8-sig")
ship = pd.read_csv(f"{OUTPUT_DIR}/shipment.csv", encoding="utf-8-sig")
inv  = pd.read_csv(f"{OUTPUT_DIR}/inventory_eop.csv", encoding="utf-8-sig")
invr = pd.read_csv(f"{OUTPUT_DIR}/inventory_ratio.csv", encoding="utf-8-sig")
combined = pd.read_csv(f"{OUTPUT_DIR}/combined_long.csv", encoding="utf-8-sig")
combined_q = pd.read_csv(f"{OUTPUT_DIR}/combined_quarterly_long.csv", encoding="utf-8-sig")

CY = ["2018CY","2019CY","2020CY","2021CY","2022CY","2023CY","2024CY","2025CY"]
QS  = [f"{y}Q{q}" for y in range(2018,2026) for q in range(1,5)]

print("=" * 70)
print("  鉱工業指数（IIP）高度分析レポート  2020年基準")
print("=" * 70)

# =====================================================================
# 2. 基本統計
# =====================================================================
print("\n【1. 基本統計】（生産指数 CY 年次 全品目）")
cy_prod = prod.set_index("品目名称")[CY].dropna(how="all")
desc = cy_prod.describe().T
print(desc.to_string())

desc.to_csv(f"{OUTPUT_DIR}/basic_stats_production_cy.csv", encoding="utf-8-sig")

# =====================================================================
# 3. コロナ前後比較（2019→2020 落ち込み vs 2021 回復）
# =====================================================================
print("\n【2. コロナ前後変化率（全品目 生産指数）】")
covid = prod[["品目名称","2019CY","2020CY","2021CY"]].dropna()
covid = covid[covid["2019CY"] > 0].copy()
covid["drop_pct"]    = (covid["2020CY"] - covid["2019CY"]) / covid["2019CY"] * 100
covid["recovery_pct"] = (covid["2021CY"] - covid["2020CY"]) / covid["2020CY"] * 100
covid["2019to2021_pct"] = (covid["2021CY"] - covid["2019CY"]) / covid["2019CY"] * 100

print("\nコロナ落ち込みワースト10（2019→2020 生産 変化率%）:")
worst = covid.nsmallest(10, "drop_pct")[["品目名称","2019CY","2020CY","drop_pct"]]
print(worst.to_string(index=False))

print("\n回復率ベスト10（2020→2021 生産 変化率%）:")
best_rec = covid.nlargest(10, "recovery_pct")[["品目名称","2020CY","2021CY","recovery_pct"]]
print(best_rec.to_string(index=False))

print("\n2019→2021 純変化ワースト10（まだ戻っていない品目）:")
still_down = covid.nsmallest(10, "2019to2021_pct")[["品目名称","2019CY","2021CY","2019to2021_pct"]]
print(still_down.to_string(index=False))

covid.to_csv(f"{OUTPUT_DIR}/covid_impact_analysis.csv", index=False, encoding="utf-8-sig")

# =====================================================================
# 4. 最新トレンド分析（2022→2024 変化率）
# =====================================================================
print("\n【3. 最新トレンド（2022→2024 生産指数）】")
trend = prod[["品目名称","2022CY","2023CY","2024CY"]].dropna()
trend = trend[trend["2022CY"] > 0].copy()
trend["chg_22_24"] = (trend["2024CY"] - trend["2022CY"]) / trend["2022CY"] * 100
trend["chg_23_24"] = (trend["2024CY"] - trend["2023CY"]) / trend["2023CY"] * 100

print("\n2022→2024 生産増加ベスト10:")
print(trend.nlargest(10, "chg_22_24")[["品目名称","2022CY","2024CY","chg_22_24"]].to_string(index=False))

print("\n2022→2024 生産減少ワースト10:")
print(trend.nsmallest(10, "chg_22_24")[["品目名称","2022CY","2024CY","chg_22_24"]].to_string(index=False))

# =====================================================================
# 5. 生産・出荷・在庫率の相関分析（鉱工業全体 四半期）
# =====================================================================
print("\n【4. 生産・出荷・在庫率 相関分析（鉱工業全体 四半期）】")
def get_agg_ts(df, cols, item_name="鉱工業"):
    """指定品目の時系列を取得"""
    row = df[df["品目名称"] == item_name]
    if row.empty:
        return None
    available = [c for c in cols if c in df.columns]
    return row[available].values.flatten()

prod_q_ts  = np.array(get_agg_ts(prod, QS), dtype=float)
ship_q_ts  = np.array(get_agg_ts(ship, QS), dtype=float)
invr_q_ts  = np.array(get_agg_ts(invr, QS), dtype=float)

valid_mask = (~np.isnan(prod_q_ts)) & (~np.isnan(ship_q_ts)) & (~np.isnan(invr_q_ts))
p_v = prod_q_ts[valid_mask]
s_v = ship_q_ts[valid_mask]
ir_v = invr_q_ts[valid_mask]

r_ps, pval_ps = stats.pearsonr(p_v, s_v)
r_pir, pval_pir = stats.pearsonr(p_v, ir_v)
r_sir, pval_sir = stats.pearsonr(s_v, ir_v)

print(f"  生産 × 出荷      : r = {r_ps:.4f}  (p={pval_ps:.4f})")
print(f"  生産 × 在庫率    : r = {r_pir:.4f}  (p={pval_pir:.4f})")
print(f"  出荷 × 在庫率    : r = {r_sir:.4f}  (p={pval_sir:.4f})")

# =====================================================================
# 6. セクター別 YoY成長率（年次）
# =====================================================================
print("\n【5. セクター別 生産指数 年次 YoY変化】")
# 大分類（品目番号が10桁の主要グループ）を抽出
major = prod[prod["品目番号"].astype(str).str.endswith("00000000")].copy()
major_cy = major.set_index("品目名称")[CY].copy()

yoy = major_cy.pct_change(axis=1) * 100
yoy.columns = [f"{c}_YoY" for c in CY]
print(yoy.to_string())

yoy.to_csv(f"{OUTPUT_DIR}/sector_yoy_production.csv", encoding="utf-8-sig")

# =====================================================================
# 7. 在庫率高止まりセクター（2024年水準）
# =====================================================================
print("\n【6. 在庫率高止まりセクター（2024CY, 2020=100基準）】")
ir_2024 = invr[["品目名称","2020CY","2024CY"]].copy()
ir_2024["2020CY"] = pd.to_numeric(ir_2024["2020CY"], errors="coerce")
ir_2024["2024CY"] = pd.to_numeric(ir_2024["2024CY"], errors="coerce")
ir_2024 = ir_2024.dropna()
ir_2024 = ir_2024[ir_2024["2020CY"] > 0].copy()
ir_2024["level_vs_2020"] = ir_2024["2024CY"] - 100  # 2020=100からの乖離
print("在庫率 2024年が基準値(100)より高い品目 TOP10:")
high_ir = ir_2024.nlargest(10, "level_vs_2020")[["品目名称","2024CY","level_vs_2020"]]
print(high_ir.to_string(index=False))

ir_2024.to_csv(f"{OUTPUT_DIR}/inventory_ratio_2024.csv", index=False, encoding="utf-8-sig")

# =====================================================================
# 8. 線形トレンド回帰（全品目 生産指数 四半期）
# =====================================================================
print("\n【7. 線形トレンド回帰（生産指数 四半期 全品目 2018Q1-2024Q4）】")
q_cols_main = [c for c in QS if c <= "2024Q4" and c in prod.columns]
trend_stats = []
for _, row in prod.iterrows():
    vals = row[q_cols_main].values.astype(float)
    mask = ~np.isnan(vals)
    if mask.sum() < 8:
        continue
    x = np.arange(len(vals))[mask]
    y = vals[mask]
    slope, intercept, r, p, se = stats.linregress(x, y)
    trend_stats.append({
        "品目名称": row["品目名称"],
        "slope": round(slope, 4),
        "r_squared": round(r**2, 4),
        "p_value": round(p, 4),
        "trend": "上昇" if slope > 0 else "下降"
    })

trend_df = pd.DataFrame(trend_stats)
print("上昇トレンド品目（slope上位10）:")
print(trend_df.nlargest(10, "slope")[["品目名称","slope","r_squared","trend"]].to_string(index=False))
print("\n下降トレンド品目（slope下位10）:")
print(trend_df.nsmallest(10, "slope")[["品目名称","slope","r_squared","trend"]].to_string(index=False))

trend_df.to_csv(f"{OUTPUT_DIR}/trend_regression_production.csv", index=False, encoding="utf-8-sig")

# =====================================================================
# 9. グラフ作成
# =====================================================================
print("\n【8. グラフ作成中...】")

# --- (A) 鉱工業全体 四半期時系列（生産・出荷・在庫率）
fig, ax1 = plt.subplots(figsize=(14, 6))
q_labels = [c for c in QS if c in prod.columns and not np.isnan(get_agg_ts(prod, [c])[0])]

def safe_ts(df, cols, name="鉱工業"):
    r = df[df["品目名称"] == name]
    return [r[c].values[0] if c in df.columns and not r[c].isna().all() else np.nan for c in cols]

p_vals = safe_ts(prod, q_labels)
s_vals = safe_ts(ship, q_labels)
ir_vals = safe_ts(invr, q_labels)

x = np.arange(len(q_labels))
ax1.plot(x, p_vals, "b-o", markersize=3, label="Production Index", linewidth=1.5)
ax1.plot(x, s_vals, "g-s", markersize=3, label="Shipment Index", linewidth=1.5)
ax1.set_ylabel("Index (2020=100)", color="black")
ax1.set_ylim(70, 130)

ax2 = ax1.twinx()
ax2.plot(x, ir_vals, "r--^", markersize=3, label="Inventory Ratio", linewidth=1.5, alpha=0.8)
ax2.set_ylabel("Inventory Ratio Index", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# X軸ラベル（年Q1のみ表示）
tick_pos = [i for i, lbl in enumerate(q_labels) if "Q1" in lbl]
tick_lbl = [lbl[:4] for lbl in q_labels if "Q1" in lbl]
ax1.set_xticks(tick_pos)
ax1.set_xticklabels(tick_lbl)

ax1.axvline(x=q_labels.index("2020Q1"), color="orange", linestyle=":", alpha=0.7, label="COVID-19 (2020Q1)")
ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.4)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
ax1.set_title("Japan IIP: Production, Shipment & Inventory Ratio (Quarterly, 2018-2025)", fontsize=12)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_iip_quarterly_trend.png", dpi=120)
plt.close()
print(f"  Saved: {FIG_DIR}/01_iip_quarterly_trend.png")

# --- (B) セクター別 2024CY 生産指数（基準100との乖離）
major_names_all = prod[prod["品目番号"].astype(str).str.endswith("00000000")]["品目名称"].tolist()
# 鉱工業・製造工業を除いたサブセクターのみ
sub_sectors = [n for n in major_names_all if n not in ["鉱工業","製造工業"]][:20]
sub_data = prod[prod["品目名称"].isin(sub_sectors)].set_index("品目名称")

def shorten(name, n=15):
    return name[:n] + "..." if len(name) > n else name

if "2024CY" in sub_data.columns:
    vals_24 = sub_data["2024CY"].dropna()
    labels_en = [shorten(l) for l in vals_24.index]
    colors = ["#d73027" if v < 100 else "#1a9850" for v in vals_24.values]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(labels_en, vals_24.values - 100, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Deviation from 2020 Base (index pts)", fontsize=10)
    ax.set_title("IIP Production: Sector Deviation from Base (2024CY, 2020=100)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    red_patch = mpatches.Patch(color="#d73027", label="Below base (2020=100)")
    green_patch = mpatches.Patch(color="#1a9850", label="Above base (2020=100)")
    ax.legend(handles=[green_patch, red_patch], fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_sector_deviation_2024.png", dpi=120)
    plt.close()
    print(f"  Saved: {FIG_DIR}/02_sector_deviation_2024.png")

# --- (C) コロナ落ち込み vs 回復散布図
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(
    covid["drop_pct"], covid["recovery_pct"],
    c=covid["2019to2021_pct"], cmap="RdYlGn",
    alpha=0.7, s=40, edgecolors="none"
)
plt.colorbar(sc, ax=ax, label="2019→2021 net change (%)")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("2019→2020 Change (%) [COVID Drop]", fontsize=10)
ax.set_ylabel("2020→2021 Change (%) [Recovery]", fontsize=10)
ax.set_title("IIP Production: COVID Impact vs Recovery by Item", fontsize=11)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_covid_drop_vs_recovery.png", dpi=120)
plt.close()
print(f"  Saved: {FIG_DIR}/03_covid_drop_vs_recovery.png")

# --- (D) 生産 vs 出荷 散布（鉱工業全体 四半期）
fig, ax = plt.subplots(figsize=(8, 6))
colors_q = plt.cm.viridis(np.linspace(0, 1, len(p_vals)))
ax.scatter(s_v, p_v, c=np.arange(len(p_v)), cmap="viridis", alpha=0.8, s=50)
m, b = np.polyfit(s_v, p_v, 1)
xs = np.linspace(s_v.min(), s_v.max(), 100)
ax.plot(xs, m*xs+b, "r--", label=f"Linear fit (r={r_ps:.3f})")
ax.set_xlabel("Shipment Index", fontsize=10)
ax.set_ylabel("Production Index", fontsize=10)
ax.set_title(f"Production vs Shipment (Quarterly, r={r_ps:.3f})", fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_production_vs_shipment.png", dpi=120)
plt.close()
print(f"  Saved: {FIG_DIR}/04_production_vs_shipment.png")

# --- (E) YoY変化率ヒートマップ（主要セクター × 年）
yoy_plot = yoy.copy()
yoy_plot.columns = [c.replace("_YoY","") for c in yoy_plot.columns]
yoy_plot = yoy_plot.iloc[:, 1:]  # 最初のNaN列除去

fig, ax = plt.subplots(figsize=(12, max(4, len(yoy_plot)*0.45 + 1.5)))
im = ax.imshow(yoy_plot.values, cmap="RdYlGn", aspect="auto",
               vmin=-15, vmax=15)
plt.colorbar(im, ax=ax, label="YoY Change (%)")
ax.set_xticks(range(len(yoy_plot.columns)))
ax.set_xticklabels(yoy_plot.columns, rotation=45, fontsize=9)
ax.set_yticks(range(len(yoy_plot.index)))
ax.set_yticklabels([shorten(n) for n in yoy_plot.index], fontsize=8)
# 値をセルに記入
for i in range(len(yoy_plot.index)):
    for j in range(len(yoy_plot.columns)):
        val = yoy_plot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6.5,
                    color="black" if abs(val) < 10 else "white")
ax.set_title("IIP Production: YoY Change % Heatmap by Major Sector", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_yoy_heatmap.png", dpi=120)
plt.close()
print(f"  Saved: {FIG_DIR}/05_yoy_heatmap.png")

# =====================================================================
# 10. サマリーレポート出力
# =====================================================================
print("\n" + "=" * 70)
print("  分析サマリー")
print("=" * 70)

# 鉱工業全体 最新値
row_iip = prod[prod["品目名称"] == "鉱工業"]
for col in ["2022CY","2023CY","2024CY","2025CY"]:
    if col in prod.columns:
        val = row_iip[col].values[0]
        print(f"  鉱工業 生産指数 {col}: {val}")

print(f"\n  分析対象品目数: {len(prod)} 品目")
print(f"  期間: 2018年～2025年（四半期）")
print(f"\n  出力ファイル:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"    csv_output/{f}")
print()
for f in sorted(os.listdir(FIG_DIR)):
    print(f"    figures/{f}")

print("\n分析完了。")
