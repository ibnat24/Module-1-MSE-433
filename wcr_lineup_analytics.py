"""
Wheelchair Rugby Lineup Analytics (MSE433 Module 1 → Phase 2-ready)

What this script does:

PHASE 1 (Descriptive)
(1) Lineup effectiveness:
    - minutes, goals_for, goals_against, net_goals, net_per_min
    - exports: outputs/lineup_effectiveness_by_team.csv
               outputs/lineup_most_used_by_team.csv
               outputs/lineup_best_by_team.csv

(2) Physical rating vs scoring (two proxies):
    (2A) Lineup-level: avg lineup rating vs goals/min + weighted correlation + binned summary
        - exports: outputs/rating_scoring_lineup_level.csv
    (2B) Player-attribution proxy: equal-split goals among 4 on-court players
        - exports: outputs/rating_scoring_player_attrib.csv

PHASE 2 INPUTS (Predictive / Player-value metrics)
(3) Player value metrics:
    - RAPM (ridge regression on net goals/min, adjusted for teammates/opponents)
    - On/Off impact (contextual net_per_min difference)
    - Synergy residual (actual - predicted based on RAPM, minutes-weighted)
    - Avg lineup net/min when on court
    - exports: outputs/player_value_metrics.csv

Plots (saved to outputs/plots):
    - scoring_vs_lineup_avg_rating.png
    - attrib_scoring_vs_rating.png
    - top10_lineups_net_per_min.png

Expected input files:
    data/stint_data.csv
    data/player_data.csv
"""

import os
import numpy as np
import pandas as pd

# -----------------------------
# Paths / folders
# -----------------------------
STINT_PATH = "data/stint_data.csv"
PLAYER_PATH = "data/player_data.csv"

OUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RUN_PLOTS = True
try:
    import matplotlib.pyplot as plt
except Exception:
    RUN_PLOTS = False

# -----------------------------
# Load
# -----------------------------
st = pd.read_csv(STINT_PATH)
pl = pd.read_csv(PLAYER_PATH)

# Validate columns
required_stint_cols = {"game_id", "h_team", "a_team", "minutes", "h_goals", "a_goals"}
if not required_stint_cols.issubset(st.columns):
    missing = sorted(list(required_stint_cols - set(st.columns)))
    raise ValueError(f"stint_data.csv missing required columns: {missing}")

if not {"player", "rating"}.issubset(pl.columns):
    raise ValueError("player_data.csv must have columns: player, rating")

home_cols = [c for c in st.columns if c.startswith("home")]
away_cols = [c for c in st.columns if c.startswith("away")]
if len(home_cols) != 4 or len(away_cols) != 4:
    raise ValueError(
        f"Expected 4 home cols and 4 away cols (home1..home4, away1..away4). "
        f"Found {len(home_cols)} home and {len(away_cols)} away."
    )

# Ensure numeric + clean
st["minutes"] = pd.to_numeric(st["minutes"], errors="coerce")
st["h_goals"] = pd.to_numeric(st["h_goals"], errors="coerce")
st["a_goals"] = pd.to_numeric(st["a_goals"], errors="coerce")
st = st.dropna(subset=["minutes", "h_goals", "a_goals"])
st = st[st["minutes"] > 0].copy()

# Canonicalize player ids to string
for c in home_cols + away_cols:
    st[c] = st[c].astype(str)
pl["player"] = pl["player"].astype(str)

# Player -> rating map
rating_map = dict(zip(pl["player"], pl["rating"]))

# -----------------------------
# Helper: canonical lineup key
# -----------------------------
def canonical_lineup(players_4):
    """Return a stable tuple for a 4-player lineup."""
    vals = [p for p in players_4 if pd.notna(p)]
    vals = [str(v) for v in vals]
    return tuple(sorted(vals))

# ============================================================
# (1) LINEUP EFFECTIVENESS
# ============================================================
st["h_lineup"] = st[home_cols].apply(lambda r: canonical_lineup(r.values), axis=1)
st["a_lineup"] = st[away_cols].apply(lambda r: canonical_lineup(r.values), axis=1)

home_view = pd.DataFrame(
    {
        "team": st["h_team"],
        "opp": st["a_team"],
        "lineup": st["h_lineup"],
        "minutes": st["minutes"],
        "goals_for": st["h_goals"],
        "goals_against": st["a_goals"],
    }
)
away_view = pd.DataFrame(
    {
        "team": st["a_team"],
        "opp": st["h_team"],
        "lineup": st["a_lineup"],
        "minutes": st["minutes"],
        "goals_for": st["a_goals"],
        "goals_against": st["h_goals"],
    }
)

team_lineups = pd.concat([home_view, away_view], ignore_index=True)

lineup_eff = (
    team_lineups.groupby(["team", "lineup"], as_index=False)
    .agg(
        minutes=("minutes", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        stints=("minutes", "count"),
    )
)

lineup_eff["net_goals"] = lineup_eff["goals_for"] - lineup_eff["goals_against"]
lineup_eff["net_per_min"] = lineup_eff["net_goals"] / lineup_eff["minutes"]
lineup_eff["gf_per_min"] = lineup_eff["goals_for"] / lineup_eff["minutes"]
lineup_eff["ga_per_min"] = lineup_eff["goals_against"] / lineup_eff["minutes"]

# Save full lineup effectiveness table
lineup_eff.to_csv(os.path.join(OUT_DIR, "lineup_effectiveness_by_team.csv"), index=False)

# Most-used lineups per team (by minutes)
lineup_most_used = (
    lineup_eff.sort_values(["team", "minutes"], ascending=[True, False])
    .groupby("team", as_index=False)
    .head(10)
)
lineup_most_used.to_csv(os.path.join(OUT_DIR, "lineup_most_used_by_team.csv"), index=False)

# Best lineups per team (by net_per_min) with minimum minutes threshold
MIN_MINUTES = 20.0
lineup_best = (
    lineup_eff[lineup_eff["minutes"] >= MIN_MINUTES]
    .sort_values(["team", "net_per_min"], ascending=[True, False])
    .groupby("team", as_index=False)
    .head(10)
)
lineup_best.to_csv(os.path.join(OUT_DIR, "lineup_best_by_team.csv"), index=False)

print("\n=== (1) LINEUP EFFECTIVENESS ===")
print(f"Saved: {OUT_DIR}/lineup_effectiveness_by_team.csv (all lineups)")
print(f"Saved: {OUT_DIR}/lineup_most_used_by_team.csv (top 10 by minutes per team)")
print(f"Saved: {OUT_DIR}/lineup_best_by_team.csv (top 10 by net_per_min per team, minutes >= {MIN_MINUTES})")

print("\nTop 10 lineups overall by net_per_min (minutes >= threshold):")
print(lineup_best.sort_values("net_per_min", ascending=False).head(10).to_string(index=False))

# ============================================================
# (2) PHYSICAL RATING vs SCORING (two proxies)
# ============================================================

def avg_rating(lineup_tuple):
    vals = [rating_map.get(p, np.nan) for p in lineup_tuple]
    vals = [v for v in vals if not pd.isna(v)]
    return float(np.mean(vals)) if len(vals) else np.nan

st["h_avg_rating"] = st["h_lineup"].apply(avg_rating)
st["a_avg_rating"] = st["a_lineup"].apply(avg_rating)

st["h_gpm"] = st["h_goals"] / st["minutes"]
st["a_gpm"] = st["a_goals"] / st["minutes"]

rating_lineup_level = pd.DataFrame(
    {
        "side": ["home"] * len(st) + ["away"] * len(st),
        "team": pd.concat([st["h_team"], st["a_team"]], ignore_index=True),
        "avg_rating": pd.concat([st["h_avg_rating"], st["a_avg_rating"]], ignore_index=True),
        "goals_per_min": pd.concat([st["h_gpm"], st["a_gpm"]], ignore_index=True),
        "minutes": pd.concat([st["minutes"], st["minutes"]], ignore_index=True),
    }
).dropna(subset=["avg_rating", "goals_per_min", "minutes"])

def weighted_corr(x, y, w):
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return np.nan
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan

x = rating_lineup_level["avg_rating"].to_numpy()
y = rating_lineup_level["goals_per_min"].to_numpy()
w = rating_lineup_level["minutes"].to_numpy()
corr_w = weighted_corr(x, y, w)

bins = [-0.1, 1.0, 2.0, 3.0, 3.5]
labels = ["<=1.0", "1.5-2.0", "2.5-3.0", "3.5"]
rating_lineup_level["avg_rating_bin"] = pd.cut(rating_lineup_level["avg_rating"], bins=bins, labels=labels)

# Avoid zero-weight bins by using observed=True (skips empty categorical bins)
rating_lineup_summary = (
    rating_lineup_level.groupby("avg_rating_bin", observed=True)
    .apply(
        lambda g: pd.Series(
            {
                "minutes": g["minutes"].sum(),
                "avg_goals_per_min": np.average(g["goals_per_min"], weights=g["minutes"])
                if g["minutes"].sum() > 0
                else np.nan,
                "n_stints": len(g),
            }
        )
    )
    .reset_index()
)

rating_lineup_level.to_csv(os.path.join(OUT_DIR, "rating_scoring_lineup_level.csv"), index=False)

print("\n=== (2A) PHYSICAL RATING vs SCORING (Lineup-level proxy) ===")
print(f"Weighted correlation(avg_rating, goals_per_min) by minutes: {corr_w:.4f}")
print(f"Saved: {OUT_DIR}/rating_scoring_lineup_level.csv")
print("\nLineup avg-rating bins summary (minutes-weighted goals/min):")
print(rating_lineup_summary.to_string(index=False))

def explode_side(st_side_team, st_side_lineup, st_side_goals, st_side_minutes):
    rows = []
    for team, lineup, goals, mins in zip(st_side_team, st_side_lineup, st_side_goals, st_side_minutes):
        if mins <= 0 or len(lineup) == 0:
            continue
        share = goals / len(lineup)
        for p in lineup:
            rows.append(
                {
                    "team": team,
                    "player": p,
                    "minutes": mins,
                    "goals_attrib": share,
                    "rating": rating_map.get(p, np.nan),
                }
            )
    return pd.DataFrame(rows)

home_long = explode_side(st["h_team"], st["h_lineup"], st["h_goals"], st["minutes"])
away_long = explode_side(st["a_team"], st["a_lineup"], st["a_goals"], st["minutes"])
player_attrib = pd.concat([home_long, away_long], ignore_index=True).dropna(subset=["rating"])

player_scoring = (
    player_attrib.groupby(["team", "player", "rating"], as_index=False)
    .agg(minutes=("minutes", "sum"), goals_attrib=("goals_attrib", "sum"))
)
player_scoring["goals_attrib_per_min"] = player_scoring["goals_attrib"] / player_scoring["minutes"]

rating_scoring = (
    player_scoring.groupby("rating", as_index=False)
    .agg(minutes=("minutes", "sum"), goals_attrib=("goals_attrib", "sum"))
)
rating_scoring["goals_attrib_per_min"] = rating_scoring["goals_attrib"] / rating_scoring["minutes"]

player_scoring.to_csv(os.path.join(OUT_DIR, "rating_scoring_player_attrib.csv"), index=False)

print("\n=== (2B) PHYSICAL RATING vs SCORING (Player attribution proxy) ===")
print(f"Saved: {OUT_DIR}/rating_scoring_player_attrib.csv")
print("\nGoals-attributed-per-minute by rating (equal-split proxy):")
print(rating_scoring.sort_values("rating").to_string(index=False))

# ============================================================
# (3) PLAYER VALUE METRICS (Phase 2 inputs)
#     - RAPM (ridge regression) on net goals/min
#     - On/Off impact
#     - Synergy residual vs RAPM prediction
# ============================================================

print("\n=== (3) PLAYER VALUE METRICS (Phase 2 inputs) ===")

# Team-perspective stints (each original stint becomes 2 rows: team + opponent)
home_view2 = pd.DataFrame(
    {
        "team": st["h_team"],
        "opp": st["a_team"],
        "lineup": st["h_lineup"],
        "opp_lineup": st["a_lineup"],
        "minutes": st["minutes"],
        "goals_for": st["h_goals"],
        "goals_against": st["a_goals"],
    }
)
away_view2 = pd.DataFrame(
    {
        "team": st["a_team"],
        "opp": st["h_team"],
        "lineup": st["a_lineup"],
        "opp_lineup": st["h_lineup"],
        "minutes": st["minutes"],
        "goals_for": st["a_goals"],
        "goals_against": st["h_goals"],
    }
)

team_stints = pd.concat([home_view2, away_view2], ignore_index=True)
team_stints["net_goals"] = team_stints["goals_for"] - team_stints["goals_against"]
team_stints["net_per_min"] = team_stints["net_goals"] / team_stints["minutes"]

# --- (3A) RAPM via Ridge Regression (Adjusted Plus/Minus) ---
players_list = pl["player"].astype(str).unique().tolist()
p2i = {p: i for i, p in enumerate(players_list)}
n = len(st)
m = len(players_list)

# y = (h_goals - a_goals)/minutes
y_r = ((st["h_goals"] - st["a_goals"]) / st["minutes"]).to_numpy(dtype=float)
w_r = st["minutes"].to_numpy(dtype=float)

X = np.zeros((n, m), dtype=float)

# +1 for home players
for col in home_cols:
    vals = st[col].astype(str).to_numpy()
    for r, p in enumerate(vals):
        j = p2i.get(p)
        if j is not None:
            X[r, j] += 1.0

# -1 for away players
for col in away_cols:
    vals = st[col].astype(str).to_numpy()
    for r, p in enumerate(vals):
        j = p2i.get(p)
        if j is not None:
            X[r, j] -= 1.0

# Weighted ridge closed-form: solve (X'WX + alpha I) b = X'W y
alpha = 1.0  # tune later in Phase 2 if needed
sw = np.sqrt(w_r)
Xw = X * sw[:, None]
yw = y_r * sw

XtX = Xw.T @ Xw
Xty = Xw.T @ yw
beta = np.linalg.solve(XtX + alpha * np.eye(m), Xty)

rapm = pd.DataFrame({"player": players_list, "rapm": beta})

# --- (3B) Minutes played per player (team-context) ---
rows = []
for team, lineup, mins in zip(team_stints["team"], team_stints["lineup"], team_stints["minutes"]):
    for p in lineup:
        rows.append({"team": team, "player": p, "minutes": mins})

player_minutes = (
    pd.DataFrame(rows)
    .groupby(["team", "player"], as_index=False)
    .agg(minutes_played=("minutes", "sum"))
)

# --- (3C) On/Off impact ---
team_totals = (
    team_stints.groupby("team", as_index=False)
    .agg(team_net_goals=("net_goals", "sum"), team_minutes=("minutes", "sum"))
)
team_totals["team_net_per_min"] = team_totals["team_net_goals"] / team_totals["team_minutes"]

rows = []
for team, lineup, mins, netg in zip(
    team_stints["team"], team_stints["lineup"], team_stints["minutes"], team_stints["net_goals"]
):
    for p in lineup:
        rows.append({"team": team, "player": p, "on_minutes": mins, "on_net_goals": netg})

on_tbl = (
    pd.DataFrame(rows)
    .groupby(["team", "player"], as_index=False)
    .agg(on_minutes=("on_minutes", "sum"), on_net_goals=("on_net_goals", "sum"))
)

onoff = on_tbl.merge(team_totals[["team", "team_net_goals", "team_minutes"]], on="team", how="left")
onoff["off_minutes"] = onoff["team_minutes"] - onoff["on_minutes"]
onoff["off_net_goals"] = onoff["team_net_goals"] - onoff["on_net_goals"]

onoff["on_net_per_min"] = np.where(onoff["on_minutes"] > 0, onoff["on_net_goals"] / onoff["on_minutes"], np.nan)
onoff["off_net_per_min"] = np.where(onoff["off_minutes"] > 0, onoff["off_net_goals"] / onoff["off_minutes"], np.nan)
onoff["on_off_net_per_min"] = onoff["on_net_per_min"] - onoff["off_net_per_min"]

# --- (3D) Synergy residual (Actual - Predicted) ---
beta_map = dict(zip(rapm["player"], rapm["rapm"]))

def lineup_value(lineup_tuple):
    if not lineup_tuple:
        return 0.0
    return float(np.sum([beta_map.get(p, 0.0) for p in lineup_tuple]))

team_stints["pred_net_per_min"] = team_stints["lineup"].apply(lineup_value) - team_stints["opp_lineup"].apply(lineup_value)
team_stints["residual_net_per_min"] = team_stints["net_per_min"] - team_stints["pred_net_per_min"]

rows = []
for team, lineup, mins, resid in zip(
    team_stints["team"], team_stints["lineup"], team_stints["minutes"], team_stints["residual_net_per_min"]
):
    for p in lineup:
        rows.append({"team": team, "player": p, "resid_minutes": mins, "resid_x_minutes": resid * mins})

syn = (
    pd.DataFrame(rows)
    .groupby(["team", "player"], as_index=False)
    .agg(resid_minutes=("resid_minutes", "sum"), resid_x_minutes=("resid_x_minutes", "sum"))
)
syn["synergy_residual_net_per_min"] = np.where(
    syn["resid_minutes"] > 0, syn["resid_x_minutes"] / syn["resid_minutes"], np.nan
)

# --- (3E) Avg lineup net/min when on (context performance) ---
rows = []
for team, lineup, mins, netpm in zip(
    team_stints["team"], team_stints["lineup"], team_stints["minutes"], team_stints["net_per_min"]
):
    for p in lineup:
        rows.append({"team": team, "player": p, "mins": mins, "netpm_x_mins": netpm * mins})

avg_netpm = (
    pd.DataFrame(rows)
    .groupby(["team", "player"], as_index=False)
    .agg(m=("mins", "sum"), netpm_x_m=("netpm_x_mins", "sum"))
)
avg_netpm["avg_lineup_net_per_min_when_on"] = np.where(
    avg_netpm["m"] > 0, avg_netpm["netpm_x_m"] / avg_netpm["m"], np.nan
)

# --- (3F) Combine into one Phase 2 input table ---
player_values = (
    player_minutes
    .merge(onoff[["team", "player", "on_net_per_min", "off_net_per_min", "on_off_net_per_min"]],
           on=["team", "player"], how="left")
    .merge(syn[["team", "player", "synergy_residual_net_per_min"]],
           on=["team", "player"], how="left")
    .merge(avg_netpm[["team", "player", "avg_lineup_net_per_min_when_on"]],
           on=["team", "player"], how="left")
    .merge(pl[["player", "rating"]], on="player", how="left")
    .merge(rapm, on="player", how="left")  # RAPM is global
)

out_player_values = os.path.join(OUT_DIR, "player_value_metrics.csv")
player_values.sort_values(["team", "rapm"], ascending=[True, False]).to_csv(out_player_values, index=False)

print(f"Saved: {out_player_values}")
print("\nTop 10 players by RAPM (overall):")
print(rapm.sort_values("rapm", ascending=False).head(10).to_string(index=False))

# ============================================================
# Plots (saved)
# ============================================================
if RUN_PLOTS:
    # Plot 1: scoring vs lineup average rating
    plt.figure()
    plt.bar(
        rating_lineup_summary["avg_rating_bin"].astype(str),
        rating_lineup_summary["avg_goals_per_min"],
    )
    plt.xlabel("Lineup average rating bin")
    plt.ylabel("Minutes-weighted goals per minute")
    plt.title("Scoring rate vs lineup average physical rating (proxy)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scoring_vs_lineup_avg_rating.png"), dpi=200)
    plt.close()

    # Plot 2: attributed scoring proxy vs rating
    plt.figure()
    plt.plot(rating_scoring["rating"], rating_scoring["goals_attrib_per_min"], marker="o")
    plt.xlabel("Player physical rating")
    plt.ylabel("Attributed goals per minute (equal-split proxy)")
    plt.title("Scoring contribution proxy vs physical rating")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "attrib_scoring_vs_rating.png"), dpi=200)
    plt.close()

    # Plot 3: top 10 lineups overall by net_per_min
    top10 = lineup_best.sort_values("net_per_min", ascending=False).head(10).copy()
    top10["label"] = top10["team"] + " | " + top10["lineup"].astype(str)

    plt.figure(figsize=(10, 5))
    plt.barh(top10["label"], top10["net_per_min"])
    plt.xlabel("Net goals per minute")
    plt.title("Top 10 lineups overall by net goals per minute (min minutes threshold)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top10_lineups_net_per_min.png"), dpi=200)
    plt.close()

    print(f"\nSaved plots to: {PLOTS_DIR}")

print("\nDONE.")
