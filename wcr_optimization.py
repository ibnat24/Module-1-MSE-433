"""
Wheelchair Rugby Lineup Optimization (MSE433 Module 1 → Phase 2)

PHASE 2: OPTIMIZATION MODEL
Uses player value metrics from Phase 1 to find optimal lineups

Key Constraints:
- Exactly 4 players on court
- Sum of physical ratings ≤ 8.0 (+ 0.5 per female player)
- Binary selection variables

Objective Functions (multiple scenarios):
1. Maximize NET RAPM (baseline individual value)
2. Maximize NET RAPM_CTX (contextual net impact)
3. Maximize O-RAPM_CTX (offensive focus)
4. Maximize DEFENSE_VALUE_CTX (defensive focus)
5. Maximize On/Off Net Impact
6. Balanced lineup (combination of offense + defense)

Outputs:
    outputs/optimal_lineups_summary.csv
    outputs/optimal_lineups_detailed.csv
"""

import os
import numpy as np
import pandas as pd
from itertools import combinations

# -----------------------------
# Configuration
# -----------------------------
PLAYER_VALUES_PATH = "outputs/player_value_metrics.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Which team to optimize for
TARGET_TEAM = "Canada"

# Constraints
LINEUP_SIZE = 4
MAX_RATING_SUM = 8.0
FEMALE_BONUS = 0.5  # Additional rating allowance per female player

print("="*70)
print("WHEELCHAIR RUGBY LINEUP OPTIMIZATION")
print("="*70)

# -----------------------------
# Load Data
# -----------------------------
print(f"\n[1] Loading player value metrics...")
player_df = pd.read_csv(PLAYER_VALUES_PATH)

# Filter to target team
team_players = player_df[player_df['team'] == TARGET_TEAM].copy()
print(f"    Team: {TARGET_TEAM}")
print(f"    Total players: {len(team_players)}")

# For this example, we'll assume no female players unless specified
# You can add a 'is_female' column to player_data.csv if needed
team_players['is_female'] = False
team_players['rating_adjusted'] = team_players['rating']  # Will be used in constraints

# Display available metrics
print(f"\n[2] Available player value metrics:")
metrics = ['rapm', 'net_rapm_ctx', 'o_rapm_ctx', 'defense_value_ctx', 
           'on_off_net_per_min', 'on_off_gf_per_min', 'on_off_ga_per_min']
for m in metrics:
    if m in team_players.columns:
        print(f"    ✓ {m}")

# -----------------------------
# Optimization Scenarios
# -----------------------------
scenarios = [
    {
        'name': 'Maximize NET RAPM',
        'objective': 'rapm',
        'description': 'Individual player value (adjusted for teammates/opponents)'
    },
    {
        'name': 'Maximize NET RAPM (Contextual)',
        'objective': 'net_rapm_ctx',
        'description': 'Contextual net impact (O-RAPM - D-RAPM)'
    },
    {
        'name': 'Maximize Offensive Impact',
        'objective': 'o_rapm_ctx',
        'description': 'Goals scored per minute (contextual)'
    },
    {
        'name': 'Maximize Defensive Impact',
        'objective': 'defense_value_ctx',
        'description': 'Goals prevented per minute (higher is better)'
    },
    {
        'name': 'Maximize On/Off Net',
        'objective': 'on_off_net_per_min',
        'description': 'Team performance difference when player is on vs off court'
    },
    {
        'name': 'Balanced Lineup (50/50)',
        'objective': 'balanced',
        'description': 'Equal weight to offense and defense',
        'weights': {'o_rapm_ctx': 0.5, 'defense_value_ctx': 0.5}
    },
    {
        'name': 'Balanced Lineup (60/40 Offense)',
        'objective': 'balanced_offense',
        'description': 'Offense-oriented balanced approach',
        'weights': {'o_rapm_ctx': 0.6, 'defense_value_ctx': 0.4}
    },
    {
        'name': 'Balanced Lineup (60/40 Defense)',
        'objective': 'balanced_defense',
        'description': 'Defense-oriented balanced approach',
        'weights': {'o_rapm_ctx': 0.4, 'defense_value_ctx': 0.6}
    }
]

# -----------------------------
# Helper Function: Evaluate a lineup
# -----------------------------
def evaluate_lineup(players_list, team_df, objective_type, weights=None):
    """
    Evaluate a lineup based on objective function.
    Returns the objective value and whether it's feasible.
    """
    lineup_df = team_df[team_df['player'].isin(players_list)].copy()
    
    # Check constraints
    total_rating = lineup_df['rating'].sum()
    if total_rating > MAX_RATING_SUM:
        return None, False  # Infeasible
    
    # Calculate objective value
    if objective_type in ['rapm', 'net_rapm_ctx', 'o_rapm_ctx', 
                           'defense_value_ctx', 'on_off_net_per_min']:
        obj_value = lineup_df[objective_type].sum()
    elif weights is not None:
        # Weighted combination
        obj_value = 0
        for metric, weight in weights.items():
            obj_value += weight * lineup_df[metric].sum()
    else:
        raise ValueError(f"Unknown objective: {objective_type}")
    
    return obj_value, True

# -----------------------------
# Solve Each Scenario via Enumeration
# -----------------------------
print(f"\n[3] Generating all possible lineups...")
all_players = team_players['player'].tolist()
all_lineups = list(combinations(all_players, LINEUP_SIZE))
print(f"    Total possible 4-player lineups: {len(all_lineups)}")

results = []

for idx, scenario in enumerate(scenarios, 1):
    print(f"\n{'='*70}")
    print(f"SCENARIO {idx}: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    print(f"{'='*70}")
    
    # Evaluate all lineups
    best_obj_value = -np.inf
    best_lineup = None
    feasible_count = 0
    
    weights = scenario.get('weights', None)
    
    for lineup in all_lineups:
        obj_value, feasible = evaluate_lineup(
            lineup, team_players, scenario['objective'], weights
        )
        
        if feasible:
            feasible_count += 1
            if obj_value > best_obj_value:
                best_obj_value = obj_value
                best_lineup = lineup
    
    print(f"\n  Feasible lineups (rating ≤ {MAX_RATING_SUM}): {feasible_count}/{len(all_lineups)}")
    
    if best_lineup is not None:
        selected_players = list(best_lineup)
        
        # Get player details
        lineup_df = team_players[team_players['player'].isin(selected_players)].copy()
        
        # Calculate lineup statistics
        total_rating = lineup_df['rating'].sum()
        total_obj_value = best_obj_value
        
        # Additional metrics
        avg_rapm = lineup_df['rapm'].mean()
        avg_net_rapm_ctx = lineup_df['net_rapm_ctx'].mean()
        avg_o_rapm_ctx = lineup_df['o_rapm_ctx'].mean()
        avg_defense_value_ctx = lineup_df['defense_value_ctx'].mean()
        total_minutes = lineup_df['minutes_played'].sum()
        
        print(f"\n✓ Optimal Solution Found")
        print(f"\nSelected Players:")
        for p in selected_players:
            player_info = team_players[team_players['player'] == p].iloc[0]
            print(f"  • {p:15s} | Rating: {player_info['rating']:.1f} | "
                  f"RAPM: {player_info['rapm']:+.3f} | "
                  f"O-RAPM_CTX: {player_info['o_rapm_ctx']:+.3f} | "
                  f"DEF_VAL: {player_info['defense_value_ctx']:+.3f}")
        
        print(f"\nLineup Statistics:")
        print(f"  Total Rating Sum: {total_rating:.1f} / {MAX_RATING_SUM:.1f}")
        print(f"  Objective Value: {total_obj_value:.4f}")
        print(f"  Avg NET RAPM: {avg_rapm:.4f}")
        print(f"  Avg NET RAPM (CTX): {avg_net_rapm_ctx:.4f}")
        print(f"  Avg O-RAPM (CTX): {avg_o_rapm_ctx:.4f}")
        print(f"  Avg Defense Value: {avg_defense_value_ctx:.4f}")
        print(f"  Total Minutes Played: {total_minutes:.0f}")
        
        # Store results
        results.append({
            'scenario': scenario['name'],
            'description': scenario['description'],
            'objective_metric': scenario['objective'],
            'players': ', '.join(sorted(selected_players)),
            'total_rating': total_rating,
            'objective_value': total_obj_value,
            'avg_rapm': avg_rapm,
            'avg_net_rapm_ctx': avg_net_rapm_ctx,
            'avg_o_rapm_ctx': avg_o_rapm_ctx,
            'avg_defense_value_ctx': avg_defense_value_ctx,
            'total_minutes_played': total_minutes,
            'status': 'Optimal'
        })
        
    else:
        print(f"\n✗ No feasible solution found.")
        results.append({
            'scenario': scenario['name'],
            'description': scenario['description'],
            'objective_metric': scenario['objective'],
            'players': None,
            'status': 'Infeasible'
        })

# -----------------------------
# Save Results
# -----------------------------
print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, "optimal_lineups_summary.csv"), index=False)
print(f"✓ Saved: {OUT_DIR}/optimal_lineups_summary.csv")

# Create detailed comparison table
comparison_data = []
for result in results:
    if result['status'] == 'Optimal':
        players = result['players'].split(', ')
        for player in players:
            player_info = team_players[team_players['player'] == player].iloc[0]
            comparison_data.append({
                'scenario': result['scenario'],
                'player': player,
                'rating': player_info['rating'],
                'rapm': player_info['rapm'],
                'net_rapm_ctx': player_info['net_rapm_ctx'],
                'o_rapm_ctx': player_info['o_rapm_ctx'],
                'd_rapm_ctx': player_info['d_rapm_ctx'],
                'defense_value_ctx': player_info['defense_value_ctx'],
                'on_off_net_per_min': player_info['on_off_net_per_min'],
                'minutes_played': player_info['minutes_played']
            })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(os.path.join(OUT_DIR, "optimal_lineups_detailed.csv"), index=False)
print(f"✓ Saved: {OUT_DIR}/optimal_lineups_detailed.csv")

# -----------------------------
# Summary Analysis
# -----------------------------
print(f"\n{'='*70}")
print("SUMMARY ANALYSIS")
print(f"{'='*70}")

print("\n[1] Lineup Composition Comparison:")
optimal_results = results_df[results_df['status'] == 'Optimal']
for _, row in optimal_results.iterrows():
    print(f"\n{row['scenario']}:")
    print(f"  Players: {row['players']}")
    print(f"  Rating: {row['total_rating']:.1f}/8.0")
    print(f"  NET RAPM: {row['avg_rapm']:.4f}")

print("\n[2] Most Frequently Selected Players:")
all_selected_players = []
for result in results:
    if result['status'] == 'Optimal':
        all_selected_players.extend(result['players'].split(', '))

from collections import Counter
player_counts = Counter(all_selected_players)
print("\nPlayer Selection Frequency:")
for player, count in player_counts.most_common():
    pct = (count / len(scenarios)) * 100
    print(f"  {player:15s}: {count}/{len(scenarios)} scenarios ({pct:.0f}%)")

print("\n[3] Key Insights:")
# Find players in every optimal lineup
universal_players = [p for p, c in player_counts.items() if c == len(scenarios)]
if universal_players:
    print(f"  • Core players (in ALL lineups): {', '.join(universal_players)}")
else:
    print(f"  • No players appear in all optimal lineups")

# Find players never selected
never_selected = set(team_players['player']) - set(all_selected_players)
if never_selected:
    print(f"  • Players never selected: {', '.join(sorted(never_selected))}")

print("\n[4] Rating Utilization:")
avg_rating_used = optimal_results['total_rating'].mean()
print(f"  Average rating used: {avg_rating_used:.2f} / 8.0 ({(avg_rating_used/8.0)*100:.1f}%)")
print(f"  Range: {optimal_results['total_rating'].min():.1f} - {optimal_results['total_rating'].max():.1f}")

print("\n[5] Performance Metrics Across Scenarios:")
print(f"  NET RAPM range: {optimal_results['avg_rapm'].min():.4f} to {optimal_results['avg_rapm'].max():.4f}")
print(f"  O-RAPM (CTX) range: {optimal_results['avg_o_rapm_ctx'].min():.4f} to {optimal_results['avg_o_rapm_ctx'].max():.4f}")
print(f"  Defense Value range: {optimal_results['avg_defense_value_ctx'].min():.4f} to {optimal_results['avg_defense_value_ctx'].max():.4f}")

print(f"\n{'='*70}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*70}\n")