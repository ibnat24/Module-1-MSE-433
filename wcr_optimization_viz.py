import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load data
player_values = pd.read_csv("outputs/player_value_metrics.csv")
optimal_summary = pd.read_csv("outputs/optimal_lineups_summary.csv")
optimal_detailed = pd.read_csv("outputs/optimal_lineups_detailed.csv")

# Filter to Canada
canada_players = player_values[player_values['team'] == 'Canada'].copy()

OUT_DIR = "outputs/optimization_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Plot 1: Player Value Comparison (Canada)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sort by NET RAPM
canada_sorted = canada_players.sort_values('rapm', ascending=True)

# 1a: NET RAPM
axes[0, 0].barh(canada_sorted['player'], canada_sorted['rapm'], 
                color=['green' if x > 0 else 'red' for x in canada_sorted['rapm']])
axes[0, 0].axvline(0, color='black', linewidth=0.8)
axes[0, 0].set_xlabel('NET RAPM')
axes[0, 0].set_title('Player Value: NET RAPM')
axes[0, 0].grid(axis='x', alpha=0.3)

# 1b: Contextual NET RAPM
canada_sorted2 = canada_players.sort_values('net_rapm_ctx', ascending=True)
axes[0, 1].barh(canada_sorted2['player'], canada_sorted2['net_rapm_ctx'],
                color=['green' if x > 0 else 'red' for x in canada_sorted2['net_rapm_ctx']])
axes[0, 1].axvline(0, color='black', linewidth=0.8)
axes[0, 1].set_xlabel('NET RAPM (Contextual)')
axes[0, 1].set_title('Player Value: NET RAPM (Contextual)')
axes[0, 1].grid(axis='x', alpha=0.3)

# 1c: Offensive Impact
canada_sorted3 = canada_players.sort_values('o_rapm_ctx', ascending=True)
axes[1, 0].barh(canada_sorted3['player'], canada_sorted3['o_rapm_ctx'],
                color='steelblue')
axes[1, 0].set_xlabel('O-RAPM (Contextual)')
axes[1, 0].set_title('Offensive Impact')
axes[1, 0].grid(axis='x', alpha=0.3)

# 1d: Defensive Value
canada_sorted4 = canada_players.sort_values('defense_value_ctx', ascending=True)
axes[1, 1].barh(canada_sorted4['player'], canada_sorted4['defense_value_ctx'],
                color=['green' if x > 0 else 'red' for x in canada_sorted4['defense_value_ctx']])
axes[1, 1].axvline(0, color='black', linewidth=0.8)
axes[1, 1].set_xlabel('Defense Value (Contextual)')
axes[1, 1].set_title('Defensive Impact (higher = better)')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "player_value_metrics_comparison.png"), dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 2: Player Selection Frequency
# ============================================================
# Count how often each player appears
from collections import Counter
all_selected = []
for _, row in optimal_summary.iterrows():
    if pd.notna(row['players']):
        all_selected.extend(row['players'].split(', '))

player_counts = Counter(all_selected)
freq_df = pd.DataFrame(list(player_counts.items()), columns=['player', 'count'])
freq_df = freq_df.sort_values('count', ascending=True)
freq_df['percentage'] = (freq_df['count'] / len(optimal_summary)) * 100

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(freq_df['player'], freq_df['count'], color='steelblue')

# Color bars differently for core players
for i, (player, count) in enumerate(zip(freq_df['player'], freq_df['count'])):
    if count == len(optimal_summary):
        bars[i].set_color('darkgreen')
    elif count >= len(optimal_summary) * 0.5:
        bars[i].set_color('orange')

ax.set_xlabel('Number of Scenarios')
ax.set_title('Player Selection Frequency Across Optimization Scenarios')
ax.set_xlim(0, len(optimal_summary))
ax.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (player, count, pct) in enumerate(zip(freq_df['player'], freq_df['count'], freq_df['percentage'])):
    ax.text(count + 0.1, i, f'{pct:.0f}%', va='center')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "player_selection_frequency.png"), dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 3: Offense vs Defense Trade-off
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot of O-RAPM vs Defense Value
scatter = ax.scatter(canada_players['o_rapm_ctx'], 
                    canada_players['defense_value_ctx'],
                    s=canada_players['minutes_played']/10,
                    c=canada_players['rating'],
                    cmap='viridis',
                    alpha=0.6,
                    edgecolors='black')

# Annotate players
for _, row in canada_players.iterrows():
    ax.annotate(row['player'], 
               (row['o_rapm_ctx'], row['defense_value_ctx']),
               fontsize=9,
               alpha=0.7)

# Add quadrant lines
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Highlight optimal players
optimal_players = set(all_selected)
for _, row in canada_players[canada_players['player'].isin(optimal_players)].iterrows():
    ax.scatter(row['o_rapm_ctx'], row['defense_value_ctx'],
              s=row['minutes_played']/10 + 100,
              facecolors='none',
              edgecolors='red',
              linewidths=2)

ax.set_xlabel('Offensive Impact (O-RAPM Contextual)')
ax.set_ylabel('Defensive Impact (Defense Value, higher = better)')
ax.set_title('Offense vs Defense Trade-off\n(Size = Minutes Played, Color = Physical Rating)')
ax.grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Physical Rating')

# Add legend for optimal players
from matplotlib.patches import Circle
legend_elements = [Circle((0, 0), 1, facecolor='none', edgecolor='red', linewidth=2, 
                         label='Selected in Optimal Lineups')]
ax.legend(handles=legend_elements, loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "offense_defense_tradeoff.png"), dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 4: Scenario Comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

scenarios_opt = optimal_summary[optimal_summary['status'] == 'Optimal'].copy()

# 4a: Rating Utilization
axes[0, 0].bar(range(len(scenarios_opt)), scenarios_opt['total_rating'], color='steelblue')
axes[0, 0].axhline(8.0, color='red', linestyle='--', label='Maximum (8.0)')
axes[0, 0].set_xticks(range(len(scenarios_opt)))
axes[0, 0].set_xticklabels([s[:20] for s in scenarios_opt['scenario']], rotation=45, ha='right', fontsize=8)
axes[0, 0].set_ylabel('Total Rating')
axes[0, 0].set_title('Rating Utilization by Scenario')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# 4b: Average NET RAPM
axes[0, 1].bar(range(len(scenarios_opt)), scenarios_opt['avg_rapm'], color='green')
axes[0, 1].set_xticks(range(len(scenarios_opt)))
axes[0, 1].set_xticklabels([s[:20] for s in scenarios_opt['scenario']], rotation=45, ha='right', fontsize=8)
axes[0, 1].set_ylabel('Avg NET RAPM')
axes[0, 1].set_title('Average NET RAPM by Scenario')
axes[0, 1].grid(axis='y', alpha=0.3)

# 4c: Offensive vs Defensive Balance
axes[1, 0].scatter(scenarios_opt['avg_o_rapm_ctx'], 
                  scenarios_opt['avg_defense_value_ctx'],
                  s=200, alpha=0.6, c=range(len(scenarios_opt)), cmap='tab10')
for i, row in scenarios_opt.iterrows():
    axes[1, 0].annotate(row['scenario'][:15], 
                       (row['avg_o_rapm_ctx'], row['avg_defense_value_ctx']),
                       fontsize=7, alpha=0.7)
axes[1, 0].set_xlabel('Avg O-RAPM (Contextual)')
axes[1, 0].set_ylabel('Avg Defense Value')
axes[1, 0].set_title('Lineup Balance: Offense vs Defense')
axes[1, 0].grid(alpha=0.3)

# 4d: Total Minutes Experience
axes[1, 1].bar(range(len(scenarios_opt)), scenarios_opt['total_minutes_played'], color='coral')
axes[1, 1].set_xticks(range(len(scenarios_opt)))
axes[1, 1].set_xticklabels([s[:20] for s in scenarios_opt['scenario']], rotation=45, ha='right', fontsize=8)
axes[1, 1].set_ylabel('Total Minutes Played')
axes[1, 1].set_title('Combined Experience by Scenario')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scenario_comparison.png"), dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 5: Rating vs Performance
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5a: Rating vs NET RAPM
axes[0].scatter(canada_players['rating'], canada_players['rapm'], 
               s=canada_players['minutes_played']/10, alpha=0.6, c='steelblue')
for _, row in canada_players.iterrows():
    axes[0].annotate(row['player'], (row['rating'], row['rapm']),
                    fontsize=8, alpha=0.6)
axes[0].set_xlabel('Physical Rating')
axes[0].set_ylabel('NET RAPM')
axes[0].set_title('Physical Rating vs Player Value\n(Size = Minutes Played)')
axes[0].grid(alpha=0.3)
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# 5b: Rating vs Net RAPM (Contextual)
axes[1].scatter(canada_players['rating'], canada_players['net_rapm_ctx'],
               s=canada_players['minutes_played']/10, alpha=0.6, c='orange')
for _, row in canada_players.iterrows():
    axes[1].annotate(row['player'], (row['rating'], row['net_rapm_ctx']),
                    fontsize=8, alpha=0.6)
axes[1].set_xlabel('Physical Rating')
axes[1].set_ylabel('NET RAPM (Contextual)')
axes[1].set_title('Physical Rating vs Contextual Player Value\n(Size = Minutes Played)')
axes[1].grid(alpha=0.3)
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rating_vs_performance.png"), dpi=200, bbox_inches='tight')
plt.close()

print(f"\n✓ All visualization plots saved to: {OUT_DIR}/")
print("\nGenerated plots:")
print("  1. player_value_metrics_comparison.png")
print("  2. player_selection_frequency.png")
print("  3. offense_defense_tradeoff.png")
print("  4. scenario_comparison.png")
print("  5. rating_vs_performance.png")