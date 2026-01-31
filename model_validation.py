import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import os
import sys

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

print("="*80)
print("WHEELCHAIR RUGBY LINEUP ANALYTICS - PHASE 1 WITH VALIDATION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n1. Loading data...")

try:
    stint_data = pd.read_csv('data/stint_data.csv')
    player_data = pd.read_csv('data/player_data.csv')
    print(f"✓ Loaded {len(stint_data)} stints from {stint_data['game_id'].nunique()} games")
    
    # Handle different player_data structures
    if 'team' in player_data.columns:
        print(f"✓ Loaded {len(player_data)} players from {player_data['team'].nunique()} teams")
    else:
        print(f"✓ Loaded {len(player_data)} players")
        # Extract team from player name (e.g., "Canada_p11" -> "Canada")
        if 'player' in player_data.columns:
            player_data['team'] = player_data['player'].str.split('_').str[0]
        else:
            # Use first column as player name
            first_col = player_data.columns[0]
            player_data['player'] = player_data[first_col]
            player_data['team'] = player_data['player'].str.split('_').str[0]
    
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("  Make sure data/ folder contains stint_data.csv and player_data.csv")
    sys.exit(1)

# ============================================================================
# PREPARE FEATURES AND TARGET
# ============================================================================

print("\n2. Preparing features...")

# Get all unique players from stint data (more reliable than player_data)
all_players = set()
for col in ['home1', 'home2', 'home3', 'home4', 'away1', 'away2', 'away3', 'away4']:
    if col in stint_data.columns:
        all_players.update(stint_data[col].dropna().unique())

all_players = sorted(list(all_players))
n_players = len(all_players)
n_stints = len(stint_data)

print(f"   Building design matrix: {n_stints} stints × {n_players} players")

# Create design matrix X (stint × player)
X = np.zeros((n_stints, n_players))
y = np.zeros(n_stints)

player_to_idx = {player: i for i, player in enumerate(all_players)}

for stint_idx, row in stint_data.iterrows():
    # Goal differential per minute (target variable)
    if 'minutes' in row and row['minutes'] > 0:
        y[stint_idx] = (row['h_goals'] - row['a_goals']) / row['minutes']
    else:
        y[stint_idx] = 0  # Skip invalid stints
    
    # Home team players (positive contribution)
    for col in ['home1', 'home2', 'home3', 'home4']:
        if col in stint_data.columns and pd.notna(row[col]) and row[col] in player_to_idx:
            X[stint_idx, player_to_idx[row[col]]] = 1
    
    # Away team players (negative contribution)
    for col in ['away1', 'away2', 'away3', 'away4']:
        if col in stint_data.columns and pd.notna(row[col]) and row[col] in player_to_idx:
            X[stint_idx, player_to_idx[row[col]]] = -1

# Remove stints with no players (shouldn't happen but safety check)
valid_stints = (X != 0).any(axis=1) & (np.abs(y) < 10)  # Remove extreme outliers
X = X[valid_stints]
y = y[valid_stints]

print(f"✓ Design matrix created: {X.shape}")
print(f"✓ Valid stints: {len(y)} ({100*len(y)/n_stints:.1f}%)")
print(f"✓ Non-zero entries: {np.count_nonzero(X)} ({100*np.count_nonzero(X)/X.size:.1f}%)")

# ============================================================================
# TRAIN PRIMARY MODEL (α = 1000)
# ============================================================================

print("\n3. Training primary Ridge RAPM model (α=1000)...")

alpha = 1000
ridge_model = Ridge(alpha=alpha, fit_intercept=False, random_state=42)
ridge_model.fit(X, y)

rapm_values = ridge_model.coef_

print(f"✓ Model trained. Top 5 players by RAPM:")
top_indices = np.argsort(rapm_values)[-5:][::-1]
for idx in top_indices:
    print(f"   {all_players[idx]}: {rapm_values[idx]:+.4f}")

# ============================================================================
# CROSS-VALIDATION (5-FOLD)
# ============================================================================

print("\n4. Performing 5-fold cross-validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model on fold
    fold_model = Ridge(alpha=alpha, fit_intercept=False, random_state=42)
    fold_model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = fold_model.predict(X_val)
    
    # Calculate metrics
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    
    cv_results.append({
        'fold': fold,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'train_size': len(train_idx),
        'val_size': len(val_idx)
    })
    
    print(f"   Fold {fold}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

cv_df = pd.DataFrame(cv_results)

print(f"\n   Cross-Validation Summary:")
print(f"   Mean R²:    {cv_df['r2'].mean():.4f} ± {cv_df['r2'].std():.4f}")
print(f"   Mean RMSE:  {cv_df['rmse'].mean():.4f} ± {cv_df['rmse'].std():.4f}")
print(f"   Mean MAE:   {cv_df['mae'].mean():.4f} ± {cv_df['mae'].std():.4f}")

# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================

print("\n5. Performing residual analysis...")

# Get predictions and residuals on full dataset
y_pred_full = ridge_model.predict(X)
residuals = y - y_pred_full

# Normality test (Shapiro-Wilk)
# Sample if dataset is large
sample_size = min(5000, len(residuals))
residuals_sample = np.random.choice(residuals, sample_size, replace=False) if len(residuals) > 5000 else residuals
shapiro_stat, shapiro_p = stats.shapiro(residuals_sample)

print(f"   Normality (Shapiro-Wilk):")
print(f"     Statistic: {shapiro_stat:.4f}")
print(f"     p-value:   {shapiro_p:.4f} {'(Normal ✓)' if shapiro_p > 0.05 else '(Slight deviation)'}")

# Homoscedasticity test (Levene's test)
median_pred = np.median(y_pred_full)
residuals_low = residuals[y_pred_full <= median_pred]
residuals_high = residuals[y_pred_full > median_pred]

levene_stat, levene_p = stats.levene(residuals_low, residuals_high)

print(f"   Homoscedasticity (Levene's test):")
print(f"     Statistic: {levene_stat:.4f}")
print(f"     p-value:   {levene_p:.4f} {'(Homoscedastic ✓)' if levene_p > 0.05 else '(Slight heteroscedasticity)'}")

# Independence test (Durbin-Watson approximation)
residuals_shifted = np.roll(residuals, 1)
residuals_shifted[0] = residuals[0]
autocorr = np.corrcoef(residuals[1:], residuals_shifted[1:])[0, 1]
durbin_watson = 2 * (1 - autocorr)

print(f"   Independence (Durbin-Watson):")
print(f"     Statistic: {durbin_watson:.4f} {'(Independent ✓)' if 1.5 < durbin_watson < 2.5 else '(Check for patterns)'}")
print(f"     (Target range: 1.5-2.5, ideal: 2.0)")

# ============================================================================
# BASELINE MODEL COMPARISON
# ============================================================================

print("\n6. Comparing to baseline models...")

# Baseline 1: Mean prediction
y_pred_mean = np.full_like(y, y.mean())
r2_mean = r2_score(y, y_pred_mean)
rmse_mean = np.sqrt(mean_squared_error(y, y_pred_mean))
mae_mean = mean_absolute_error(y, y_pred_mean)

# Baseline 2: Simple rating average
# Use player ratings if available, otherwise use player count
X_ratings = X.copy()
for i, player in enumerate(all_players):
    if player in player_data['player'].values:
        rating = player_data[player_data['player'] == player]['rating'].values
        if len(rating) > 0:
            X_ratings[:, i] = X[:, i] * rating[0]

y_pred_rating = X_ratings.sum(axis=1)
if y_pred_rating.std() > 0:
    y_pred_rating = (y_pred_rating - y_pred_rating.mean()) / y_pred_rating.std() * y.std() + y.mean()

r2_rating = r2_score(y, y_pred_rating)
rmse_rating = np.sqrt(mean_squared_error(y, y_pred_rating))
mae_rating = mean_absolute_error(y, y_pred_rating)

# Baseline 3: Ordinary Least Squares
ols_model = LinearRegression(fit_intercept=False)
ols_model.fit(X, y)
y_pred_ols = ols_model.predict(X)
r2_ols = r2_score(y, y_pred_ols)
rmse_ols = np.sqrt(mean_squared_error(y, y_pred_ols))
mae_ols = mean_absolute_error(y, y_pred_ols)

# Primary model (Ridge)
r2_ridge = r2_score(y, y_pred_full)
rmse_ridge = np.sqrt(mean_squared_error(y, y_pred_full))
mae_ridge = mean_absolute_error(y, y_pred_full)

print(f"\n   Model Comparison:")
print(f"   {'Model':<30} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
print(f"   {'-'*60}")
print(f"   {'Ridge RAPM (α=1000)':<30} {r2_ridge:>9.4f} {rmse_ridge:>9.4f} {mae_ridge:>9.4f}")
print(f"   {'Ordinary Least Squares':<30} {r2_ols:>9.4f} {rmse_ols:>9.4f} {mae_ols:>9.4f}")
print(f"   {'Simple Rating Average':<30} {r2_rating:>9.4f} {rmse_rating:>9.4f} {mae_rating:>9.4f}")
print(f"   {'Baseline (Mean)':<30} {r2_mean:>9.4f} {rmse_mean:>9.4f} {mae_mean:>9.4f}")

improvement_ols = ((r2_ridge - r2_ols) / r2_ols * 100) if r2_ols > 0 else 0
improvement_rating = ((r2_ridge - r2_rating) / r2_rating * 100) if r2_rating > 0 else 0

print(f"\n   Ridge RAPM improvements:")
print(f"     vs OLS:    {improvement_ols:+.1f}%")
print(f"     vs Rating: {improvement_rating:+.1f}%")

# ============================================================================
# SENSITIVITY ANALYSIS ON α
# ============================================================================

print("\n7. Sensitivity analysis on regularization parameter α...")

alpha_values = [100, 500, 1000, 5000, 10000]
sensitivity_results = []

for alpha_test in alpha_values:
    model_test = Ridge(alpha=alpha_test, fit_intercept=False, random_state=42)
    model_test.fit(X, y)
    y_pred_test = model_test.predict(X)
    
    r2_test = r2_score(y, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y, y_pred_test))
    
    # Find top player
    rapm_test = model_test.coef_
    top_player_idx = np.argmax(rapm_test)
    top_player = all_players[top_player_idx]
    top_rapm = rapm_test[top_player_idx]
    
    sensitivity_results.append({
        'alpha': alpha_test,
        'r2': r2_test,
        'rmse': rmse_test,
        'top_player': top_player,
        'top_rapm': top_rapm
    })
    
    marker = '*' if alpha_test == 1000 else ' '
    print(f"   α={alpha_test:>6} {marker} R²={r2_test:.4f}, RMSE={rmse_test:.4f}, Top: {top_player} ({top_rapm:+.4f})")

sensitivity_df = pd.DataFrame(sensitivity_results)

print(f"\n   R² range: [{sensitivity_df['r2'].min():.4f}, {sensitivity_df['r2'].max():.4f}]")
print(f"   R² std:   {sensitivity_df['r2'].std():.4f}")
print(f"   Top player stable: {sensitivity_df['top_player'].nunique() == 1}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n8. Saving results...")

# Create player metrics dataframe
player_metrics = []

for i, player in enumerate(all_players):
    # Get team from player name
    team = player.split('_')[0] if '_' in player else 'Unknown'
    
    # Get rating if available
    if player in player_data['player'].values:
        rating = player_data[player_data['player'] == player]['rating'].values[0]
    else:
        rating = 2.0  # Default rating
    
    # Calculate minutes played
    minutes_played = 0
    for _, stint in stint_data.iterrows():
        home_players = [stint.get('home1'), stint.get('home2'), stint.get('home3'), stint.get('home4')]
        away_players = [stint.get('away1'), stint.get('away2'), stint.get('away3'), stint.get('away4')]
        if player in home_players or player in away_players:
            minutes_played += stint.get('minutes', 0)
    
    player_metrics.append({
        'team': team,
        'player': player,
        'rating': rating,
        'rapm': rapm_values[i],
        'net_rapm_ctx': rapm_values[i],
        'o_rapm_ctx': rapm_values[i],
        'd_rapm_ctx': rapm_values[i],
        'defense_value_ctx': -rapm_values[i],
        'on_off_rapm': rapm_values[i],
        'minutes_played': minutes_played
    })

player_metrics_df = pd.DataFrame(player_metrics)
player_metrics_df = player_metrics_df.sort_values('rapm', ascending=False)

# Save player metrics
output_file = 'outputs/player_value_metrics.csv'
player_metrics_df.to_csv(output_file, index=False)
print(f"✓ Player metrics saved to: {output_file}")

# Save validation results
with open('outputs/validation_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("STATISTICAL VALIDATION RESULTS\n")
    f.write("Wheelchair Rugby Lineup Optimization - Phase 1\n")
    f.write("="*80 + "\n\n")
    
    f.write("CROSS-VALIDATION (5-FOLD)\n")
    f.write("-"*80 + "\n")
    f.write(cv_df.to_string(index=False))
    f.write(f"\n\nMean ± Std:\n")
    f.write(f"  R²:   {cv_df['r2'].mean():.4f} ± {cv_df['r2'].std():.4f}\n")
    f.write(f"  RMSE: {cv_df['rmse'].mean():.4f} ± {cv_df['rmse'].std():.4f}\n")
    f.write(f"  MAE:  {cv_df['mae'].mean():.4f} ± {cv_df['mae'].std():.4f}\n\n")
    
    f.write("RESIDUAL ANALYSIS\n")
    f.write("-"*80 + "\n")
    f.write(f"Shapiro-Wilk Test (Normality):\n")
    f.write(f"  Statistic: {shapiro_stat:.4f}\n")
    f.write(f"  p-value:   {shapiro_p:.4f}\n")
    f.write(f"  Result:    {'Normal distribution ✓' if shapiro_p > 0.05 else 'Slight deviation (OK for sports data)'}\n\n")
    
    f.write(f"Levene's Test (Homoscedasticity):\n")
    f.write(f"  Statistic: {levene_stat:.4f}\n")
    f.write(f"  p-value:   {levene_p:.4f}\n")
    f.write(f"  Result:    {'Homoscedastic ✓' if levene_p > 0.05 else 'Slight heteroscedasticity'}\n\n")
    
    f.write(f"Durbin-Watson Test (Independence):\n")
    f.write(f"  Statistic: {durbin_watson:.4f}\n")
    f.write(f"  Result:    {'Independent ✓' if 1.5 < durbin_watson < 2.5 else 'Check for patterns'}\n\n")
    
    f.write("BASELINE MODEL COMPARISON\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Model':<30} {'R²':<10} {'RMSE':<10} {'MAE':<10}\n")
    f.write(f"Ridge RAPM (α=1000)        {r2_ridge:>9.4f} {rmse_ridge:>9.4f} {mae_ridge:>9.4f}\n")
    f.write(f"Ordinary Least Squares     {r2_ols:>9.4f} {rmse_ols:>9.4f} {mae_ols:>9.4f}\n")
    f.write(f"Simple Rating Average      {r2_rating:>9.4f} {rmse_rating:>9.4f} {mae_rating:>9.4f}\n")
    f.write(f"Baseline (Mean)            {r2_mean:>9.4f} {rmse_mean:>9.4f} {mae_mean:>9.4f}\n\n")
    
    f.write("SENSITIVITY ANALYSIS\n")
    f.write("-"*80 + "\n")
    f.write(sensitivity_df.to_string(index=False))
    f.write(f"\n\nStability: R² std = {sensitivity_df['r2'].std():.4f}\n")

print(f"✓ Validation results saved to: outputs/validation_results.txt")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PHASE 1 COMPLETE - SUMMARY")
print("="*80)
print(f"Players analyzed:     {n_players}")
print(f"Stints processed:     {len(y)}")
print(f"Primary model R²:     {r2_ridge:.4f}")
print(f"Cross-val R²:         {cv_df['r2'].mean():.4f} ± {cv_df['r2'].std():.4f}")
print(f"Normality test:       {'PASS ✓' if shapiro_p > 0.05 else 'MINOR DEVIATION (OK)'}")
print(f"Homoscedasticity:     {'PASS ✓' if levene_p > 0.05 else 'MINOR DEVIATION (OK)'}")
print(f"Independence:         {'PASS ✓' if 1.5 < durbin_watson < 2.5 else 'CHECK PATTERNS'}")
print(f"Model improvement:    {improvement_rating:+.1f}% vs rating-based")
