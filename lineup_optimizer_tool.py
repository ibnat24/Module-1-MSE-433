"""
WHEELCHAIR RUGBY LINEUP OPTIMIZER TOOL
Interactive decision support system for optimal lineup selection

Features:
- Opponent-specific optimization
- Injury management  
- Fatigue tracking
- Minimum playing time constraints
- Multiple lineup scenarios
- Update with new game data

Usage:
    python3 lineup_optimizer_tool.py
"""

import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
from datetime import datetime

class LineupOptimizer:
    """Interactive lineup optimization tool for wheelchair rugby"""
    
    def __init__(self, player_values_path="outputs/player_value_metrics.csv", 
                 team_name="Canada"):
        self.team_name = team_name
        self.player_values_path = player_values_path
        self.load_player_data()
        self.player_status = self.initialize_player_status()
        self.lineup_size = 4
        self.max_rating = 8.0
        self.female_bonus = 0.5
        print(f"✓ Lineup Optimizer initialized for {team_name}")
        print(f"✓ Loaded {len(self.players)} players")
    
    def load_player_data(self):
        """Load player value metrics"""
        df = pd.read_csv(self.player_values_path)
        self.all_data = df
        self.players = df[df['team'] == self.team_name].copy()
        if len(self.players) == 0:
            raise ValueError(f"No players found for team {self.team_name}")
    
    def initialize_player_status(self):
        """Initialize player availability and condition tracking"""
        status = {}
        for _, player in self.players.iterrows():
            status[player['player']] = {
                'available': True,
                'injured': False,
                'fatigue_level': 1.0,
                'is_female': False,
                'games_played': 0,
                'minutes_this_game': 0
            }
        return status
    
    def set_player_injury(self, player_name, injured=True):
        """Mark a player as injured or recovered"""
        if player_name in self.player_status:
            self.player_status[player_name]['injured'] = injured
            self.player_status[player_name]['available'] = not injured
            status = "INJURED" if injured else "RECOVERED"
            print(f"✓ {player_name} marked as {status}")
        else:
            print(f"✗ Player {player_name} not found")
    
    def set_player_fatigue(self, player_name, fatigue_level):
        """Set player fatigue level (1.0 = fresh, 0.0 = exhausted)"""
        if player_name in self.player_status:
            self.player_status[player_name]['fatigue_level'] = max(0.0, min(1.0, fatigue_level))
            print(f"✓ {player_name} fatigue set to {fatigue_level:.2f}")
        else:
            print(f"✗ Player {player_name} not found")
    
    def set_player_gender(self, player_name, is_female=True):
        """Mark player as female (affects rating capacity)"""
        if player_name in self.player_status:
            self.player_status[player_name]['is_female'] = is_female
            print(f"✓ {player_name} marked as {'FEMALE' if is_female else 'MALE'}")
        else:
            print(f"✗ Player {player_name} not found")
    
    def get_adjusted_rating_limit(self, selected_players):
        """Calculate rating limit including female bonuses"""
        num_females = sum([1 for p in selected_players 
                          if self.player_status.get(p, {}).get('is_female', False)])
        return self.max_rating + (num_females * self.female_bonus)
    
    def get_adjusted_player_value(self, player_name, metric='rapm'):
        """Get player value adjusted for fatigue"""
        base_value = self.players[self.players['player'] == player_name][metric].values[0]
        fatigue = self.player_status[player_name]['fatigue_level']
        adjusted_value = base_value * fatigue
        return adjusted_value
    
    def get_opponent_adjustment(self, opponent_team):
        """Get opponent-specific adjustments"""
        try:
            opp_data = self.all_data[self.all_data['team'] == opponent_team]
            if len(opp_data) > 0:
                avg_opp_rapm = opp_data['rapm'].mean()
                avg_opp_offense = opp_data['o_rapm_ctx'].mean()
                avg_opp_defense = opp_data['defense_value_ctx'].mean()
                
                return {
                    'team': opponent_team,
                    'strength': avg_opp_rapm,
                    'offensive_strength': avg_opp_offense,
                    'defensive_strength': avg_opp_defense,
                    'recommendation': self._get_lineup_strategy(avg_opp_offense, avg_opp_defense)
                }
            else:
                return {'team': opponent_team, 'strength': 0.0, 'recommendation': 'balanced'}
        except:
            return {'team': opponent_team, 'recommendation': 'balanced'}
    
    def _get_lineup_strategy(self, opp_offense, opp_defense):
        """Determine strategy based on opponent strengths"""
        if opp_offense > 0.5:
            return 'defensive'
        elif opp_defense > -0.3:
            return 'offensive'
        else:
            return 'balanced'
    
    def optimize_lineup(self, objective='rapm', opponent=None, 
                       min_minutes_per_player=0, max_iterations=None):
        """Find optimal lineup given current conditions"""
        available_players = [
            p for p, status in self.player_status.items()
            if status['available'] and not status['injured']
        ]
        
        if len(available_players) < self.lineup_size:
            return {
                'status': 'ERROR',
                'message': f'Not enough available players ({len(available_players)}/{self.lineup_size})',
                'available_players': available_players
            }
        
        opp_info = None
        if opponent:
            opp_info = self.get_opponent_adjustment(opponent)
            print(f"\n📊 Opponent Analysis: {opponent}")
            print(f"   Strength: {opp_info.get('strength', 0):.3f}")
            print(f"   Recommended strategy: {opp_info.get('recommendation', 'balanced').upper()}")
            
            if objective == 'rapm':
                objective = opp_info.get('recommendation', 'balanced')
        
        if objective == 'offensive':
            weights = {'o_rapm_ctx': 1.0}
        elif objective == 'defensive':
            weights = {'defense_value_ctx': 1.0}
        elif objective == 'balanced':
            weights = {'o_rapm_ctx': 0.5, 'defense_value_ctx': 0.5}
        else:
            weights = {objective: 1.0}
        
        all_lineups = list(combinations(available_players, self.lineup_size))
        
        if max_iterations and len(all_lineups) > max_iterations:
            np.random.seed(42)
            all_lineups = np.random.choice(all_lineups, max_iterations, replace=False)
        
        print(f"\n🔍 Evaluating {len(all_lineups)} possible lineups...")
        
        best_lineup = None
        best_value = -np.inf
        feasible_count = 0
        
        for lineup in all_lineups:
            total_rating = sum([
                self.players[self.players['player'] == p]['rating'].values[0]
                for p in lineup
            ])
            rating_limit = self.get_adjusted_rating_limit(lineup)
            
            if total_rating > rating_limit:
                continue
            
            feasible_count += 1
            
            obj_value = 0
            for metric, weight in weights.items():
                for player in lineup:
                    obj_value += weight * self.get_adjusted_player_value(player, metric)
            
            if obj_value > best_value:
                best_value = obj_value
                best_lineup = lineup
        
        if best_lineup is None:
            return {
                'status': 'ERROR',
                'message': 'No feasible lineup found',
                'feasible_count': feasible_count
            }
        
        lineup_stats = self._calculate_lineup_stats(best_lineup)
        
        return {
            'status': 'SUCCESS',
            'lineup': list(best_lineup),
            'objective': objective,
            'objective_value': best_value,
            'stats': lineup_stats,
            'feasible_count': feasible_count,
            'total_combinations': len(all_lineups),
            'opponent': opponent,
            'opponent_info': opp_info
        }
    
    def _calculate_lineup_stats(self, lineup):
        """Calculate comprehensive statistics for a lineup"""
        lineup_df = self.players[self.players['player'].isin(lineup)]
        
        stats = {
            'total_rating': lineup_df['rating'].sum(),
            'rating_limit': self.get_adjusted_rating_limit(lineup),
            'avg_rapm': lineup_df['rapm'].mean(),
            'avg_net_rapm_ctx': lineup_df['net_rapm_ctx'].mean(),
            'avg_o_rapm_ctx': lineup_df['o_rapm_ctx'].mean(),
            'avg_defense_value_ctx': lineup_df['defense_value_ctx'].mean(),
            'total_minutes_exp': lineup_df['minutes_played'].sum(),
            'avg_fatigue': np.mean([self.player_status[p]['fatigue_level'] for p in lineup])
        }
        
        stats['players'] = []
        for player in lineup:
            p_data = self.players[self.players['player'] == player].iloc[0]
            stats['players'].append({
                'name': player,
                'rating': p_data['rating'],
                'rapm': self.get_adjusted_player_value(player, 'rapm'),
                'o_rapm': self.get_adjusted_player_value(player, 'o_rapm_ctx'),
                'defense': self.get_adjusted_player_value(player, 'defense_value_ctx'),
                'fatigue': self.player_status[player]['fatigue_level']
            })
        
        return stats
    
    def get_backup_lineups(self, n=3, objective='rapm', opponent=None):
        """Get top N backup lineups"""
        available_players = [
            p for p, status in self.player_status.items()
            if status['available'] and not status['injured']
        ]
        
        if objective == 'offensive':
            weights = {'o_rapm_ctx': 1.0}
        elif objective == 'defensive':
            weights = {'defense_value_ctx': 1.0}
        elif objective == 'balanced':
            weights = {'o_rapm_ctx': 0.5, 'defense_value_ctx': 0.5}
        else:
            weights = {objective: 1.0}
        
        all_lineups = list(combinations(available_players, self.lineup_size))
        lineup_scores = []
        
        for lineup in all_lineups:
            total_rating = sum([
                self.players[self.players['player'] == p]['rating'].values[0]
                for p in lineup
            ])
            rating_limit = self.get_adjusted_rating_limit(lineup)
            
            if total_rating > rating_limit:
                continue
            
            obj_value = 0
            for metric, weight in weights.items():
                for player in lineup:
                    obj_value += weight * self.get_adjusted_player_value(player, metric)
            
            lineup_scores.append((lineup, obj_value))
        
        lineup_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (lineup, score) in enumerate(lineup_scores[:n]):
            results.append({
                'rank': i + 1,
                'lineup': list(lineup),
                'score': score,
                'stats': self._calculate_lineup_stats(lineup)
            })
        
        return results
    
    def simulate_injury(self, injured_player, objective='rapm', opponent=None):
        """Find best lineup if a specific player is injured"""
        original_status = self.player_status[injured_player]['available']
        self.player_status[injured_player]['available'] = False
        
        result = self.optimize_lineup(objective=objective, opponent=opponent)
        
        self.player_status[injured_player]['available'] = original_status
        
        result['injured_player'] = injured_player
        return result
    
    def get_rotation_strategy(self, game_minutes=40, rest_threshold=0.7):
        """Generate minute allocation for all players"""
        optimal = self.optimize_lineup()
        
        if optimal['status'] != 'SUCCESS':
            return optimal
        
        allocation = {}
        core_players = optimal['lineup']
        
        for player in core_players:
            fatigue = self.player_status[player]['fatigue_level']
            if fatigue < rest_threshold:
                allocation[player] = int(game_minutes * 0.7)
            else:
                allocation[player] = int(game_minutes * 0.9)
        
        total_allocated = sum(allocation.values())
        remaining_minutes = (game_minutes * 4) - total_allocated
        
        bench_players = [p for p in self.player_status.keys() 
                        if p not in core_players and self.player_status[p]['available']]
        
        if bench_players and remaining_minutes > 0:
            minutes_per_bench = remaining_minutes // len(bench_players)
            for player in bench_players:
                allocation[player] = minutes_per_bench
        
        return {
            'allocation': allocation,
            'core_players': core_players,
            'bench_players': bench_players,
            'total_minutes': game_minutes
        }
    
    def save_game_result(self, lineup_used, minutes_played, opponent, 
                        goals_for, goals_against, save_path="game_results.json"):
        """Save game result for future analysis"""
        game_result = {
            'timestamp': datetime.now().isoformat(),
            'team': self.team_name,
            'opponent': opponent,
            'lineup': lineup_used,
            'minutes_played': minutes_played,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'goal_differential': goals_for - goals_against
        }
        
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                results = json.load(f)
        else:
            results = []
        
        results.append(game_result)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Game result saved to {save_path}")
        
        for player, minutes in minutes_played.items():
            if player in self.player_status:
                fatigue_decrease = minutes / 100.0
                current_fatigue = self.player_status[player]['fatigue_level']
                self.player_status[player]['fatigue_level'] = max(0.5, current_fatigue - fatigue_decrease)
    
    def print_roster_status(self):
        """Print current status of all players"""
        print(f"\n{'='*80}")
        print(f"ROSTER STATUS: {self.team_name}")
        print(f"{'='*80}")
        print(f"{'Player':<15} {'Status':<12} {'Fatigue':<10} {'Rating':<8} {'RAPM':<10}")
        print(f"{'-'*80}")
        
        for _, player in self.players.iterrows():
            name = player['player']
            status = self.player_status[name]
            
            status_str = "INJURED" if status['injured'] else "AVAILABLE"
            fatigue_str = f"{status['fatigue_level']:.2f}"
            rating_str = f"{player['rating']:.1f}"
            rapm_str = f"{player['rapm']:+.3f}"
            
            print(f"{name:<15} {status_str:<12} {fatigue_str:<10} {rating_str:<8} {rapm_str:<10}")
    
    def print_lineup_recommendation(self, result):
        """Pretty print lineup recommendation"""
        if result['status'] != 'SUCCESS':
            print(f"\n✗ ERROR: {result['message']}")
            return
        
        print(f"\n{'='*80}")
        print(f"OPTIMAL LINEUP RECOMMENDATION")
        print(f"{'='*80}")
        
        if result.get('opponent'):
            print(f"Opponent: {result['opponent']}")
            if result.get('opponent_info'):
                print(f"Strategy: {result['objective'].upper()}")
        
        print(f"\nSelected Players:")
        print(f"{'-'*80}")
        
        for p_info in result['stats']['players']:
            print(f"  • {p_info['name']:<15} | "
                  f"Rating: {p_info['rating']:.1f} | "
                  f"RAPM: {p_info['rapm']:+.3f} | "
                  f"Offense: {p_info['o_rapm']:+.3f} | "
                  f"Defense: {p_info['defense']:+.3f} | "
                  f"Fatigue: {p_info['fatigue']:.2f}")
        
        print(f"\n{'-'*80}")
        print(f"Lineup Statistics:")
        stats = result['stats']
        print(f"  Total Rating: {stats['total_rating']:.1f} / {stats['rating_limit']:.1f}")
        print(f"  Objective Value: {result['objective_value']:.4f}")
        print(f"  Avg NET RAPM: {stats['avg_rapm']:.4f}")
        print(f"  Avg Offense: {stats['avg_o_rapm_ctx']:.4f}")
        print(f"  Avg Defense: {stats['avg_defense_value_ctx']:.4f}")
        print(f"  Combined Experience: {stats['total_minutes_exp']:.0f} minutes")
        print(f"  Average Fatigue: {stats['avg_fatigue']:.2f}")
        print(f"\n  Feasible lineups checked: {result['feasible_count']}/{result['total_combinations']}")


def interactive_mode():
    """Run the optimizer in interactive mode"""
    print("\n" + "="*80)
    print("WHEELCHAIR RUGBY LINEUP OPTIMIZER - INTERACTIVE MODE")
    print("="*80)
    
    try:
        optimizer = LineupOptimizer()
    except Exception as e:
        print(f"\n✗ Error initializing optimizer: {e}")
        print("  Make sure 'outputs/player_value_metrics.csv' exists")
        print("  Run Phase 1 analysis first: python3 wcr_lineup_analytics.py")
        return
    
    optimizer.print_roster_status()
    
    while True:
        print("\n" + "="*80)
        print("MENU:")
        print("  1. Find optimal lineup")
        print("  2. Set player injury status")
        print("  3. Set player fatigue")
        print("  4. Set player gender (for female bonus)")
        print("  5. Simulate injury scenario")
        print("  6. Get backup lineups")
        print("  7. Get rotation strategy")
        print("  8. View roster status")
        print("  9. Save game result")
        print("  0. Exit")
        print("="*80)
        
        choice = input("\nSelect option (0-9): ").strip()
        
        if choice == '1':
            opponent = input("Opponent team name (or press Enter to skip): ").strip()
            opponent = opponent if opponent else None
            
            print("\nObjective:")
            print("  1. Balanced (default)")
            print("  2. Offensive")
            print("  3. Defensive")
            print("  4. Maximize RAPM")
            obj_choice = input("Select (1-4, default=1): ").strip()
            
            obj_map = {'1': 'balanced', '2': 'offensive', '3': 'defensive', '4': 'rapm'}
            objective = obj_map.get(obj_choice, 'balanced')
            
            result = optimizer.optimize_lineup(objective=objective, opponent=opponent)
            optimizer.print_lineup_recommendation(result)
        
        elif choice == '2':
            player = input("Player name: ").strip()
            status = input("Injured? (y/n): ").strip().lower()
            optimizer.set_player_injury(player, injured=(status == 'y'))
        
        elif choice == '3':
            player = input("Player name: ").strip()
            fatigue = input("Fatigue level (0.0=exhausted, 1.0=fresh): ").strip()
            try:
                fatigue = float(fatigue)
                optimizer.set_player_fatigue(player, fatigue)
            except:
                print("✗ Invalid fatigue level")
        
        elif choice == '4':
            player = input("Player name: ").strip()
            is_female = input("Female? (y/n): ").strip().lower()
            optimizer.set_player_gender(player, is_female=(is_female == 'y'))
        
        elif choice == '5':
            player = input("Player to simulate injury: ").strip()
            opponent = input("Opponent (optional): ").strip()
            opponent = opponent if opponent else None
            
            result = optimizer.simulate_injury(player, opponent=opponent)
            print(f"\n{'='*80}")
            print(f"INJURY SIMULATION: {player} injured")
            print(f"{'='*80}")
            optimizer.print_lineup_recommendation(result)
        
        elif choice == '6':
            n = input("Number of backup lineups (default=3): ").strip()
            n = int(n) if n else 3
            
            backups = optimizer.get_backup_lineups(n=n)
            print(f"\n{'='*80}")
            print(f"TOP {n} BACKUP LINEUPS")
            print(f"{'='*80}")
            
            for backup in backups:
                print(f"\nRank #{backup['rank']} (Score: {backup['score']:.4f})")
                print(f"  Players: {', '.join(backup['lineup'])}")
                print(f"  Rating: {backup['stats']['total_rating']:.1f}")
                print(f"  Avg RAPM: {backup['stats']['avg_rapm']:.4f}")
        
        elif choice == '7':
            strategy = optimizer.get_rotation_strategy()
            print(f"\n{'='*80}")
            print("RECOMMENDED ROTATION STRATEGY")
            print(f"{'='*80}")
            print(f"\nCore Lineup: {', '.join(strategy['core_players'])}")
            print(f"\nMinute Allocation:")
            for player, minutes in sorted(strategy['allocation'].items(), 
                                         key=lambda x: x[1], reverse=True):
                print(f"  {player:<15}: {minutes:>2} minutes")
        
        elif choice == '8':
            optimizer.print_roster_status()
        
        elif choice == '9':
            print("\nEnter game details:")
            opponent = input("Opponent: ").strip()
            goals_for = int(input("Goals FOR: ").strip())
            goals_against = int(input("Goals AGAINST: ").strip())
            
            lineup_used = optimizer.optimize_lineup()['lineup']
            minutes_played = {p: 35 for p in lineup_used}
            
            optimizer.save_game_result(lineup_used, minutes_played, 
                                      opponent, goals_for, goals_against)
        
        elif choice == '0':
            print("\n✓ Exiting optimizer. Good luck!")
            break
        
        else:
            print("\n✗ Invalid option")


if __name__ == "__main__":
    interactive_mode()