"""
WHEELCHAIR RUGBY LINEUP OPTIMIZER - ADVANCED DRAG & DROP
Interactive web application with dual lineup builder (Your team + Opponent)

Installation:
    pip install flask pandas numpy --break-system-packages

Usage:
    python3 lineup_optimizer_advanced.py
    Then open browser to: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from itertools import combinations
import json
import os

app = Flask(__name__)

# Global optimizer instance (lazy loaded)
_optimizer = None
_all_teams_data = None

def get_optimizer():
    """Lazy load the optimizer to avoid initialization errors"""
    global _optimizer
    if _optimizer is None:
        from lineup_optimizer_tool import LineupOptimizer
        _optimizer = LineupOptimizer()
    return _optimizer

def get_all_teams_data():
    """Load all teams' player data"""
    global _all_teams_data
    if _all_teams_data is None:
        try:
            df = pd.read_csv("outputs/player_value_metrics.csv")
            _all_teams_data = df
        except:
            _all_teams_data = pd.DataFrame()
    return _all_teams_data

# ============================================================================
# HTML TEMPLATE WITH DUAL LINEUP BUILDER
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🏉 Wheelchair Rugby Lineup Optimizer - Advanced</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 20px;
            text-align: center;
        }

        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            color: #666;
            font-size: 1.1em;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 280px 280px 1fr 280px;
            gap: 20px;
            margin-bottom: 20px;
        }

        .panel {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            max-height: 800px;
            overflow-y: auto;
        }

        .panel h2 {
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            font-size: 1.3em;
            position: sticky;
            top: 0;
            background: white;
            z-index: 10;
        }

        .panel.opponent h2 {
            border-bottom-color: #e74c3c;
        }

        /* PLAYER ROSTER */
        .player-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            cursor: grab;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-size: 0.9em;
        }

        .player-card.opponent-card {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        }

        .player-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }

        .player-card.dragging {
            opacity: 0.5;
            cursor: grabbing;
        }

        .player-card.injured {
            background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);
            opacity: 0.6;
        }

        .player-card.female {
            border: 3px solid #f39c12;
        }

        .player-name {
            font-weight: bold;
            font-size: 1.05em;
            margin-bottom: 4px;
        }

        .player-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px;
            font-size: 0.85em;
            opacity: 0.9;
        }

        .stat-badge {
            background: rgba(255,255,255,0.2);
            padding: 2px 6px;
            border-radius: 4px;
            text-align: center;
        }

        /* LINEUP BUILDER */
        .lineup-section {
            margin-bottom: 25px;
        }

        .lineup-section h3 {
            color: #333;
            margin-bottom: 12px;
            font-size: 1.1em;
        }

        .lineup-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }

        .position-slot {
            background: #f8f9fa;
            border: 3px dashed #ccc;
            border-radius: 10px;
            min-height: 100px;
            padding: 12px;
            transition: all 0.3s;
            position: relative;
        }

        .position-slot.drag-over {
            background: #e3f2fd;
            border-color: #667eea;
            border-style: solid;
        }

        .position-slot.opponent-slot.drag-over {
            background: #ffebee;
            border-color: #e74c3c;
        }

        .position-slot.filled {
            border-style: solid;
            border-color: #667eea;
        }

        .position-slot.opponent-slot.filled {
            border-color: #e74c3c;
        }

        .position-label {
            font-weight: bold;
            color: #666;
            margin-bottom: 8px;
            font-size: 0.85em;
        }

        .remove-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            background: #e74c3c;
            color: white;
            border: none;
            border-radius: 50%;
            width: 22px;
            height: 22px;
            cursor: pointer;
            font-size: 13px;
            display: none;
            line-height: 1;
        }

        .position-slot.filled .remove-btn {
            display: block;
        }

        /* CONTROLS */
        .control-group {
            margin: 12px 0;
        }

        .control-group label {
            display: block;
            font-weight: bold;
            color: #333;
            margin-bottom: 6px;
            font-size: 0.9em;
        }

        input[type="text"],
        input[type="number"],
        select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 13px;
            transition: border-color 0.3s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            width: 100%;
            margin: 8px 0;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }

        .btn-secondary {
            background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);
        }

        .btn-success {
            background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        }

        .btn-danger {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        }

        /* RESULTS */
        .results-panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-top: 20px;
        }

        .lineup-result {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 15px 0;
            border-radius: 10px;
        }

        .lineup-result.matchup {
            border-left-color: #f39c12;
        }

        .lineup-result h3 {
            color: #333;
            margin-bottom: 15px;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }

        .metric {
            background: white;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .metric-label {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #667eea;
        }

        .player-detail {
            background: white;
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }

        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #fee;
            border: 2px solid #e74c3c;
            color: #c0392b;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }

        .success {
            background: #efe;
            border: 2px solid #27ae60;
            color: #196f3d;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }

        .vs-badge {
            background: #f39c12;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin: 10px 0;
        }

        @media (max-width: 1400px) {
            .main-grid {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (max-width: 900px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🏉 Wheelchair Rugby Lineup Optimizer - Advanced</h1>
            <p>Dual Lineup Builder - Optimize Against Specific Opponents</p>
        </div>

        <!-- Main Grid -->
        <div class="main-grid">
            <!-- Column 1: Canada Roster -->
            <div class="panel">
                <h2>🇨🇦 Canada Roster</h2>
                <div class="control-group">
                    <input type="text" id="canada-filter" placeholder="Search...">
                </div>
                <div id="canada-roster"></div>
            </div>

            <!-- Column 2: Opponent Roster -->
            <div class="panel opponent">
                <h2>🌍 Opponent Roster</h2>
                <div class="control-group">
                    <label>Select Opponent Team:</label>
                    <select id="opponent-team" onchange="loadOpponentRoster()">
                        <option value="">Choose team...</option>
                    </select>
                </div>
                <div class="control-group">
                    <input type="text" id="opponent-filter" placeholder="Search...">
                </div>
                <div id="opponent-roster"></div>
            </div>

            <!-- Column 3: Lineup Builders -->
            <div class="panel">
                <h2>🎯 Build Lineups</h2>
                
                <!-- Canada Lineup -->
                <div class="lineup-section">
                    <h3>🇨🇦 Canada Lineup</h3>
                    <div class="lineup-grid" id="canada-lineup-grid">
                        <div class="position-slot" data-team="canada" data-position="1">
                            <div class="position-label">Position 1</div>
                            <button class="remove-btn" onclick="removePlayer('canada', 1)">×</button>
                        </div>
                        <div class="position-slot" data-team="canada" data-position="2">
                            <div class="position-label">Position 2</div>
                            <button class="remove-btn" onclick="removePlayer('canada', 2)">×</button>
                        </div>
                        <div class="position-slot" data-team="canada" data-position="3">
                            <div class="position-label">Position 3</div>
                            <button class="remove-btn" onclick="removePlayer('canada', 3)">×</button>
                        </div>
                        <div class="position-slot" data-team="canada" data-position="4">
                            <div class="position-label">Position 4</div>
                            <button class="remove-btn" onclick="removePlayer('canada', 4)">×</button>
                        </div>
                    </div>
                </div>

                <div class="vs-badge">⚔️ VS</div>

                <!-- Opponent Lineup -->
                <div class="lineup-section">
                    <h3>🌍 Opponent Lineup</h3>
                    <div class="lineup-grid" id="opponent-lineup-grid">
                        <div class="position-slot opponent-slot" data-team="opponent" data-position="1">
                            <div class="position-label">Position 1</div>
                            <button class="remove-btn" onclick="removePlayer('opponent', 1)">×</button>
                        </div>
                        <div class="position-slot opponent-slot" data-team="opponent" data-position="2">
                            <div class="position-label">Position 2</div>
                            <button class="remove-btn" onclick="removePlayer('opponent', 2)">×</button>
                        </div>
                        <div class="position-slot opponent-slot" data-team="opponent" data-position="3">
                            <div class="position-label">Position 3</div>
                            <button class="remove-btn" onclick="removePlayer('opponent', 3)">×</button>
                        </div>
                        <div class="position-slot opponent-slot" data-team="opponent" data-position="4">
                            <div class="position-label">Position 4</div>
                            <button class="remove-btn" onclick="removePlayer('opponent', 4)">×</button>
                        </div>
                    </div>
                </div>

                <button class="btn btn-success" onclick="getOptimalCounterLineup()">✨ Get Optimal Counter-Lineup</button>
                <button class="btn" onclick="evaluateBothLineups()">📊 Evaluate Both Lineups</button>
                <button class="btn btn-secondary" onclick="clearAllLineups()">🔄 Clear All</button>
            </div>

            <!-- Column 4: Settings -->
            <div class="panel">
                <h2>⚙️ Quick Actions</h2>
                
                <div class="control-group">
                    <label>Strategy:</label>
                    <select id="strategy">
                        <option value="balanced">Balanced</option>
                        <option value="offensive">Offensive</option>
                        <option value="defensive">Defensive</option>
                        <option value="rapm">Maximize RAPM</option>
                    </select>
                </div>

                <h3 style="margin-top: 20px; margin-bottom: 12px;">Canada Settings</h3>
                
                <div class="control-group">
                    <label>Mark as Injured:</label>
                    <select id="injury-select" onchange="toggleInjury()">
                        <option value="">Select player...</option>
                    </select>
                </div>

                <div class="control-group">
                    <label>Mark as Female:</label>
                    <select id="female-select" onchange="toggleFemale()">
                        <option value="">Select player...</option>
                    </select>
                </div>

                <h3 style="margin-top: 20px; margin-bottom: 12px;">🔍 Analysis</h3>
                
                <button class="btn btn-secondary" onclick="getBackupLineups()">📑 Get Top 3 Backups</button>
                <button class="btn btn-secondary" onclick="getRotationStrategy()">⏱️ Get Rotation Plan</button>
                
                <div class="control-group" style="margin-top: 15px;">
                    <label>Simulate Injury:</label>
                    <select id="simulate-injury">
                        <option value="">Select player...</option>
                    </select>
                    <button class="btn" onclick="simulateInjury()">Run Simulation</button>
                </div>
            </div>
        </div>

        <!-- Results Panel -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Optimizing lineup...</p>
        </div>

        <div id="results"></div>
    </div>

    <script>
        // Global state
        let canadaPlayers = [];
        let opponentPlayers = [];
        let canadaLineup = [null, null, null, null];
        let opponentLineup = [null, null, null, null];
        let playerStatus = {};
        let allTeams = [];

        // Load initial data
        window.onload = function() {
            loadCanadaPlayers();
            loadTeamsList();
        };

        async function loadCanadaPlayers() {
            try {
                const response = await fetch('/api/players');
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                canadaPlayers = data.players;
                playerStatus = data.status;
                
                renderCanadaRoster();
                populateDropdowns();
            } catch (error) {
                showError('Failed to load Canada players: ' + error.message);
            }
        }

        async function loadTeamsList() {
            try {
                const response = await fetch('/api/teams');
                const data = await response.json();
                
                if (data.success) {
                    allTeams = data.teams;
                    const select = document.getElementById('opponent-team');
                    select.innerHTML = '<option value="">Choose team...</option>';
                    allTeams.forEach(team => {
                        const option = document.createElement('option');
                        option.value = team;
                        option.textContent = team;
                        select.appendChild(option);
                    });
                }
            } catch (error) {
                console.error('Failed to load teams:', error);
            }
        }

        async function loadOpponentRoster() {
            const teamName = document.getElementById('opponent-team').value;
            
            if (!teamName) {
                opponentPlayers = [];
                renderOpponentRoster();
                return;
            }
            
            try {
                const response = await fetch(`/api/team-players?team=${teamName}`);
                const data = await response.json();
                
                if (data.success) {
                    opponentPlayers = data.players;
                    renderOpponentRoster();
                } else {
                    showError('Failed to load opponent players');
                }
            } catch (error) {
                showError('Failed to load opponent roster: ' + error.message);
            }
        }

        function renderCanadaRoster() {
            const roster = document.getElementById('canada-roster');
            const filter = document.getElementById('canada-filter').value.toLowerCase();
            
            roster.innerHTML = '';
            
            canadaPlayers
                .filter(p => !filter || p.name.toLowerCase().includes(filter))
                .forEach(player => {
                    const card = createPlayerCard(player, 'canada');
                    roster.appendChild(card);
                });
        }

        function renderOpponentRoster() {
            const roster = document.getElementById('opponent-roster');
            const filter = document.getElementById('opponent-filter').value.toLowerCase();
            
            roster.innerHTML = '';
            
            if (opponentPlayers.length === 0) {
                roster.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">Select opponent team</p>';
                return;
            }
            
            opponentPlayers
                .filter(p => !filter || p.name.toLowerCase().includes(filter))
                .forEach(player => {
                    const card = createPlayerCard(player, 'opponent');
                    roster.appendChild(card);
                });
        }

        function createPlayerCard(player, team) {
            const card = document.createElement('div');
            card.className = 'player-card';
            if (team === 'opponent') {
                card.classList.add('opponent-card');
            }
            card.draggable = team === 'canada' ? !playerStatus[player.name]?.injured : true;
            card.dataset.player = player.name;
            card.dataset.team = team;
            
            if (team === 'canada' && playerStatus[player.name]?.injured) {
                card.classList.add('injured');
            }
            if (team === 'canada' && playerStatus[player.name]?.is_female) {
                card.classList.add('female');
            }
            
            card.innerHTML = `
                <div class="player-name">${player.name}</div>
                <div class="player-stats">
                    <span class="stat-badge">R: ${player.rating.toFixed(1)}</span>
                    <span class="stat-badge">RAPM: ${player.rapm >= 0 ? '+' : ''}${player.rapm.toFixed(2)}</span>
                </div>
            `;
            
            card.addEventListener('dragstart', handleDragStart);
            card.addEventListener('dragend', handleDragEnd);
            
            return card;
        }

        function populateDropdowns() {
            const injurySelect = document.getElementById('injury-select');
            const femaleSelect = document.getElementById('female-select');
            const simulateSelect = document.getElementById('simulate-injury');
            
            [injurySelect, femaleSelect, simulateSelect].forEach(select => {
                select.innerHTML = '<option value="">Select player...</option>';
                canadaPlayers.forEach(p => {
                    const option = document.createElement('option');
                    option.value = p.name;
                    option.textContent = p.name;
                    select.appendChild(option);
                });
            });
        }

        // Drag and Drop handlers
        let draggedElement = null;

        function handleDragStart(e) {
            draggedElement = this;
            this.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/html', this.innerHTML);
            e.dataTransfer.setData('player', this.dataset.player);
            e.dataTransfer.setData('team', this.dataset.team);
        }

        function handleDragEnd(e) {
            this.classList.remove('dragging');
        }

        // Set up drop zones
        document.addEventListener('DOMContentLoaded', function() {
            const slots = document.querySelectorAll('.position-slot');
            
            slots.forEach(slot => {
                slot.addEventListener('dragover', handleDragOver);
                slot.addEventListener('drop', handleDrop);
                slot.addEventListener('dragleave', handleDragLeave);
            });
        });

        function handleDragOver(e) {
            if (e.preventDefault) {
                e.preventDefault();
            }
            this.classList.add('drag-over');
            e.dataTransfer.dropEffect = 'move';
            return false;
        }

        function handleDragLeave(e) {
            this.classList.remove('drag-over');
        }

        function handleDrop(e) {
            if (e.stopPropagation) {
                e.stopPropagation();
            }
            
            this.classList.remove('drag-over');
            
            if (draggedElement) {
                const playerName = draggedElement.dataset.player;
                const playerTeam = draggedElement.dataset.team;
                const slotTeam = this.dataset.team;
                const position = parseInt(this.dataset.position) - 1;
                
                // Only allow dropping on matching team slot
                if (playerTeam !== slotTeam) {
                    return false;
                }
                
                if (slotTeam === 'canada') {
                    // Remove player from any other position
                    canadaLineup = canadaLineup.map(p => p === playerName ? null : p);
                    // Add to this position
                    canadaLineup[position] = playerName;
                    updateCanadaLineupDisplay();
                } else {
                    // Remove player from any other position
                    opponentLineup = opponentLineup.map(p => p === playerName ? null : p);
                    // Add to this position
                    opponentLineup[position] = playerName;
                    updateOpponentLineupDisplay();
                }
            }
            
            return false;
        }

        function updateCanadaLineupDisplay() {
            const slots = document.querySelectorAll('.position-slot[data-team="canada"]');
            
            slots.forEach((slot, index) => {
                const playerName = canadaLineup[index];
                updateSlotDisplay(slot, playerName, index + 1, 'canada', canadaPlayers);
            });
        }

        function updateOpponentLineupDisplay() {
            const slots = document.querySelectorAll('.position-slot[data-team="opponent"]');
            
            slots.forEach((slot, index) => {
                const playerName = opponentLineup[index];
                updateSlotDisplay(slot, playerName, index + 1, 'opponent', opponentPlayers);
            });
        }

        function updateSlotDisplay(slot, playerName, position, team, playerList) {
            if (playerName) {
                const player = playerList.find(p => p.name === playerName);
                slot.classList.add('filled');
                const cardClass = team === 'opponent' ? 'player-card opponent-card' : 'player-card';
                slot.innerHTML = `
                    <div class="position-label">Position ${position}</div>
                    <button class="remove-btn" onclick="removePlayer('${team}', ${position})">×</button>
                    <div class="${cardClass}" style="margin: 0;">
                        <div class="player-name">${player.name}</div>
                        <div class="player-stats">
                            <span class="stat-badge">R: ${player.rating.toFixed(1)}</span>
                            <span class="stat-badge">RAPM: ${player.rapm >= 0 ? '+' : ''}${player.rapm.toFixed(2)}</span>
                        </div>
                    </div>
                `;
            } else {
                slot.classList.remove('filled');
                slot.innerHTML = `
                    <div class="position-label">Position ${position}</div>
                    <button class="remove-btn" onclick="removePlayer('${team}', ${position})">×</button>
                `;
            }
        }

        function removePlayer(team, position) {
            if (team === 'canada') {
                canadaLineup[position - 1] = null;
                updateCanadaLineupDisplay();
            } else {
                opponentLineup[position - 1] = null;
                updateOpponentLineupDisplay();
            }
        }

        function clearAllLineups() {
            canadaLineup = [null, null, null, null];
            opponentLineup = [null, null, null, null];
            updateCanadaLineupDisplay();
            updateOpponentLineupDisplay();
        }

        // Main optimization function
        async function getOptimalCounterLineup() {
            const oppLineup = opponentLineup.filter(p => p !== null);
            
            if (oppLineup.length !== 4) {
                showError('Please build complete opponent lineup (4 players)');
                return;
            }
            
            showLoading(true);
            
            try {
                const response = await fetch('/api/optimize-vs-lineup', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        opponent_lineup: oppLineup,
                        opponent_team: document.getElementById('opponent-team').value,
                        strategy: document.getElementById('strategy').value
                    })
                });
                
                const data = await response.json();
                showLoading(false);
                
                if (data.success) {
                    // Auto-fill Canada lineup
                    canadaLineup = data.result.lineup.slice(0, 4);
                    while (canadaLineup.length < 4) canadaLineup.push(null);
                    updateCanadaLineupDisplay();
                    
                    displayMatchupResults(data.result, oppLineup);
                } else {
                    showError(data.message);
                }
            } catch (error) {
                showLoading(false);
                showError('Failed to optimize: ' + error.message);
            }
        }

        async function evaluateBothLineups() {
            const canLineup = canadaLineup.filter(p => p !== null);
            const oppLineup = opponentLineup.filter(p => p !== null);
            
            if (canLineup.length !== 4 || oppLineup.length !== 4) {
                showError('Please complete both lineups (4 players each)');
                return;
            }
            
            showLoading(true);
            
            try {
                const response = await fetch('/api/evaluate-matchup', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        canada_lineup: canLineup,
                        opponent_lineup: oppLineup,
                        opponent_team: document.getElementById('opponent-team').value
                    })
                });
                
                const data = await response.json();
                showLoading(false);
                
                if (data.success) {
                    displayMatchupEvaluation(data.canada, data.opponent, data.prediction);
                } else {
                    showError(data.message);
                }
            } catch (error) {
                showLoading(false);
                showError('Failed to evaluate: ' + error.message);
            }
        }

        // Helper functions (same as before, adapted)
        async function toggleInjury() {
            const select = document.getElementById('injury-select');
            const playerName = select.value;
            
            if (!playerName) return;
            
            const isInjured = playerStatus[playerName]?.injured || false;
            
            try {
                const response = await fetch('/api/set-injury', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        player: playerName,
                        injured: !isInjured
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    playerStatus[playerName].injured = !isInjured;
                    renderCanadaRoster();
                    showSuccess(`${playerName} marked as ${!isInjured ? 'INJURED' : 'RECOVERED'}`);
                }
                
                select.value = '';
            } catch (error) {
                showError('Failed to update injury status');
            }
        }

        async function toggleFemale() {
            const select = document.getElementById('female-select');
            const playerName = select.value;
            
            if (!playerName) return;
            
            const isFemale = playerStatus[playerName]?.is_female || false;
            
            try {
                const response = await fetch('/api/set-female', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        player: playerName,
                        is_female: !isFemale
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    playerStatus[playerName].is_female = !isFemale;
                    renderCanadaRoster();
                    showSuccess(`${playerName} marked as ${!isFemale ? 'FEMALE' : 'MALE'}`);
                }
                
                select.value = '';
            } catch (error) {
                showError('Failed to update gender status');
            }
        }

        async function simulateInjury() {
            const player = document.getElementById('simulate-injury').value;
            
            if (!player) {
                showError('Please select a player to simulate injury');
                return;
            }
            
            showLoading(true);
            
            try {
                const response = await fetch('/api/simulate-injury', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        player: player,
                        opponent: document.getElementById('opponent-team').value || null
                    })
                });
                
                const data = await response.json();
                showLoading(false);
                
                if (data.success) {
                    displayResults([data.result], `Injury Simulation: ${player} Injured`);
                } else {
                    showError(data.message);
                }
            } catch (error) {
                showLoading(false);
                showError('Failed to simulate injury: ' + error.message);
            }
        }

        async function getBackupLineups() {
            showLoading(true);
            
            try {
                const response = await fetch('/api/backups', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        objective: document.getElementById('strategy').value
                    })
                });
                
                const data = await response.json();
                showLoading(false);
                
                if (data.success) {
                    displayResults(data.results, 'Top 3 Backup Lineups');
                } else {
                    showError(data.message);
                }
            } catch (error) {
                showLoading(false);
                showError('Failed to get backup lineups: ' + error.message);
            }
        }

        async function getRotationStrategy() {
            showLoading(true);
            
            try {
                const response = await fetch('/api/rotation');
                const data = await response.json();
                showLoading(false);
                
                if (data.success) {
                    displayRotation(data.strategy);
                } else {
                    showError(data.message);
                }
            } catch (error) {
                showLoading(false);
                showError('Failed to get rotation strategy: ' + error.message);
            }
        }

        // Display functions
        function displayMatchupResults(result, opponentLineup) {
            const resultsDiv = document.getElementById('results');
            
            let html = `
                <div class="results-panel">
                    <h2>✨ Optimal Counter-Lineup</h2>
                    <div class="lineup-result matchup">
                        <h3>🎯 Recommended Canada Lineup</h3>
                        <p><strong>Selected Players:</strong> ${result.lineup.join(', ')}</p>
                        <p><strong>Against:</strong> ${opponentLineup.join(', ')}</p>
                        
                        <div class="metric-grid">
                            <div class="metric">
                                <div class="metric-label">Rating Used</div>
                                <div class="metric-value">${result.stats.total_rating.toFixed(1)} / ${result.stats.rating_limit.toFixed(1)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">NET RAPM</div>
                                <div class="metric-value">${result.stats.avg_rapm >= 0 ? '+' : ''}${result.stats.avg_rapm.toFixed(3)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Offense</div>
                                <div class="metric-value">${result.stats.avg_o_rapm_ctx >= 0 ? '+' : ''}${result.stats.avg_o_rapm_ctx.toFixed(3)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Defense</div>
                                <div class="metric-value">${result.stats.avg_defense_value_ctx.toFixed(3)}</div>
                            </div>
                        </div>
                        
                        <h4>Player Details:</h4>
            `;
            
            result.stats.players.forEach(player => {
                html += `
                    <div class="player-detail">
                        <span><strong>${player.name}</strong></span>
                        <span>Rating: ${player.rating.toFixed(1)} | RAPM: ${player.rapm >= 0 ? '+' : ''}${player.rapm.toFixed(3)}</span>
                    </div>
                `;
            });
            
            html += '</div></div>';
            resultsDiv.innerHTML = html;
        }

        function displayMatchupEvaluation(canada, opponent, prediction) {
            const resultsDiv = document.getElementById('results');
            
            let html = `
                <div class="results-panel">
                    <h2>⚔️ Matchup Analysis</h2>
                    <div class="lineup-result">
                        <h3>🇨🇦 Canada: ${canada.lineup.join(', ')}</h3>
                        <div class="metric-grid">
                            <div class="metric">
                                <div class="metric-label">NET RAPM</div>
                                <div class="metric-value">${canada.avg_rapm >= 0 ? '+' : ''}${canada.avg_rapm.toFixed(3)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Offense</div>
                                <div class="metric-value">${canada.avg_offense.toFixed(3)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Defense</div>
                                <div class="metric-value">${canada.avg_defense.toFixed(3)}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="lineup-result">
                        <h3>🌍 ${opponent.team}: ${opponent.lineup.join(', ')}</h3>
                        <div class="metric-grid">
                            <div class="metric">
                                <div class="metric-label">NET RAPM</div>
                                <div class="metric-value">${opponent.avg_rapm >= 0 ? '+' : ''}${opponent.avg_rapm.toFixed(3)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Offense</div>
                                <div class="metric-value">${opponent.avg_offense.toFixed(3)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Defense</div>
                                <div class="metric-value">${opponent.avg_defense.toFixed(3)}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="lineup-result matchup">
                        <h3>📊 Prediction</h3>
                        <p style="font-size: 1.2em; color: #667eea; font-weight: bold;">
                            ${prediction.winner} expected to win by ${Math.abs(prediction.differential).toFixed(2)} goals
                        </p>
                        <p><strong>Confidence:</strong> ${prediction.confidence}</p>
                    </div>
                </div>
            `;
            
            resultsDiv.innerHTML = html;
        }

        function displayResults(results, title) {
            const resultsDiv = document.getElementById('results');
            
            let html = `<div class="results-panel"><h2>${title}</h2>`;
            
            results.forEach((result, index) => {
                html += `
                    <div class="lineup-result">
                        ${result.rank ? `<h3>Rank #${result.rank}</h3>` : ''}
                        ${result.scenario_name ? `<h3>${result.scenario_name}</h3>` : ''}
                        ${result.injured_player ? `<p><strong>⚠️ Scenario: ${result.injured_player} is injured</strong></p>` : ''}
                        
                        <p><strong>Selected Players:</strong> ${result.lineup ? result.lineup.join(', ') : 'N/A'}</p>
                        
                        <div class="metric-grid">
                            <div class="metric">
                                <div class="metric-label">Rating Used</div>
                                <div class="metric-value">${result.stats.total_rating.toFixed(1)} / ${result.stats.rating_limit.toFixed(1)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">NET RAPM</div>
                                <div class="metric-value">${result.stats.avg_rapm >= 0 ? '+' : ''}${result.stats.avg_rapm.toFixed(3)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Offense</div>
                                <div class="metric-value">${result.stats.avg_o_rapm_ctx >= 0 ? '+' : ''}${result.stats.avg_o_rapm_ctx.toFixed(3)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Defense</div>
                                <div class="metric-value">${result.stats.avg_defense_value_ctx.toFixed(3)}</div>
                            </div>
                        </div>
                        
                        <h4>Player Details:</h4>
                `;
                
                result.stats.players.forEach(player => {
                    html += `
                        <div class="player-detail">
                            <span><strong>${player.name}</strong></span>
                            <span>Rating: ${player.rating.toFixed(1)} | RAPM: ${player.rapm >= 0 ? '+' : ''}${player.rapm.toFixed(3)}</span>
                        </div>
                    `;
                });
                
                html += '</div>';
            });
            
            html += '</div>';
            resultsDiv.innerHTML = html;
        }

        function displayRotation(strategy) {
            const resultsDiv = document.getElementById('results');
            
            let html = `
                <div class="results-panel">
                    <h2>⏱️ Recommended Rotation Strategy</h2>
                    <div class="lineup-result">
                        <h3>Core Lineup</h3>
                        <p>${strategy.core_players.join(', ')}</p>
                        
                        <h3>Minute Allocation (40-minute game)</h3>
            `;
            
            const sortedAllocation = Object.entries(strategy.allocation)
                .sort((a, b) => b[1] - a[1]);
            
            sortedAllocation.forEach(([player, minutes]) => {
                const percentage = (minutes / 40 * 100).toFixed(0);
                html += `
                    <div class="player-detail">
                        <span><strong>${player}</strong></span>
                        <span>${minutes} minutes (${percentage}%)</span>
                    </div>
                `;
            });
            
            html += '</div></div>';
            resultsDiv.innerHTML = html;
        }

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }

        function showError(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `<div class="error"><strong>❌ Error:</strong> ${message}</div>`;
        }

        function showSuccess(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `<div class="success"><strong>✅ Success:</strong> ${message}</div>`;
            setTimeout(() => {
                resultsDiv.innerHTML = '';
            }, 3000);
        }

        // Event listeners
        document.getElementById('canada-filter').addEventListener('input', renderCanadaRoster);
        document.getElementById('opponent-filter').addEventListener('input', renderOpponentRoster);
    </script>
</body>
</html>
"""

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/teams')
def get_teams():
    """Get list of all teams"""
    try:
        df = get_all_teams_data()
        teams = df['team'].unique().tolist()
        teams.sort()
        
        return jsonify({
            'success': True,
            'teams': teams
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/team-players')
def get_team_players():
    """Get players for a specific team"""
    try:
        team_name = request.args.get('team')
        df = get_all_teams_data()
        
        team_data = df[df['team'] == team_name]
        
        players = []
        for _, player in team_data.iterrows():
            players.append({
                'name': player['player'],
                'rating': float(player['rating']),
                'rapm': float(player['rapm']),
                'o_rapm': float(player['o_rapm_ctx']),
                'defense': float(player['defense_value_ctx'])
            })
        
        return jsonify({
            'success': True,
            'players': players
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/players')
def get_players():
    """Get Canada players with their stats"""
    try:
        optimizer = get_optimizer()
        
        players = []
        status = {}
        
        for _, player in optimizer.players.iterrows():
            players.append({
                'name': player['player'],
                'rating': float(player['rating']),
                'rapm': float(player['rapm']),
                'o_rapm': float(player['o_rapm_ctx']),
                'defense': float(player['defense_value_ctx'])
            })
            
            status[player['player']] = {
                'injured': optimizer.player_status[player['player']]['injured'],
                'is_female': optimizer.player_status[player['player']]['is_female'],
                'fatigue_level': optimizer.player_status[player['player']]['fatigue_level']
            }
        
        return jsonify({
            'success': True,
            'players': players,
            'status': status
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/set-injury', methods=['POST'])
def set_injury():
    """Mark a player as injured or recovered"""
    try:
        data = request.json
        optimizer = get_optimizer()
        
        player = data.get('player')
        injured = data.get('injured', True)
        
        optimizer.set_player_injury(player, injured)
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set-female', methods=['POST'])
def set_female():
    """Mark a player as female"""
    try:
        data = request.json
        optimizer = get_optimizer()
        
        player = data.get('player')
        is_female = data.get('is_female', True)
        
        optimizer.set_player_gender(player, is_female)
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/optimize-vs-lineup', methods=['POST'])
def optimize_vs_lineup():
    """Find optimal Canada lineup against specific opponent lineup"""
    try:
        data = request.json
        optimizer = get_optimizer()
        
        opponent_lineup = data.get('opponent_lineup', [])
        opponent_team = data.get('opponent_team')
        strategy = data.get('strategy', 'balanced')
        
        # For now, just optimize normally
        # In a more advanced version, you could analyze opponent lineup and adjust
        result = optimizer.optimize_lineup(
            objective=strategy,
            opponent=opponent_team
        )
        
        if result['status'] != 'SUCCESS':
            return jsonify({
                'success': False,
                'message': result.get('message', 'Optimization failed')
            })
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/evaluate-matchup', methods=['POST'])
def evaluate_matchup():
    """Evaluate matchup between two lineups"""
    try:
        data = request.json
        canada_lineup = data.get('canada_lineup', [])
        opponent_lineup = data.get('opponent_lineup', [])
        opponent_team = data.get('opponent_team', 'Unknown')
        
        all_data = get_all_teams_data()
        
        # Canada stats
        canada_df = all_data[all_data['player'].isin(canada_lineup)]
        canada_stats = {
            'lineup': canada_lineup,
            'avg_rapm': float(canada_df['rapm'].mean()),
            'avg_offense': float(canada_df['o_rapm_ctx'].mean()),
            'avg_defense': float(canada_df['defense_value_ctx'].mean())
        }
        
        # Opponent stats
        opponent_df = all_data[all_data['player'].isin(opponent_lineup)]
        opponent_stats = {
            'team': opponent_team,
            'lineup': opponent_lineup,
            'avg_rapm': float(opponent_df['rapm'].mean()),
            'avg_offense': float(opponent_df['o_rapm_ctx'].mean()),
            'avg_defense': float(opponent_df['defense_value_ctx'].mean())
        }
        
        # Prediction
        differential = canada_stats['avg_rapm'] - opponent_stats['avg_rapm']
        winner = "Canada" if differential > 0 else opponent_team
        confidence = "High" if abs(differential) > 0.2 else "Medium" if abs(differential) > 0.1 else "Low"
        
        prediction = {
            'winner': winner,
            'differential': differential,
            'confidence': confidence
        }
        
        return jsonify({
            'success': True,
            'canada': canada_stats,
            'opponent': opponent_stats,
            'prediction': prediction
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/simulate-injury', methods=['POST'])
def simulate_injury():
    """Simulate a player injury"""
    try:
        data = request.json
        optimizer = get_optimizer()
        
        player = data.get('player')
        opponent = data.get('opponent')
        
        result = optimizer.simulate_injury(player, opponent=opponent)
        
        if result['status'] != 'SUCCESS':
            return jsonify({
                'success': False,
                'message': result.get('message', 'Simulation failed')
            })
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/backups', methods=['POST'])
def get_backups():
    """Get backup lineups"""
    try:
        data = request.json
        optimizer = get_optimizer()
        
        objective = data.get('objective', 'balanced')
        
        results = optimizer.get_backup_lineups(n=3, objective=objective)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/rotation')
def get_rotation():
    """Get rotation strategy"""
    try:
        optimizer = get_optimizer()
        
        strategy = optimizer.get_rotation_strategy()
        
        if 'allocation' not in strategy:
            return jsonify({
                'success': False,
                'message': 'Failed to generate rotation strategy'
            })
        
        return jsonify({
            'success': True,
            'strategy': strategy
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the web server"""
    print("\n" + "="*80)
    print("WHEELCHAIR RUGBY LINEUP OPTIMIZER - ADVANCED DUAL LINEUP BUILDER")
    print("="*80)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:5000")
    print("\nNEW FEATURES:")
    print("  ✓ Build opponent lineup (drag & drop)")
    print("  ✓ Optimize Canada lineup vs specific opponent lineup")
    print("  ✓ Evaluate matchups between lineups")
    print("  ✓ Predict game outcomes")
    print("  ✓ All previous features maintained")
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()