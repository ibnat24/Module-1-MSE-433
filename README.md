# Module-1-MSE-433

# Wheelchair Rugby Lineup Optimization - MSE433 Case Study

**Team Canada Decision Support System**

A comprehensive analytics pipeline for optimizing wheelchair rugby lineups using data-driven methods. This project integrates statistical modeling (Ridge Regression RAPM), operations research (Binary Integer Programming), and interactive visualization (Flask web application) to provide coaches with actionable recommendations.


## Project Overview

This case study develops a three-phase analytics solution for Team Canada's wheelchair rugby program:

### **Phase 1: Player Value Quantification**
- Method: Regularized Adjusted Plus-Minus (RAPM) using Ridge Regression
- Input: 7,448 game stints across 144 players from 8 international teams
- Output: Player impact metrics (NET RAPM, O-RAPM, D-RAPM)
- Validation: 5-fold cross-validation, residual analysis, sensitivity testing

### **Phase 2: Lineup Optimization**
- Method: Binary Integer Programming (complete enumeration)
- Constraint: 4 players, total classification rating ≤ 8.0
- Objective: Maximize team NET RAPM
- Output: Optimal lineup configurations for different game scenarios

### **Phase 3: Interactive Decision Support**
- Method: Flask web application with drag-and-drop interface
- Features: Opponent-specific optimization, injury simulation, rotation planning
- Deployment: Local web server (http://localhost:5000)

---

## 💻 System Requirements

### **Required Software**

- **Python**: 3.10 or higher
- **pip**: Latest version (usually included with Python)
- **Web Browser**: Chrome 90+, Firefox 88+, or Safari 14+

### **Required Python Packages**

All dependencies are listed in `requirements.txt`:
- pandas (data processing)
- numpy (numerical operations)
- scikit-learn (machine learning)
- scipy (statistical tests)
- matplotlib (visualizations)
- flask (web application)

---

## 📦 Installation

### **Step 1: Extract Project Files**

```bash
# Extract the ZIP file
unzip wheelchair_rugby_optimization.zip
//cd wheelchair_rugby_optimization/
```

### **Step 2: Install Python Dependencies**

**Standard Installation**

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
wheelchair_rugby_optimization/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/                              # Input data
│   ├── stint_data.csv                 # Game stint records (7,448 rows)
│   └── player_data.csv                # Player ratings (144 players)
│
├── model_validation.py                # Phase 1: RAPM + Statistical Validation
├── wcr_optimization.py                # Phase 2: Lineup Optimization (BIP)
├── wcr_optimization_viz.py            # Phase 3: Visualization Generation
├── lineup_optimizer_tool.py           # Interactive Tool Logic
├── lineup_optimizer_web.py            # Web Application (Interactive Tool)
│
└── outputs/                           # Generated results
    ├── player_value_metrics.csv       # Player RAPM values
    ├── validation_results.txt         # Statistical validation metrics
    ├── optimal_lineups_summary.csv    # Top optimal lineups
    ├── optimal_lineups_detailed.csv   # Detailed lineup analysis
    └── optimization_plots/            # Visualizations (PNG files)
```

---

## 🔄 Reproducing Results

Follow these steps to reproduce all results from the report:

### **Phase 1: Player Value Quantification + Validation**

```bash
python wcr_lineup_analytics.py
python wcr_lineup_analytics.py
python model_validation.py

```

**What it does:**
- Calculates RAPM values for all 144 players
- Performs 5-fold cross-validation
- Runs statistical tests (Shapiro-Wilk, Levene's, Durbin-Watson)
- Compares against baseline models
- Tests sensitivity to regularization parameter α


### **Phase 2: Lineup Optimization**

```bash
python3 wcr_optimization.py
```

**What it does:**
- Loads player RAPM values from Phase 1
- Evaluates all possible 4-player combinations
- Finds optimal lineups under rating constraint (≤8.0)
- Tests multiple optimization objectives (balanced, offensive, defensive)

### **Phase 3: Visualizations**

```bash
python3 wcr_optimization_viz.py
```

**What it does:**
- Generates 5 analytical plots
- Shows tradeoffs, rankings, and comparisons

## 🌐 Running the Web Application

### **Start the Web Server**

```bash
python3 lineup_optimizer_web.py
```

### **Using the Web Application**

**Basic Workflow:**

1. **Select opponent team** from dropdown (e.g., "USA")
2. **Drag 4 opponent players** into their lineup slots
3. **Choose Lineup Strategy** from dropdown (e.g "Offensive")
3. **Click "Get Optimal Counter-Lineup"**
4. **Result**: Best Canada lineup auto-populates

**Advanced Features:**

- **Injury Simulation**: Mark players as injured → see backup recommendations
- **Female Players**: Mark players as female → rating limit increases to 8.5+
- **Manual Lineup Builder**: Drag your own lineups for both teams
- **Head-to-Head Evaluation**: Compare two lineups with win prediction
- **Backup Lineups**: Get top 3 alternative lineups
- **Rotation Planning**: Generate playing time recommendations

