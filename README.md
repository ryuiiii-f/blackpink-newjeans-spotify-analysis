# BlackPink vs NewJeans: Spotify Data Analysis

A data-driven comparison of BlackPink and NewJeans using Spotify API to see who performs better on the streaming platform.

## Key Findings

**NewJeans dominates Spotify performance:**
- **NewJeans**: 77.6 average popularity (9 super hits ≥75)
- **BLACKPINK**: 68.2 average popularity (0 super hits)
- **Advantage**: +9.4 points (+13.8%) for NewJeans

## Live Dashboard

Check out the [interactive dashboard](https://ryuiiii-f.github.io/blackpink-newjeans-spotify-analysis/results/nj_vs_bp_dashboard.html) with detailed charts and analysis!

## Quick Stats

| Metric | BLACKPINK | NewJeans | Winner |
|--------|-----------|----------|---------|
| Average Popularity | 68.2 | 77.6 | 🏆 NewJeans |
| Super Hits (≥75) | 0 | 9 | 🏆 NewJeans |
| Average Duration | 3.26 min | 2.97 min | 🏆 NewJeans |
| Hit Rate | 0% | 90% | 🏆 NewJeans |

## Main Insight

NewJeans' shorter song format (2.97 min avg) is perfectly optimized for streaming platforms, achieving 90% hit rate vs BLACKPINK's 0%. This shows how 4th gen groups are adapting to the streaming era.

## Tech Stack

| Component | Technologies |
|-----------|-------------|
| **Data Collection** | Spotify Web API, Spotipy |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (KMeans, PCA, t-SNE, Random Forest) |
| **Statistical Analysis** | StandardScaler, Train-Test Split |
| **Visualization** | Matplotlib, Seaborn, Plotly (Express & Graph Objects) |
| **Interactive Dashboard** | Chart.js, HTML/CSS/JavaScript |
| **Model Persistence** | Pickle |

## 📁 Files

- `spotify_data_collector.py` - Spotify API data collection
- `advanced_spotify_analysis.py` - Main analysis script
- `nj_vs_bp_dashboard.html` - Interactive dashboard
- Various CSV data files with track information

## Data Sources

- 20 tracks total (10 per group)
- Spotify popularity scores (0-100)
- Track duration and performance metrics
- Analysis period: 2020-2025

---

**TL;DR**: As of August 2025, NewJeans beats BLACKPINK on Spotify with better optimized tracks for the streaming era. Data shows shorter songs = higher engagement.
