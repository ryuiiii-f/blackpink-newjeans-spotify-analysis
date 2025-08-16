import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 🎵 基于你的真实Spotify数据
blackpink_data = {
    'title': ['How You Like That', 'Pink Venom', 'Jump'],
    'release_year': [2020, 2022, 2024],
    'track_popularity': [73, 74, 73],
    'performance_score': [157.2, 158.4, 157.2],
    'duration_minutes': [3.02, 3.12, 2.75],
    'relative_popularity': [0.84, 0.85, 0.84]  # 基于你的数据计算
}

df = pd.DataFrame(blackpink_data)

# 设置BLACKPINK主题
plt.style.use('dark_background')
pink_colors = ['#ff1493', '#ff69b4', '#ffb6c1', '#ff007f', '#ff6eb4']

# 1. 📈 时间趋势分析
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('🖤💗 BLACKPINK Spotify Data Analysis', fontsize=20, color='#ff69b4', fontweight='bold')

# 人气度趋势
axes[0, 0].plot(df['release_year'], df['track_popularity'], 
                marker='o', linewidth=3, markersize=10, color='#ff1493')
axes[0, 0].set_title('Track Popularity Over Time', color='#ffb6c1', fontsize=14)
axes[0, 0].set_xlabel('Year', color='white')
axes[0, 0].set_ylabel('Popularity', color='white')
axes[0, 0].grid(True, alpha=0.3)

# 表现分数趋势
axes[0, 1].plot(df['release_year'], df['performance_score'], 
                marker='s', linewidth=3, markersize=10, color='#ff69b4')
axes[0, 1].set_title('Performance Score Evolution', color='#ffb6c1', fontsize=14)
axes[0, 1].set_xlabel('Year', color='white')
axes[0, 1].set_ylabel('Performance Score', color='white')
axes[0, 1].grid(True, alpha=0.3)

# 时长变化
axes[1, 0].bar(df['title'], df['duration_minutes'], color=pink_colors[:3], alpha=0.8)
axes[1, 0].set_title('Song Duration Comparison', color='#ffb6c1', fontsize=14)
axes[1, 0].set_ylabel('Duration (minutes)', color='white')
axes[1, 0].tick_params(axis='x', rotation=45)

# 2. 📊 相关性分析
numeric_cols = ['release_year', 'track_popularity', 'performance_score', 'duration_minutes']
correlation_matrix = df[numeric_cols].corr()

# 自定义colormap (黑粉主题)
colors = ['#000000', '#ff1493', '#ff69b4', '#ffb6c1']
n_bins = 100
cmap = sns.blend_palette(colors, n_colors=n_bins, as_cmap=True)

sns.heatmap(correlation_matrix, annot=True, cmap=cmap, center=0,
            square=True, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
axes[1, 1].set_title('Feature Correlation Matrix', color='#ffb6c1', fontsize=14)

plt.tight_layout()
plt.show()

# 3. 🎯 多维散点图分析
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['track_popularity'], df['performance_score'], 
                     s=df['duration_minutes']*100,  # 气泡大小=时长
                     c=df['release_year'], cmap='plasma', 
                     alpha=0.8, edgecolors='white', linewidth=2)

# 添加歌曲标签
for i, txt in enumerate(df['title']):
    plt.annotate(txt, (df['track_popularity'].iloc[i], df['performance_score'].iloc[i]),
                xytext=(5, 5), textcoords='offset points', 
                color='#ffb6c1', fontsize=11, fontweight='bold')

plt.colorbar(scatter, label='Release Year')
plt.xlabel('Track Popularity', fontsize=14, color='white')
plt.ylabel('Performance Score', fontsize=14, color='white')
plt.title('🎵 BLACKPINK Multi-Dimensional Analysis\n(Bubble size = Duration)', 
          fontsize=16, color='#ff69b4', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.gca().set_facecolor('black')
plt.show()

# 4. 📈 洞察报告
print("🔍 BLACKPINK Spotify Data Insights:")
print("=" * 50)

# 趋势分析
popularity_trend = df['track_popularity'].diff().dropna()
if popularity_trend.mean() > 0:
    trend = "上升"
elif popularity_trend.mean() < 0:
    trend = "下降"
else:
    trend = "稳定"

print(f"📈 人气度趋势: {trend}")
print(f"🎵 最高表现: {df.loc[df['performance_score'].idxmax(), 'title']}")
print(f"⏱️ 时长趋势: {'缩短' if df['duration_minutes'].iloc[-1] < df['duration_minutes'].iloc[0] else '延长'}")

# 相关性发现
strong_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.7:  # 强相关性
            strong_correlations.append(
                f"{correlation_matrix.columns[i]} ↔ {correlation_matrix.columns[j]}: {corr_value:.3f}"
            )

if strong_correlations:
    print(f"\n🔗 强相关性发现:")
    for corr in strong_correlations:
        print(f"   {corr}")
else:
    print(f"\n💡 数据建议: 需要更多歌曲数据来发现强相关性模式")

print(f"\n📊 数据总结:")
print(f"   平均人气度: {df['track_popularity'].mean():.1f}")
print(f"   人气度标准差: {df['track_popularity'].std():.1f} (稳定性指标)")
print(f"   最佳表现歌曲: {df.loc[df['performance_score'].idxmax(), 'title']} ({df['performance_score'].max()})")
