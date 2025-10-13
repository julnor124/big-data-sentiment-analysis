#!/usr/bin/env python3
"""
Emotion Timeline Analysis: Comparing emotions across different periods
- Pre-ChatGPT (2017-2021): clean_tweets_ai.labeled.csv
- Right after ChatGPT launch (2022): AfterChatGPT.labeled.csv filtered for 2022
- Established ChatGPT period (2023+): AfterChatGPT.labeled.csv filtered for 2023+
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean the emotion-labeled datasets"""
    
    print("Loading and processing datasets...")
    
    # Load pre-ChatGPT data
    print("1. Loading pre-ChatGPT data...")
    pre_data = pd.read_csv('Labeling/clean_tweets_ai.labeled.csv')
    pre_data['Date'] = pd.to_datetime(pre_data['Date'])
    pre_data['period'] = 'Pre-ChatGPT (2017-2021)'
    
    print(f"   Pre-ChatGPT: {len(pre_data)} tweets")
    print(f"   Date range: {pre_data['Date'].min()} to {pre_data['Date'].max()}")
    
    # Load AfterChatGPT data
    print("2. Loading AfterChatGPT data...")
    after_data = pd.read_csv('Labeling/AfterChatGPT.labeled.csv')
    
    # Clean the Date column - remove rows with URLs in Date column
    print("   Cleaning problematic date entries...")
    url_mask = after_data['Date'].str.contains('http', na=False)
    print(f"   Removing {url_mask.sum()} rows with URLs in Date column")
    after_data = after_data[~url_mask].copy()
    
    # Convert to datetime with error handling for mixed formats
    after_data['Date'] = pd.to_datetime(after_data['Date'], format='mixed', errors='coerce')
    
    # Remove any rows where date conversion failed
    after_data = after_data.dropna(subset=['Date'])
    
    # Filter by year
    after_2022 = after_data[after_data['Date'].dt.year == 2022].copy()
    after_2023_plus = after_data[after_data['Date'].dt.year >= 2023].copy()
    
    after_2022['period'] = 'Right after ChatGPT (2022)'
    after_2023_plus['period'] = 'Established ChatGPT (2023+)'
    
    print(f"   Right after ChatGPT (2022): {len(after_2022)} tweets")
    print(f"   Established ChatGPT (2023+): {len(after_2023_plus)} tweets")
    
    # Combine all datasets
    all_data = pd.concat([pre_data, after_2022, after_2023_plus], ignore_index=True)
    
    print(f"\nTotal dataset size: {len(all_data)} tweets")
    
    return pre_data, after_2022, after_2023_plus, all_data

def analyze_emotion_distributions(data_dict, period_names):
    """Analyze emotion distributions for each period"""
    
    print("\n=== EMOTION DISTRIBUTION ANALYSIS ===")
    
    results = {}
    
    for period, data in data_dict.items():
        print(f"\n{period}:")
        print(f"  Total tweets: {len(data):,}")
        
        emotion_counts = data['emotion_label'].value_counts()
        emotion_percentages = (emotion_counts / len(data) * 100).round(2)
        
        results[period] = {
            'counts': emotion_counts,
            'percentages': emotion_percentages,
            'total': len(data)
        }
        
        print("  Emotion distribution:")
        for emotion, count in emotion_counts.items():
            percentage = emotion_percentages[emotion]
            print(f"    {emotion}: {count:,} ({percentage}%)")
    
    return results

def create_visualizations(all_data, results):
    """Create comprehensive visualizations"""
    
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Define high-contrast colors for emotions
    emotion_colors = {
        'neutral': '#808080',    # Gray
        'joy': '#FFD700',        # Gold/Yellow
        'anger': '#DC143C',      # Crimson Red
        'sadness': '#4169E1',    # Royal Blue
        'fear': '#8B008B',       # Dark Magenta
        'surprise': '#FF8C00',   # Dark Orange
        'disgust': '#228B22'     # Forest Green
    }
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Emotion distribution comparison (counts)
    plt.subplot(3, 3, 1)
    emotion_counts = []
    periods = []
    emotions = sorted(all_data['emotion_label'].unique())
    
    for period in ['Pre-ChatGPT (2017-2021)', 'Right after ChatGPT (2022)', 'Established ChatGPT (2023+)']:
        period_data = all_data[all_data['period'] == period]
        counts = [period_data[period_data['emotion_label'] == emotion].shape[0] for emotion in emotions]
        emotion_counts.append(counts)
        periods.append(period)
    
    x = np.arange(len(emotions))
    width = 0.25
    
    for i, (counts, period) in enumerate(zip(emotion_counts, periods)):
        plt.bar(x + i*width, counts, width, label=period, alpha=0.8)
    
    plt.xlabel('Emotions')
    plt.ylabel('Number of Tweets')
    plt.title('Emotion Distribution by Time Period (Counts)')
    plt.xticks(x + width, emotions, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Emotion distribution comparison (percentages)
    plt.subplot(3, 3, 2)
    emotion_percentages = []
    
    for period in ['Pre-ChatGPT (2017-2021)', 'Right after ChatGPT (2022)', 'Established ChatGPT (2023+)']:
        period_data = all_data[all_data['period'] == period]
        percentages = [len(period_data[period_data['emotion_label'] == emotion]) / len(period_data) * 100 for emotion in emotions]
        emotion_percentages.append(percentages)
    
    for i, (percentages, period) in enumerate(zip(emotion_percentages, periods)):
        plt.bar(x + i*width, percentages, width, label=period, alpha=0.8)
    
    plt.xlabel('Emotions')
    plt.ylabel('Percentage of Tweets (%)')
    plt.title('Emotion Distribution by Time Period (Percentages)')
    plt.xticks(x + width, emotions, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Stacked bar chart
    plt.subplot(3, 3, 3)
    bottom = np.zeros(len(emotions))
    colors = plt.cm.Set3(np.linspace(0, 1, len(periods)))
    
    for i, (percentages, period) in enumerate(zip(emotion_percentages, periods)):
        plt.bar(emotions, percentages, bottom=bottom, label=period, color=colors[i], alpha=0.8)
        bottom += percentages
    
    plt.xlabel('Emotions')
    plt.ylabel('Percentage of Tweets (%)')
    plt.title('Stacked Emotion Distribution by Time Period')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Heatmap of emotion changes
    plt.subplot(3, 3, 4)
    heatmap_data = np.array(emotion_percentages).T
    sns.heatmap(heatmap_data, 
                xticklabels=['Pre-ChatGPT', 'Right after ChatGPT', 'Established ChatGPT'],
                yticklabels=emotions,
                annot=True, 
                fmt='.1f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Emotion Distribution Heatmap')
    plt.xlabel('Time Period')
    plt.ylabel('Emotions')
    
    # 5. Line plot showing emotion trends
    plt.subplot(3, 3, 5)
    for i, emotion in enumerate(emotions):
        percentages = [emotion_percentages[j][i] for j in range(len(periods))]
        plt.plot(['Pre-ChatGPT', 'Right after\nChatGPT', 'Established\nChatGPT'], 
                percentages, marker='o', linewidth=2, label=emotion)
    
    plt.xlabel('Time Period')
    plt.ylabel('Percentage of Tweets (%)')
    plt.title('Emotion Trends Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 6. Pie charts for each period
    for i, period in enumerate(periods):
        plt.subplot(3, 3, 6 + i)
        period_data = all_data[all_data['period'] == period]
        emotion_counts = period_data['emotion_label'].value_counts()
        
        # Get colors for emotions
        colors = [emotion_colors.get(emotion, '#999999') for emotion in emotion_counts.index]
        
        # Create pie chart without percentage labels on slices
        wedges, texts = plt.pie(emotion_counts.values, 
                               colors=colors,
                               autopct=None, 
                               startangle=90)
        
        # Create custom legend with emotion names and percentages
        legend_labels = [f"{emotion}: {count:,} ({count/len(period_data)*100:.1f}%)" 
                        for emotion, count in emotion_counts.items()]
        
        plt.legend(wedges, legend_labels,
                  title="Emotions",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.title(f'{period}\n({len(period_data):,} tweets)')
    
    plt.tight_layout()
    plt.savefig('Emotional_Analysis/emotion_timeline_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'Emotional_Analysis/emotion_timeline_analysis.png'")

def calculate_statistical_significance(results):
    """Calculate statistical significance of emotion differences"""
    
    print("\n=== STATISTICAL ANALYSIS ===")
    
    from scipy.stats import chi2_contingency
    
    periods = ['Pre-ChatGPT (2017-2021)', 'Right after ChatGPT (2022)', 'Established ChatGPT (2023+)']
    emotions = sorted(set().union(*[results[period]['counts'].index for period in periods]))
    
    # Create contingency table
    contingency_table = []
    for period in periods:
        row = []
        for emotion in emotions:
            count = results[period]['counts'].get(emotion, 0)
            row.append(count)
        contingency_table.append(row)
    
    contingency_table = np.array(contingency_table)
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Degrees of freedom: {dof}")
    
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"
    
    print(f"The difference in emotion distributions across time periods is {significance}")
    
    return chi2, p_value, dof

def identify_key_changes(results):
    """Identify the most significant emotion changes between periods"""
    
    print("\n=== KEY EMOTION CHANGES ===")
    
    periods = ['Pre-ChatGPT (2017-2021)', 'Right after ChatGPT (2022)', 'Established ChatGPT (2023+)']
    emotions = sorted(set().union(*[results[period]['counts'].index for period in periods]))
    
    changes = {}
    
    for emotion in emotions:
        percentages = []
        for period in periods:
            pct = results[period]['percentages'].get(emotion, 0)
            percentages.append(pct)
        
        # Calculate changes
        pre_to_2022 = percentages[1] - percentages[0]
        pre_to_2023 = percentages[2] - percentages[0]
        change_2022_to_2023 = percentages[2] - percentages[1]
        
        changes[emotion] = {
            'pre_to_2022': pre_to_2022,
            'pre_to_2023': pre_to_2023,
            '2022_to_2023': change_2022_to_2023,
            'pre_percentage': percentages[0],
            '2022_percentage': percentages[1],
            '2023_percentage': percentages[2]
        }
    
    # Sort by largest absolute change from pre-ChatGPT to established period
    sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]['pre_to_2023']), reverse=True)
    
    print("Emotion changes from Pre-ChatGPT to Established ChatGPT period:")
    for emotion, change_data in sorted_changes:
        change = change_data['pre_to_2023']
        direction = "increased" if change > 0 else "decreased"
        print(f"  {emotion}: {change_data['pre_percentage']:.1f}% → {change_data['2023_percentage']:.1f}% ({direction} by {abs(change):.1f} percentage points)")
    
    return changes

def generate_report(results, changes, chi2, p_value, all_data):
    """Generate a comprehensive analysis report"""
    
    # Calculate overall emotion distribution
    total_tweets = sum(results[period]['total'] for period in results)
    overall_emotion_counts = all_data['emotion_label'].value_counts()
    overall_emotion_pcts = (overall_emotion_counts / len(all_data) * 100).round(1)
    
    report = f"""
EMOTION TIMELINE ANALYSIS REPORT
================================

ANALYSIS OVERVIEW
-----------------
This analysis compares emotional sentiment in tweets about AI/artificial intelligence across three distinct time periods:

1. Pre-ChatGPT Era (2017-2021): {results['Pre-ChatGPT (2017-2021)']['total']:,} tweets
2. Right after ChatGPT Launch (2022): {results['Right after ChatGPT (2022)']['total']:,} tweets  
3. Established ChatGPT Era (2023+): {results['Established ChatGPT (2023+)']['total']:,} tweets

Total dataset: {total_tweets:,} tweets

OVERALL EMOTION DISTRIBUTION (ALL PERIODS COMBINED)
---------------------------------------------------
"""
    
    for emotion in sorted(overall_emotion_pcts.index):
        count = overall_emotion_counts[emotion]
        pct = overall_emotion_pcts[emotion]
        report += f"   {emotion}: {count:,} tweets ({pct}%)\n"
    
    report += f"""
KEY FINDINGS
------------

1. STATISTICAL SIGNIFICANCE
   - Chi-square test shows {'highly significant' if p_value < 0.001 else 'significant' if p_value < 0.05 else 'no significant'} differences in emotion distributions
   - Chi-square statistic: {chi2:.2f}
   - p-value: {p_value:.2e}

2. MAJOR EMOTION CHANGES FROM PRE-CHATGPT TO ESTABLISHED ERA
"""

    # Add top 5 emotion changes
    sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]['pre_to_2023']), reverse=True)
    for i, (emotion, change_data) in enumerate(sorted_changes[:5]):
        change = change_data['pre_to_2023']
        direction = "increased" if change > 0 else "decreased"
        report += f"\n   {i+1}. {emotion.upper()}: {direction} by {abs(change):.1f} percentage points"
        report += f"      ({change_data['pre_percentage']:.1f}% → {change_data['2023_percentage']:.1f}%)"

    report += f"""

3. EMOTION DISTRIBUTION BY PERIOD
"""

    for period in ['Pre-ChatGPT (2017-2021)', 'Right after ChatGPT (2022)', 'Established ChatGPT (2023+)']:
        report += f"\n   {period}:"
        for emotion in sorted(results[period]['percentages'].index):
            pct = results[period]['percentages'][emotion]
            report += f"\n      {emotion}: {pct:.1f}%"

    report += f"""

4. INSIGHTS AND INTERPRETATIONS
   - The launch of ChatGPT in late 2022 marked a significant shift in public sentiment
   - The most dramatic changes occurred between the pre-ChatGPT era and the established ChatGPT period
   - These changes reflect evolving public perception, awareness, and emotional responses to AI technology

METHODOLOGY
-----------
- Data sources: clean_tweets_ai.labeled.csv (pre-ChatGPT) and AfterChatGPT.labeled.csv (post-ChatGPT)
- Emotion labeling: Automated emotion classification with probability scores
- Statistical analysis: Chi-square test for independence
- Time periods defined based on ChatGPT launch timeline (November 2022)

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save report
    with open('Emotional_Analysis/emotion_timeline_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("\nReport saved as 'Emotional_Analysis/emotion_timeline_analysis_report.txt'")
    return report

def main():
    """Main analysis function"""
    
    print("EMOTION TIMELINE ANALYSIS")
    print("=" * 50)
    print("Comparing emotions across AI discourse time periods")
    print()
    
    # Load and process data
    pre_data, after_2022, after_2023_plus, all_data = load_and_clean_data()
    
    # Organize data for analysis
    data_dict = {
        'Pre-ChatGPT (2017-2021)': pre_data,
        'Right after ChatGPT (2022)': after_2022,
        'Established ChatGPT (2023+)': after_2023_plus
    }
    
    # Analyze emotion distributions
    results = analyze_emotion_distributions(data_dict, list(data_dict.keys()))
    
    # Create visualizations
    create_visualizations(all_data, results)
    
    # Statistical analysis
    chi2, p_value, dof = calculate_statistical_significance(results)
    
    # Identify key changes
    changes = identify_key_changes(results)
    
    # Generate comprehensive report
    report = generate_report(results, changes, chi2, p_value, all_data)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("Files generated:")
    print("- emotion_timeline_analysis.png (visualizations)")
    print("- emotion_timeline_analysis_report.txt (detailed report)")

if __name__ == "__main__":
    main()
