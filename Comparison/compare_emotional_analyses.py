#!/usr/bin/env python3
"""
Compare Emotional Analysis Results
===================================
Compares the results from the first and second iterations of emotional timeline analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_analysis_data():
    """Load emotion data from both iterations"""
    
    print("Loading data from both iterations...")
    
    # Iteration 1 data
    print("\n1. Loading Iteration 1 data...")
    iter1_pre = pd.read_csv('Labeling/clean_tweets_ai.labeled.csv')
    iter1_after = pd.read_csv('Labeling/AfterChatGPT.labeled.csv')
    
    # Clean iteration 1 dates
    iter1_pre['Date'] = pd.to_datetime(iter1_pre['Date'])
    iter1_after['Date'] = pd.to_datetime(iter1_after['Date'], format='mixed', errors='coerce')
    iter1_after = iter1_after.dropna(subset=['Date'])
    
    # Filter iteration 1 by periods
    iter1_pre_data = iter1_pre.copy()
    iter1_2022 = iter1_after[iter1_after['Date'].dt.year == 2022].copy()
    iter1_2023 = iter1_after[iter1_after['Date'].dt.year >= 2023].copy()
    
    print(f"   Iter1 Pre-ChatGPT: {len(iter1_pre_data):,} tweets")
    print(f"   Iter1 2022: {len(iter1_2022):,} tweets")
    print(f"   Iter1 2023+: {len(iter1_2023):,} tweets")
    
    # Iteration 2 data
    print("\n2. Loading Iteration 2 data...")
    iter2_pre = pd.read_csv('Sampling_iter2/tweets_ai_downsampled.labeled.csv')
    iter2_after = pd.read_csv('Sampling_iter2/postlaunch.labeled.csv')
    
    # Standardize column names for iter2
    if 'date' in iter2_pre.columns:
        iter2_pre.rename(columns={'date': 'Date', 'tweet': 'Tweet'}, inplace=True)
    
    # Clean iteration 2 dates
    iter2_pre['Date'] = pd.to_datetime(iter2_pre['Date'])
    iter2_after['Date'] = pd.to_datetime(iter2_after['Date'], format='mixed', errors='coerce')
    iter2_after = iter2_after.dropna(subset=['Date'])
    
    # Filter iteration 2 by periods
    iter2_pre_data = iter2_pre.copy()
    iter2_2022 = iter2_after[iter2_after['Date'].dt.year == 2022].copy()
    iter2_2023 = iter2_after[iter2_after['Date'].dt.year >= 2023].copy()
    
    print(f"   Iter2 Pre-ChatGPT: {len(iter2_pre_data):,} tweets")
    print(f"   Iter2 2022: {len(iter2_2022):,} tweets")
    print(f"   Iter2 2023+: {len(iter2_2023):,} tweets")
    
    return {
        'iter1': {
            'pre': iter1_pre_data,
            '2022': iter1_2022,
            '2023+': iter1_2023
        },
        'iter2': {
            'pre': iter2_pre_data,
            '2022': iter2_2022,
            '2023+': iter2_2023
        }
    }

def calculate_emotion_percentages(data_dict):
    """Calculate emotion percentages for each iteration and period"""
    
    results = {}
    
    for iteration in ['iter1', 'iter2']:
        results[iteration] = {}
        for period in ['pre', '2022', '2023+']:
            data = data_dict[iteration][period]
            emotion_counts = data['emotion_label'].value_counts()
            emotion_pcts = (emotion_counts / len(data) * 100).round(2)
            
            results[iteration][period] = {
                'counts': emotion_counts,
                'percentages': emotion_pcts,
                'total': len(data)
            }
    
    return results

def compare_iterations(results):
    """Compare emotion distributions between iterations"""
    
    print("\n" + "="*80)
    print("COMPARISON OF EMOTIONAL ANALYSIS ITERATIONS")
    print("="*80)
    
    periods = ['pre', '2022', '2023+']
    period_names = {
        'pre': 'Pre-ChatGPT (2017-2021)',
        '2022': 'Right after ChatGPT (2022)',
        '2023+': 'Established ChatGPT (2023+)'
    }
    
    comparisons = []
    
    for period in periods:
        print(f"\n{'='*80}")
        print(f"{period_names[period]}")
        print(f"{'='*80}")
        
        iter1_pcts = results['iter1'][period]['percentages']
        iter2_pcts = results['iter2'][period]['percentages']
        
        iter1_total = results['iter1'][period]['total']
        iter2_total = results['iter2'][period]['total']
        
        print(f"\nDataset Sizes:")
        print(f"  Iteration 1: {iter1_total:,} tweets")
        print(f"  Iteration 2: {iter2_total:,} tweets")
        print(f"  Difference: {abs(iter1_total - iter2_total):,} tweets ({((iter2_total/iter1_total - 1) * 100):.1f}%)")
        
        print(f"\nEmotion Distribution Comparison:")
        print(f"{'Emotion':<12} {'Iter1 %':>10} {'Iter2 %':>10} {'Difference':>12} {'Agreement':>12}")
        print("-" * 80)
        
        # Get all emotions from both iterations
        all_emotions = sorted(set(iter1_pcts.index) | set(iter2_pcts.index))
        
        for emotion in all_emotions:
            iter1_val = iter1_pcts.get(emotion, 0)
            iter2_val = iter2_pcts.get(emotion, 0)
            diff = iter2_val - iter1_val
            
            # Calculate agreement (lower difference = higher agreement)
            agreement = 100 - abs(diff)
            
            comparisons.append({
                'period': period_names[period],
                'emotion': emotion,
                'iter1_pct': iter1_val,
                'iter2_pct': iter2_val,
                'difference': diff,
                'agreement': agreement
            })
            
            print(f"{emotion:<12} {iter1_val:>9.1f}% {iter2_val:>9.1f}% {diff:>+11.1f}% {agreement:>11.1f}%")
    
    return pd.DataFrame(comparisons)

def calculate_overall_agreement(comparison_df):
    """Calculate overall agreement metrics"""
    
    print("\n" + "="*80)
    print("OVERALL AGREEMENT METRICS")
    print("="*80)
    
    # Calculate average absolute difference
    avg_abs_diff = comparison_df['difference'].abs().mean()
    
    # Calculate correlation
    correlation = comparison_df['iter1_pct'].corr(comparison_df['iter2_pct'])
    
    # Calculate RMSE (Root Mean Square Error)
    rmse = np.sqrt(((comparison_df['iter1_pct'] - comparison_df['iter2_pct']) ** 2).mean())
    
    # Calculate mean absolute error
    mae = comparison_df['difference'].abs().mean()
    
    print(f"\nAgreement Statistics Across All Periods and Emotions:")
    print(f"  Correlation coefficient: {correlation:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} percentage points")
    print(f"  Root Mean Square Error (RMSE): {rmse:.2f} percentage points")
    print(f"  Average Agreement: {(100 - mae):.1f}%")
    
    # Agreement by period
    print(f"\nAgreement by Period:")
    for period in comparison_df['period'].unique():
        period_data = comparison_df[comparison_df['period'] == period]
        period_mae = period_data['difference'].abs().mean()
        period_agreement = 100 - period_mae
        print(f"  {period}: {period_agreement:.1f}% (MAE: {period_mae:.2f}pp)")
    
    # Agreement by emotion
    print(f"\nAgreement by Emotion (across all periods):")
    for emotion in sorted(comparison_df['emotion'].unique()):
        emotion_data = comparison_df[comparison_df['emotion'] == emotion]
        emotion_mae = emotion_data['difference'].abs().mean()
        emotion_agreement = 100 - emotion_mae
        print(f"  {emotion}: {emotion_agreement:.1f}% (MAE: {emotion_mae:.2f}pp)")
    
    return {
        'correlation': correlation,
        'mae': mae,
        'rmse': rmse,
        'avg_agreement': 100 - mae
    }

def create_comparison_visualizations(comparison_df, results):
    """Create comprehensive comparison visualizations"""
    
    print("\nCreating comparison visualizations...")
    
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Side-by-side emotion distributions for each period
    periods = ['pre', '2022', '2023+']
    period_names = {
        'pre': 'Pre-ChatGPT (2017-2021)',
        '2022': 'Right after ChatGPT (2022)',
        '2023+': 'Established ChatGPT (2023+)'
    }
    
    for idx, period in enumerate(periods):
        plt.subplot(4, 3, idx + 1)
        
        iter1_pcts = results['iter1'][period]['percentages']
        iter2_pcts = results['iter2'][period]['percentages']
        
        emotions = sorted(set(iter1_pcts.index) | set(iter2_pcts.index))
        x = np.arange(len(emotions))
        width = 0.35
        
        iter1_vals = [iter1_pcts.get(e, 0) for e in emotions]
        iter2_vals = [iter2_pcts.get(e, 0) for e in emotions]
        
        plt.bar(x - width/2, iter1_vals, width, label='Iteration 1', alpha=0.8)
        plt.bar(x + width/2, iter2_vals, width, label='Iteration 2', alpha=0.8)
        
        plt.xlabel('Emotions')
        plt.ylabel('Percentage (%)')
        plt.title(f'{period_names[period]}')
        plt.xticks(x, emotions, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. Difference heatmap
    plt.subplot(4, 3, 4)
    
    emotions = sorted(comparison_df['emotion'].unique())
    periods_list = [period_names[p] for p in periods]
    
    diff_matrix = []
    for period in periods:
        period_name = period_names[period]
        row = []
        for emotion in emotions:
            diff = comparison_df[(comparison_df['period'] == period_name) & 
                                (comparison_df['emotion'] == emotion)]['difference'].values
            row.append(diff[0] if len(diff) > 0 else 0)
        diff_matrix.append(row)
    
    sns.heatmap(np.array(diff_matrix).T, 
                xticklabels=['Pre-ChatGPT', '2022', '2023+'],
                yticklabels=emotions,
                annot=True,
                fmt='.1f',
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Difference (Iter2 - Iter1, %)'})
    plt.title('Emotion Distribution Differences\n(Iteration 2 - Iteration 1)')
    
    # 3. Correlation scatter plot
    plt.subplot(4, 3, 5)
    plt.scatter(comparison_df['iter1_pct'], comparison_df['iter2_pct'], alpha=0.6)
    
    # Add perfect agreement line
    max_val = max(comparison_df['iter1_pct'].max(), comparison_df['iter2_pct'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Agreement')
    
    # Add regression line
    z = np.polyfit(comparison_df['iter1_pct'], comparison_df['iter2_pct'], 1)
    p = np.poly1d(z)
    plt.plot(comparison_df['iter1_pct'], p(comparison_df['iter1_pct']), 
             'g-', alpha=0.8, label=f'Regression (R={comparison_df["iter1_pct"].corr(comparison_df["iter2_pct"]):.3f})')
    
    plt.xlabel('Iteration 1 Emotion %')
    plt.ylabel('Iteration 2 Emotion %')
    plt.title('Emotion Distribution Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Agreement by emotion (bar chart)
    plt.subplot(4, 3, 6)
    emotion_agreement = comparison_df.groupby('emotion')['difference'].apply(lambda x: 100 - abs(x).mean())
    emotion_agreement = emotion_agreement.sort_values(ascending=False)
    
    colors = ['green' if x >= 90 else 'orange' if x >= 80 else 'red' for x in emotion_agreement.values]
    plt.barh(emotion_agreement.index, emotion_agreement.values, color=colors, alpha=0.7)
    plt.xlabel('Agreement (%)')
    plt.title('Agreement by Emotion (Across All Periods)')
    plt.axvline(x=90, color='g', linestyle='--', alpha=0.5, label='90% threshold')
    plt.axvline(x=80, color='orange', linestyle='--', alpha=0.5, label='80% threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Dataset size comparison
    plt.subplot(4, 3, 7)
    size_data = []
    for period in periods:
        size_data.append({
            'Period': period_names[period],
            'Iteration 1': results['iter1'][period]['total'],
            'Iteration 2': results['iter2'][period]['total']
        })
    
    size_df = pd.DataFrame(size_data)
    x = np.arange(len(periods))
    width = 0.35
    
    plt.bar(x - width/2, size_df['Iteration 1'], width, label='Iteration 1', alpha=0.8)
    plt.bar(x + width/2, size_df['Iteration 2'], width, label='Iteration 2', alpha=0.8)
    
    plt.xlabel('Time Period')
    plt.ylabel('Number of Tweets')
    plt.title('Dataset Size Comparison')
    plt.xticks(x, ['Pre-ChatGPT', '2022', '2023+'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6-8. Emotion change comparison for each period
    for idx, period in enumerate(periods):
        plt.subplot(4, 3, 10 + idx)
        
        period_name = period_names[period]
        period_data = comparison_df[comparison_df['period'] == period_name]
        
        emotions_sorted = period_data.sort_values('difference')['emotion'].values
        diffs = period_data.sort_values('difference')['difference'].values
        
        colors = ['red' if d < 0 else 'green' for d in diffs]
        plt.barh(emotions_sorted, diffs, color=colors, alpha=0.7)
        plt.xlabel('Difference (Iter2 - Iter1, %)')
        plt.title(f'Emotion Differences: {period_names[period][:15]}...')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Comparison/emotional_analysis_comparison.png', dpi=300, bbox_inches='tight')
    
    print("Visualizations saved as 'Comparison/emotional_analysis_comparison.png'")

def generate_comparison_report(comparison_df, metrics, results):
    """Generate comprehensive comparison report"""
    
    report = f"""
EMOTIONAL ANALYSIS COMPARISON REPORT
====================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
--------
This report compares the emotional timeline analysis results from two iterations:

Iteration 1:
- Pre-ChatGPT: clean_tweets_ai.labeled.csv ({results['iter1']['pre']['total']:,} tweets)
- Post-ChatGPT: AfterChatGPT.labeled.csv 
  * 2022: {results['iter1']['2022']['total']:,} tweets
  * 2023+: {results['iter1']['2023+']['total']:,} tweets
- Total: {sum(results['iter1'][p]['total'] for p in ['pre', '2022', '2023+']):,} tweets

Iteration 2:
- Pre-ChatGPT: tweets_ai_downsampled.labeled.csv ({results['iter2']['pre']['total']:,} tweets)
- Post-ChatGPT: postlaunch.labeled.csv
  * 2022: {results['iter2']['2022']['total']:,} tweets
  * 2023+: {results['iter2']['2023+']['total']:,} tweets
- Total: {sum(results['iter2'][p]['total'] for p in ['pre', '2022', '2023+']):,} tweets

OVERALL AGREEMENT METRICS
--------------------------
Correlation Coefficient: {metrics['correlation']:.4f}
Mean Absolute Error: {metrics['mae']:.2f} percentage points
RMSE: {metrics['rmse']:.2f} percentage points
Average Agreement: {metrics['avg_agreement']:.1f}%

Interpretation:
- Correlation of {metrics['correlation']:.3f} indicates {'strong' if metrics['correlation'] > 0.8 else 'moderate' if metrics['correlation'] > 0.6 else 'weak'} positive relationship
- Average agreement of {metrics['avg_agreement']:.1f}% shows {'high' if metrics['avg_agreement'] > 90 else 'moderate' if metrics['avg_agreement'] > 80 else 'low'} consistency
- MAE of {metrics['mae']:.2f}pp means predictions differ by {metrics['mae']:.2f} percentage points on average

AGREEMENT BY TIME PERIOD
-------------------------
"""
    
    periods = ['pre', '2022', '2023+']
    period_names = {
        'pre': 'Pre-ChatGPT (2017-2021)',
        '2022': 'Right after ChatGPT (2022)',
        '2023+': 'Established ChatGPT (2023+)'
    }
    
    for period in periods:
        period_name = period_names[period]
        period_data = comparison_df[comparison_df['period'] == period_name]
        period_mae = period_data['difference'].abs().mean()
        period_agreement = 100 - period_mae
        
        report += f"\n{period_name}:\n"
        report += f"  Agreement: {period_agreement:.1f}%\n"
        report += f"  MAE: {period_mae:.2f} percentage points\n"
        report += f"  Dataset sizes: Iter1={results['iter1'][period]['total']:,}, Iter2={results['iter2'][period]['total']:,}\n"
        
        report += f"\n  Emotion Distribution Comparison:\n"
        for _, row in period_data.iterrows():
            report += f"    {row['emotion']}: {row['iter1_pct']:.1f}% → {row['iter2_pct']:.1f}% (diff: {row['difference']:+.1f}pp)\n"

    report += f"""

AGREEMENT BY EMOTION
--------------------
"""
    
    for emotion in sorted(comparison_df['emotion'].unique()):
        emotion_data = comparison_df[comparison_df['emotion'] == emotion]
        emotion_mae = emotion_data['difference'].abs().mean()
        emotion_agreement = 100 - emotion_mae
        
        report += f"\n{emotion.upper()}:\n"
        report += f"  Agreement: {emotion_agreement:.1f}%\n"
        report += f"  MAE: {emotion_mae:.2f} percentage points\n"
        
        for _, row in emotion_data.iterrows():
            report += f"    {row['period'][:20]:20} {row['iter1_pct']:5.1f}% → {row['iter2_pct']:5.1f}% (diff: {row['difference']:+.1f}pp)\n"

    report += f"""

KEY FINDINGS
------------

1. STRONGEST AGREEMENTS (emotions with >95% agreement):
"""
    
    high_agreement = comparison_df.groupby('emotion')['difference'].apply(lambda x: 100 - abs(x).mean())
    high_agreement = high_agreement[high_agreement >= 95].sort_values(ascending=False)
    
    if len(high_agreement) > 0:
        for emotion, agreement in high_agreement.items():
            report += f"   - {emotion}: {agreement:.1f}% agreement\n"
    else:
        report += "   - None found (threshold: 95%)\n"

    report += f"""

2. LARGEST DISCREPANCIES (emotions with highest differences):
"""
    
    max_diffs = comparison_df.groupby('emotion')['difference'].apply(lambda x: abs(x).mean())
    max_diffs = max_diffs.sort_values(ascending=False).head(5)
    
    for emotion, diff in max_diffs.items():
        report += f"   - {emotion}: {diff:.2f} percentage points average difference\n"

    report += f"""

3. PERIOD-SPECIFIC INSIGHTS:
"""
    
    for period in periods:
        period_name = period_names[period]
        period_data = comparison_df[comparison_df['period'] == period_name]
        max_diff_row = period_data.loc[period_data['difference'].abs().idxmax()]
        
        report += f"\n   {period_name}:\n"
        report += f"   - Largest difference: {max_diff_row['emotion']} ({max_diff_row['difference']:+.1f}pp)\n"
        report += f"   - Iter1: {max_diff_row['iter1_pct']:.1f}%, Iter2: {max_diff_row['iter2_pct']:.1f}%\n"

    report += f"""

CONCLUSIONS
-----------

Dataset Comparison:
- Iteration 1 used the full datasets
- Iteration 2 used downsampled pre-ChatGPT data and combined post-launch data
- Size difference: Iter1 has {sum(results['iter1'][p]['total'] for p in periods):,} tweets vs Iter2 has {sum(results['iter2'][p]['total'] for p in periods):,} tweets

Validation:
- Overall correlation of {metrics['correlation']:.3f} suggests the two analyses are {'highly consistent' if metrics['correlation'] > 0.9 else 'moderately consistent' if metrics['correlation'] > 0.7 else 'somewhat consistent'}
- Key emotion trends (e.g., neutral decrease, joy/surprise increase) are {'confirmed' if metrics['correlation'] > 0.8 else 'partially confirmed'} across both iterations
- The main findings about emotional shifts after ChatGPT's launch are {'robust' if metrics['avg_agreement'] > 85 else 'moderately robust' if metrics['avg_agreement'] > 75 else 'variable'} to dataset choice

Recommendations:
- {'Use either dataset with confidence' if metrics['avg_agreement'] > 90 else 'Consider dataset differences when interpreting results' if metrics['avg_agreement'] > 75 else 'Be cautious about generalizing findings'}
- Focus on emotions with high agreement (>90%) for strongest conclusions
- Investigate large discrepancies for potential insights about sampling effects
"""
    
    # Save report
    with open('Comparison/emotional_analysis_comparison_report.txt', 'w') as f:
        f.write(report)
    
    print("\nReport saved as 'Comparison/emotional_analysis_comparison_report.txt'")
    
    return report

def main():
    """Main comparison function"""
    
    print("="*80)
    print("EMOTIONAL ANALYSIS COMPARISON")
    print("="*80)
    print("Comparing Iteration 1 vs Iteration 2 results")
    print()
    
    # Load data
    data_dict = load_analysis_data()
    
    # Calculate emotion percentages
    results = calculate_emotion_percentages(data_dict)
    
    # Compare iterations
    comparison_df = compare_iterations(results)
    
    # Calculate overall agreement
    metrics = calculate_overall_agreement(comparison_df)
    
    # Create visualizations
    create_comparison_visualizations(comparison_df, results)
    
    # Generate report
    generate_comparison_report(comparison_df, metrics, results)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("Files generated:")
    print("- Comparison/emotional_analysis_comparison.png")
    print("- Comparison/emotional_analysis_comparison_report.txt")
    print()

if __name__ == "__main__":
    main()

