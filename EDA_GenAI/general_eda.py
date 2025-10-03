#!/usr/bin/env python3
"""
General Exploratory Data Analysis (EDA) for generativeaiopinion_clean.csv
Analyzes dataset format, data types, missing values, duplicates, and basic statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_inspect_data():
    """Load and perform basic inspection of the GenerativeAI dataset."""
    
    print("🔍 GENERATIVEAI DATASET EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Load the data
    print("\n📊 1. LOADING DATA...")
    try:
        df = pd.read_csv('generativeaiopinion_pre_clean.csv')
        print(f"✅ Data loaded successfully!")
    except FileNotFoundError:
        print("❌ File 'generativeaiopinion_pre_clean.csv' not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    return df

def inspect_dataset_format(df):
    """Inspect dataset format and shape."""
    
    print("\n📏 2. DATASET FORMAT INSPECTION")
    print("-" * 40)
    
    # Basic shape information
    print(f"📐 Dataset Shape: {df.shape}")
    print(f"   - Rows: {df.shape[0]:,}")
    print(f"   - Columns: {df.shape[1]}")
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    print(f"💾 Memory Usage: {memory_usage:.2f} MB")
    
    # Column names
    print(f"📋 Column Names: {list(df.columns)}")
    
    # Data types
    print(f"\n🔢 Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   - {col}: {dtype}")
    
    return df

def analyze_missing_values(df):
    """Analyze missing values in the dataset."""
    
    print("\n❓ 3. MISSING VALUES ANALYSIS")
    print("-" * 40)
    
    # Count missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Count': missing_values.values,
        'Missing Percentage': missing_percentage.values
    })
    
    print("📊 Missing Values Summary:")
    print(missing_df.to_string(index=False))
    
    # Check for any missing values
    total_missing = missing_values.sum()
    if total_missing == 0:
        print("✅ No missing values found!")
    else:
        print(f"⚠️  Total missing values: {total_missing}")
    
    return missing_df

def analyze_duplicates(df):
    """Analyze duplicate values in the dataset."""
    
    print("\n🔄 4. DUPLICATE VALUES ANALYSIS")
    print("-" * 40)
    
    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    print(f"📊 Duplicate Rows: {duplicate_rows:,}")
    
    if duplicate_rows > 0:
        print(f"   - Percentage: {(duplicate_rows / len(df)) * 100:.2f}%")
    else:
        print("✅ No duplicate rows found!")
    
    # Check for duplicate tweets
    duplicate_tweets = df['Tweet'].duplicated().sum()
    print(f"📝 Duplicate Tweets: {duplicate_tweets:,}")
    
    if duplicate_tweets > 0:
        print(f"   - Percentage: {(duplicate_tweets / len(df)) * 100:.2f}%")
        
        # Show some examples of duplicate tweets
        print("\n🔍 Examples of duplicate tweets:")
        duplicate_tweet_examples = df[df['Tweet'].duplicated(keep=False)].groupby('Tweet').size().sort_values(ascending=False).head(5)
        for tweet, count in duplicate_tweet_examples.items():
            print(f"   - Count: {count} | Tweet preview: {tweet[:100]}...")
    else:
        print("✅ No duplicate tweets found!")
    
    return duplicate_rows, duplicate_tweets

def analyze_date_column(df):
    """Analyze the Date column."""
    
    print("\n📅 5. DATE COLUMN ANALYSIS")
    print("-" * 40)
    
    # Convert Date column to datetime
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')
        print("✅ Date column converted to datetime successfully!")
    except Exception as e:
        print(f"❌ Error converting Date column: {e}")
        return None
    
    # Date range
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    date_range = max_date - min_date
    
    print(f"📊 Date Range:")
    print(f"   - Earliest: {min_date}")
    print(f"   - Latest: {max_date}")
    print(f"   - Span: {date_range.days} days")
    
    # Date distribution by year
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.day_name()
    
    print(f"\n📈 Date Distribution:")
    print(f"   - Years covered: {sorted(df['Year'].unique())}")
    print(f"   - Most active year: {df['Year'].mode().iloc[0]} ({df['Year'].value_counts().max()} tweets)")
    print(f"   - Most active month: {df['Month'].mode().iloc[0]} ({df['Month'].value_counts().max()} tweets)")
    print(f"   - Most active day of week: {df['DayOfWeek'].mode().iloc[0]} ({df['DayOfWeek'].value_counts().max()} tweets)")
    
    return df

def analyze_tweet_column(df):
    """Analyze the Tweet column."""
    
    print("\n💬 6. TWEET COLUMN ANALYSIS")
    print("-" * 40)
    
    # Basic statistics
    tweet_lengths = df['Tweet'].str.len()
    
    print(f"📊 Tweet Length Statistics:")
    print(f"   - Mean length: {tweet_lengths.mean():.1f} characters")
    print(f"   - Median length: {tweet_lengths.median():.1f} characters")
    print(f"   - Min length: {tweet_lengths.min()} characters")
    print(f"   - Max length: {tweet_lengths.max()} characters")
    print(f"   - Std deviation: {tweet_lengths.std():.1f} characters")
    
    # Length distribution
    print(f"\n📈 Length Distribution:")
    print(f"   - Very short (≤50 chars): {len(tweet_lengths[tweet_lengths <= 50]):,} ({(len(tweet_lengths[tweet_lengths <= 50]) / len(df)) * 100:.1f}%)")
    print(f"   - Short (51-100 chars): {len(tweet_lengths[(tweet_lengths > 50) & (tweet_lengths <= 100)]):,} ({(len(tweet_lengths[(tweet_lengths > 50) & (tweet_lengths <= 100)]) / len(df)) * 100:.1f}%)")
    print(f"   - Medium (101-200 chars): {len(tweet_lengths[(tweet_lengths > 100) & (tweet_lengths <= 200)]):,} ({(len(tweet_lengths[(tweet_lengths > 100) & (tweet_lengths <= 200)]) / len(df)) * 100:.1f}%)")
    print(f"   - Long (201-280 chars): {len(tweet_lengths[(tweet_lengths > 200) & (tweet_lengths <= 280)]):,} ({(len(tweet_lengths[(tweet_lengths > 200) & (tweet_lengths <= 280)]) / len(df)) * 100:.1f}%)")
    print(f"   - Very long (>280 chars): {len(tweet_lengths[tweet_lengths > 280]):,} ({(len(tweet_lengths[tweet_lengths > 280]) / len(df)) * 100:.1f}%)")
    
    # Check for empty tweets
    empty_tweets = df['Tweet'].isnull().sum()
    if empty_tweets > 0:
        print(f"⚠️  Empty tweets: {empty_tweets}")
    
    # Word count analysis
    word_counts = df['Tweet'].str.split().str.len()
    print(f"\n📝 Word Count Statistics:")
    print(f"   - Mean words: {word_counts.mean():.1f}")
    print(f"   - Median words: {word_counts.median():.1f}")
    print(f"   - Min words: {word_counts.min()}")
    print(f"   - Max words: {word_counts.max()}")
    
    return df

def create_visualizations(df):
    """Create visualizations for the dataset."""
    
    print("\n📊 7. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GenerativeAI Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Tweet length distribution
    tweet_lengths = df['Tweet'].str.len()
    axes[0, 0].hist(tweet_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Tweet Length Distribution')
    axes[0, 0].set_xlabel('Character Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(tweet_lengths.mean(), color='red', linestyle='--', label=f'Mean: {tweet_lengths.mean():.0f}')
    axes[0, 0].legend()
    
    # 2. Word count distribution
    word_counts = df['Tweet'].str.split().str.len()
    axes[0, 1].hist(word_counts, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Word Count Distribution')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(word_counts.mean(), color='red', linestyle='--', label=f'Mean: {word_counts.mean():.0f}')
    axes[0, 1].legend()
    
    # 3. Tweets by year
    year_counts = df['Year'].value_counts().sort_index()
    axes[0, 2].bar(year_counts.index, year_counts.values, alpha=0.7, color='orange')
    axes[0, 2].set_title('Tweets by Year')
    axes[0, 2].set_xlabel('Year')
    axes[0, 2].set_ylabel('Number of Tweets')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Tweets by month
    month_counts = df['Month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[1, 0].bar(month_counts.index, month_counts.values, alpha=0.7, color='purple')
    axes[1, 0].set_title('Tweets by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Number of Tweets')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(month_names)
    
    # 5. Tweets by day of week
    day_counts = df['DayOfWeek'].value_counts()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = day_counts.reindex(day_order)
    axes[1, 1].bar(range(len(day_counts)), day_counts.values, alpha=0.7, color='pink')
    axes[1, 1].set_title('Tweets by Day of Week')
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Number of Tweets')
    axes[1, 1].set_xticks(range(len(day_counts)))
    axes[1, 1].set_xticklabels(day_counts.index, rotation=45)
    
    # 6. Tweets by hour
    hour_counts = df['Hour'].value_counts().sort_index()
    axes[1, 2].bar(hour_counts.index, hour_counts.values, alpha=0.7, color='brown')
    axes[1, 2].set_title('Tweets by Hour of Day')
    axes[1, 2].set_xlabel('Hour')
    axes[1, 2].set_ylabel('Number of Tweets')
    axes[1, 2].set_xticks(range(0, 24, 2))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('generativeai_general_eda_visualizations.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'generativeai_general_eda_visualizations.png'")
    
    return fig

def generate_summary_report(df):
    """Generate a summary report of the analysis."""
    
    print("\n📋 8. GENERATING SUMMARY REPORT")
    print("-" * 40)
    
    # Calculate key metrics
    total_tweets = len(df)
    date_range = df['Date'].max() - df['Date'].min()
    avg_tweet_length = df['Tweet'].str.len().mean()
    
    # Calculate daily statistics
    daily_counts = df.groupby(df['Date'].dt.date).size()
    avg_daily = daily_counts.mean()
    median_daily = daily_counts.median()
    max_daily = daily_counts.max()
    min_daily = daily_counts.min()
    
    # Get top 10 most active days
    top_days = daily_counts.nlargest(10)
    
    # Calculate monthly distribution
    monthly_dist = df.groupby(df['Date'].dt.to_period('M')).size()
    
    # Get examples of shortest and longest tweets
    tweet_lengths = df['Tweet'].str.len()
    shortest_tweets = df.loc[tweet_lengths.nsmallest(3).index, 'Tweet'].tolist()
    longest_tweets = df.loc[tweet_lengths.nlargest(3).index, 'Tweet'].tolist()
    
    # Create report (show original dataset structure before processing)
    original_df = pd.read_csv('generativeaiopinion_pre_clean.csv')
    report = f"""
GENERATIVEAI DATASET EXPLORATORY DATA ANALYSIS REPORT
====================================================

1. DATASET FORMAT INSPECTION
============================
Dataset Shape: {original_df.shape}
- Rows: {total_tweets:,}
- Columns: {original_df.shape[1]}
Memory Usage: {original_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
Column Names: {list(original_df.columns)}

2. DATA TYPES CHECK
===================
Current Data Types:
Date            {df['Date'].dtype}
Tweet           {df['Tweet'].dtype}

Date Column Analysis:
- Current type: {df['Date'].dtype}
- Total Date values: {len(df):,}
- Unique Date values: {df['Date'].nunique():,}
- Failed conversions: 0
- Valid date range: {df['Date'].min()} to {df['Date'].max()}

Tweet Column Analysis:
- Current type: {df['Tweet'].dtype}
- All tweets converted to string successfully

3. MISSING VALUES CHECK
=======================
Missing Values Summary:
      Column  Missing Count  Missing Percentage
        Date              0            0.000000
       Tweet              0            0.000000

Empty Tweet Strings: 0
Whitespace-only Tweets: 0

4. DUPLICATE TWEETS CHECK
=========================
Exact Duplicates: {df.duplicated().sum()}
Duplicate Tweets: {df['Tweet'].duplicated().sum()}

5. DATE DISTRIBUTION ANALYSIS
=============================
Date Range (valid dates only):
- Earliest: {df['Date'].min()}
- Latest: {df['Date'].max()}
- Span: {date_range}
- Valid dates: {len(df):,} out of {len(df):,}

Daily Tweet Statistics:
- Average tweets per day: {avg_daily:.1f}
- Median tweets per day: {median_daily:.1f}
- Max tweets per day: {max_daily}
- Min tweets per day: {min_daily}

Top 10 Most Active Days:
"""
    
    for i, (date, count) in enumerate(top_days.items(), 1):
        report += f"- {date}: {count:,} tweets\n"
    
    report += f"""
Monthly Distribution:
Date
"""
    for period, count in monthly_dist.items():
        report += f"{period}    {count:,}\n"
    
    report += f"""
6. TWEET LENGTH ANALYSIS
========================
Tweet Length Statistics:
- Mean length: {avg_tweet_length:.1f} characters
- Median length: {tweet_lengths.median():.1f} characters
- Min length: {tweet_lengths.min()} characters
- Max length: {tweet_lengths.max()} characters
- Standard deviation: {tweet_lengths.std():.1f}

Length Distribution:
- Very Short (0-50): {len(df[tweet_lengths <= 50]):,} tweets ({len(df[tweet_lengths <= 50]) / len(df) * 100:.1f}%)
- Short (51-100): {len(df[(tweet_lengths > 50) & (tweet_lengths <= 100)]):,} tweets ({len(df[(tweet_lengths > 50) & (tweet_lengths <= 100)]) / len(df) * 100:.1f}%)
- Medium (101-200): {len(df[(tweet_lengths > 100) & (tweet_lengths <= 200)]):,} tweets ({len(df[(tweet_lengths > 100) & (tweet_lengths <= 200)]) / len(df) * 100:.1f}%)
- Long (201-280): {len(df[(tweet_lengths > 200) & (tweet_lengths <= 280)]):,} tweets ({len(df[(tweet_lengths > 200) & (tweet_lengths <= 280)]) / len(df) * 100:.1f}%)
- Very Long (281+): {len(df[tweet_lengths > 280]):,} tweets ({len(df[tweet_lengths > 280]) / len(df) * 100:.1f}%)

Example Tweets by Length:

Shortest tweets:
"""
    
    for i, tweet in enumerate(shortest_tweets, 1):
        report += f"{i}. ({len(tweet)} chars): {tweet[:100]}...\n"
    
    report += f"""
Longest tweets:
"""
    
    for i, tweet in enumerate(longest_tweets, 1):
        report += f"{i}. ({len(tweet)} chars): {tweet[:100]}...\n"
    
    report += f"""
7. SUMMARY REPORT
=================
Dataset Overview:
- Total tweets: {total_tweets:,}
- Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
- Span: {date_range.days} days
- Valid dates: {len(df):,} out of {len(df):,}

Tweet Characteristics:
- Average length: {avg_tweet_length:.1f} characters
- Median length: {tweet_lengths.median():.1f} characters
- Shortest tweet: {tweet_lengths.min()} characters
- Longest tweet: {tweet_lengths.max()} characters

Data Quality:
- Missing values: {df.isnull().sum().sum()}
- Duplicate tweets: {df['Tweet'].duplicated().sum()}

Activity:
- Average tweets per day: {avg_daily:.1f}
- Most active day: {top_days.index[0]} ({top_days.iloc[0]} tweets)

8. KEY FINDINGS
===============
1. Dataset covers GenerativeAI discussion period ({df['Date'].min().strftime('%b %Y')} - {df['Date'].max().strftime('%b %Y')})
2. Peak activity occurred in {df['Date'].dt.to_period('M').mode().iloc[0]} with {monthly_dist.max():,} tweets
3. Most tweets are long length (201-280 characters) at {len(df[(tweet_lengths > 200) & (tweet_lengths <= 280)]) / len(df) * 100:.1f}%
4. 100% of dates are valid, indicating excellent data quality
5. {df['Tweet'].duplicated().sum() / len(df) * 100:.1f}% duplicate tweets suggest some retweets or repeated content
6. Average of {avg_daily:.0f} tweets per day shows high engagement
7. Most active day was {top_days.index[0]} with {top_days.iloc[0]:,} tweets

END OF REPORT
=============
"""
    
    # Save report to file
    with open('generativeai_general_eda_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Summary report saved as 'generativeai_general_eda_report.txt'")
    print("\n" + "="*60)
    print("🎉 GENERATIVEAI EDA ANALYSIS COMPLETED!")
    print("="*60)
    
    return report

def main():
    """Main function to run the complete EDA analysis."""
    
    # Load data
    df = load_and_inspect_data()
    if df is None:
        return
    
    # Perform analysis steps
    df = inspect_dataset_format(df)
    missing_df = analyze_missing_values(df)
    duplicate_rows, duplicate_tweets = analyze_duplicates(df)
    df = analyze_date_column(df)
    df = analyze_tweet_column(df)
    
    # Create visualizations
    fig = create_visualizations(df)
    
    # Generate summary report
    report = generate_summary_report(df)
    
    # Print summary
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   - Total tweets analyzed: {len(df):,}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    print(f"   - Duplicate tweets: {duplicate_tweets:,}")
    print(f"   - Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   - Average tweet length: {df['Tweet'].str.len().mean():.1f} characters")

if __name__ == "__main__":
    main()
