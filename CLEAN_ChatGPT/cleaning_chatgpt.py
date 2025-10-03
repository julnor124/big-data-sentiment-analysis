#!/usr/bin/env python3
"""
ChatGPT Dataset Cleaning Script
Cleans the ChatGPT_pre_clean.csv dataset by removing duplicates, handling missing values,
and preprocessing text content.
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import additional libraries for text processing
try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  NLTK not available for stopwords")

try:
    import emoji
    EMOJI_AVAILABLE = True
    print("✅ Emoji library available for emoji removal")
except ImportError:
    EMOJI_AVAILABLE = False
    print("⚠️  Emoji library not available - using regex fallback")

# Download required NLTK data if available
if NLTK_AVAILABLE:
    try:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()

def load_data():
    """Load the ChatGPT dataset."""
    print("📊 1. LOADING DATA...")
    try:
        df = pd.read_csv('../EDA_ChatGPT/ChatGPT_pre_clean.csv')
        print(f"✅ Data loaded successfully!")
        print(f"   - Original shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ File 'ChatGPT_pre_clean.csv' not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def remove_missing_dates(df):
    """Remove rows with missing dates."""
    print("\n🗓️ 2. REMOVING MISSING DATES...")
    
    initial_count = len(df)
    df_clean = df.dropna(subset=['Date'])
    removed_count = initial_count - len(df_clean)
    removed_percentage = (removed_count / initial_count * 100) if initial_count > 0 else 0
    
    print(f"   - Rows before: {initial_count:,}")
    print(f"   - Rows after: {len(df_clean):,}")
    print(f"   - Removed: {removed_count:,} rows with missing dates ({removed_percentage:.2f}%)")
    
    return df_clean, removed_count, removed_percentage

def remove_duplicates(df):
    """Remove exact duplicates (same tweet and same date/time)."""
    print("\n🔄 3. REMOVING DUPLICATES...")
    
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=['Date', 'Tweet'])
    removed_count = initial_count - len(df_clean)
    removed_percentage = (removed_count / initial_count * 100) if initial_count > 0 else 0
    
    print(f"   - Rows before: {initial_count:,}")
    print(f"   - Rows after: {len(df_clean):,}")
    print(f"   - Removed: {removed_count:,} duplicate rows ({removed_percentage:.2f}%)")
    
    return df_clean, removed_count, removed_percentage

def handle_nan_values(df):
    """Remove or handle tweets that contain NaN values."""
    print("\n❓ 4. HANDLING NaN VALUES...")
    
    initial_count = len(df)
    
    # Check for NaN values in Tweet column
    nan_mask = df['Tweet'].isna() | (df['Tweet'] == 'nan') | (df['Tweet'].str.contains('nan', case=False, na=False))
    nan_count = nan_mask.sum()
    removed_percentage = (nan_count / initial_count * 100) if initial_count > 0 else 0
    
    if nan_count > 0:
        print(f"   - Found {nan_count:,} tweets with NaN values")
        df_clean = df[~nan_mask].copy()
        print(f"   - Removed {nan_count:,} tweets with NaN values ({removed_percentage:.2f}%)")
    else:
        print("   - No NaN values found in tweets")
        df_clean = df.copy()
    
    print(f"   - Rows before: {initial_count:,}")
    print(f"   - Rows after: {len(df_clean):,}")
    
    return df_clean, nan_count, removed_percentage

def remove_stopwords(text):
    """Remove stopwords from text."""
    if not NLTK_AVAILABLE or not stop_words:
        return text
    
    try:
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    except:
        return text

def remove_emojis_and_mentions(text):
    """Remove emojis, mentions, and hashtags from text."""
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (#hashtag) - more comprehensive pattern
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'#\w*', '', text)
    
    # Remove emojis using emoji library if available
    if EMOJI_AVAILABLE:
        # Use emoji library for comprehensive emoji removal
        text = emoji.replace_emoji(text, replace='')
    else:
        # Fallback: basic emoji removal pattern
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"  # dingbats
            u"\U000024C2-\U0001F251"  # enclosed characters
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)
    
    return text

def remove_links(text):
    """Remove links from text."""
    # Remove http/https URLs
    text = re.sub(r'http\S+', '', text)
    # Remove www URLs
    text = re.sub(r'www\S+', '', text)
    # Remove t.co links (Twitter short URLs)
    text = re.sub(r't\.co/\S+', '', text)
    return text

def remove_special_characters(text):
    """Remove special characters from text."""
    # Keep only alphanumeric characters, spaces, and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
    # Remove extra punctuation that might be left
    text = re.sub(r'[.,!?;:]{2,}', '', text)  # Remove repeated punctuation
    return text

def clean_whitespace(text):
    """Clean whitespace and lowercase text."""
    # Strip leading and trailing whitespace
    text = text.strip()
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    
    return text

def preprocess_text(df):
    """Apply all text preprocessing steps."""
    print("\n📝 5. TEXT PREPROCESSING...")
    
    initial_count = len(df)
    df_clean = df.copy()
    
    print("   - Removing stopwords...")
    df_clean['Tweet'] = df_clean['Tweet'].apply(remove_stopwords)
    
    print("   - Removing emojis, mentions, and hashtags...")
    df_clean['Tweet'] = df_clean['Tweet'].apply(remove_emojis_and_mentions)
    
    print("   - Removing links...")
    df_clean['Tweet'] = df_clean['Tweet'].apply(remove_links)
    
    print("   - Removing special characters...")
    df_clean['Tweet'] = df_clean['Tweet'].apply(remove_special_characters)
    
    print("   - Cleaning whitespace and lowercasing...")
    df_clean['Tweet'] = df_clean['Tweet'].apply(clean_whitespace)
    
    # Remove tweets that became empty after cleaning
    empty_mask = (df_clean['Tweet'] == '') | (df_clean['Tweet'].str.len() < 3)
    empty_count = empty_mask.sum()
    removed_percentage = (empty_count / initial_count * 100) if initial_count > 0 else 0
    
    if empty_count > 0:
        print(f"   - Removing {empty_count:,} tweets that became empty after cleaning ({removed_percentage:.2f}%)")
        df_clean = df_clean[~empty_mask].copy()
    
    print(f"   - Rows before: {initial_count:,}")
    print(f"   - Rows after: {len(df_clean):,}")
    print(f"   - Removed: {initial_count - len(df_clean):,} rows")
    
    return df_clean, empty_count, removed_percentage

def save_cleaned_data(df):
    """Save the cleaned dataset."""
    print("\n💾 6. SAVING CLEANED DATA...")
    
    output_file = 'ChatGPT_cleaned.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Cleaned dataset saved as '{output_file}'")
    print(f"   - Final shape: {df.shape}")
    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return output_file

def create_cleaning_visualizations(original_df, cleaned_df, cleaning_stats):
    """Create visualizations for the cleaning process."""
    print("\n📊 7. CREATING VISUALIZATIONS...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ChatGPT Dataset Cleaning Analysis', fontsize=16, fontweight='bold')
    
    # 1. Dataset size comparison
    sizes = [len(original_df), len(cleaned_df)]
    labels = ['Original', 'Cleaned']
    colors = ['#ff6b6b', '#4ecdc4']
    
    axes[0, 0].bar(labels, sizes, color=colors)
    axes[0, 0].set_title('Dataset Size Comparison')
    axes[0, 0].set_ylabel('Number of Rows')
    for i, v in enumerate(sizes):
        axes[0, 0].text(i, v + max(sizes)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # 2. Cleaning steps breakdown
    steps = ['Missing Dates', 'Duplicates', 'NaN Values', 'Empty Tweets']
    removed_counts = [
        cleaning_stats['missing_dates_removed'],
        cleaning_stats['duplicates_removed'], 
        cleaning_stats['nan_removed'],
        cleaning_stats['empty_removed']
    ]
    
    # Filter out zero values for better visualization
    non_zero_mask = [count > 0 for count in removed_counts]
    steps_filtered = [step for step, mask in zip(steps, non_zero_mask) if mask]
    counts_filtered = [count for count, mask in zip(removed_counts, non_zero_mask) if mask]
    
    if counts_filtered:
        bars = axes[0, 1].bar(range(len(steps_filtered)), counts_filtered, color='#ff9ff3')
        axes[0, 1].set_title('Rows Removed by Cleaning Step')
        axes[0, 1].set_ylabel('Number of Rows')
        axes[0, 1].set_xticks(range(len(steps_filtered)))
        axes[0, 1].set_xticklabels(steps_filtered, rotation=45, ha='right')
        
        # Add percentage labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts_filtered)):
            height = bar.get_height()
            percentage = (count / len(original_df)) * 100
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(counts_filtered)*0.01,
                           f'{percentage:.2f}%', ha='center', va='bottom')
    
    # 3. Tweet length distribution (original vs cleaned)
    original_lengths = original_df['Tweet'].str.len()
    cleaned_lengths = cleaned_df['Tweet'].str.len()
    
    axes[0, 2].hist(original_lengths, bins=50, alpha=0.7, label='Original', color='#ff6b6b', density=True)
    axes[0, 2].hist(cleaned_lengths, bins=50, alpha=0.7, label='Cleaned', color='#4ecdc4', density=True)
    axes[0, 2].set_title('Tweet Length Distribution')
    axes[0, 2].set_xlabel('Character Count')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].legend()
    
    # 4. Memory usage comparison
    original_memory = original_df.memory_usage(deep=True).sum() / 1024**2
    cleaned_memory = cleaned_df.memory_usage(deep=True).sum() / 1024**2
    memory_sizes = [original_memory, cleaned_memory]
    
    axes[1, 0].bar(labels, memory_sizes, color=colors)
    axes[1, 0].set_title('Memory Usage Comparison')
    axes[1, 0].set_ylabel('Memory (MB)')
    for i, v in enumerate(memory_sizes):
        axes[1, 0].text(i, v + max(memory_sizes)*0.01, f'{v:.1f} MB', ha='center', va='bottom')
    
    # 5. Length statistics comparison
    stats_data = {
        'Mean': [original_lengths.mean(), cleaned_lengths.mean()],
        'Median': [original_lengths.median(), cleaned_lengths.median()],
        'Max': [original_lengths.max(), cleaned_lengths.max()]
    }
    
    x = np.arange(len(stats_data))
    width = 0.35
    
    for i, (stat, values) in enumerate(stats_data.items()):
        axes[1, 1].bar(x[i] - width/2, values[0], width, label='Original' if i == 0 else '', color='#ff6b6b', alpha=0.7)
        axes[1, 1].bar(x[i] + width/2, values[1], width, label='Cleaned' if i == 0 else '', color='#4ecdc4', alpha=0.7)
    
    axes[1, 1].set_title('Tweet Length Statistics')
    axes[1, 1].set_ylabel('Character Count')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(stats_data.keys())
    axes[1, 1].legend()
    
    # 6. Cleaning summary pie chart
    total_removed = len(original_df) - len(cleaned_df)
    remaining = len(cleaned_df)
    
    if total_removed > 0:
        pie_data = [remaining, total_removed]
        pie_labels = ['Kept', 'Removed']
        pie_colors = ['#4ecdc4', '#ff6b6b']
        
        wedges, texts, autotexts = axes[1, 2].pie(pie_data, labels=pie_labels, colors=pie_colors, 
                                                 autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Data Retention Summary')
        
        # Add count labels
        for i, (wedge, count) in enumerate(zip(wedges, pie_data)):
            angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
            x = 0.7 * np.cos(np.radians(angle))
            y = 0.7 * np.sin(np.radians(angle))
            axes[1, 2].text(x, y, f'{count:,}', ha='center', va='center', fontweight='bold')
    else:
        axes[1, 2].text(0.5, 0.5, 'No data removed', ha='center', va='center', 
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Data Retention Summary')
    
    plt.tight_layout()
    plt.savefig('ChatGPT_cleaning_visualizations.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'ChatGPT_cleaning_visualizations.png'")
    plt.close()

def generate_cleaning_report(original_df, cleaned_df, cleaning_stats):
    """Generate a cleaning report with detailed percentages."""
    print("\n📋 8. GENERATING CLEANING REPORT...")
    
    original_count = len(original_df)
    cleaned_count = len(cleaned_df)
    removed_count = original_count - cleaned_count
    
    # Calculate statistics
    original_memory = original_df.memory_usage(deep=True).sum() / 1024**2
    cleaned_memory = cleaned_df.memory_usage(deep=True).sum() / 1024**2
    
    # Tweet length statistics
    original_lengths = original_df['Tweet'].str.len()
    cleaned_lengths = cleaned_df['Tweet'].str.len()
    
    report = f"""
CHATGPT DATASET CLEANING REPORT
===============================

CLEANING SUMMARY:
- Original dataset: {original_count:,} rows
- Cleaned dataset: {cleaned_count:,} rows
- Removed: {removed_count:,} rows ({(removed_count/original_count)*100:.2f}%)

DETAILED CLEANING BREAKDOWN:
- Missing dates: {cleaning_stats['missing_dates_removed']:,} rows ({cleaning_stats['missing_dates_percentage']:.2f}%)
- Duplicates: {cleaning_stats['duplicates_removed']:,} rows ({cleaning_stats['duplicates_percentage']:.2f}%)
- NaN values: {cleaning_stats['nan_removed']:,} rows ({cleaning_stats['nan_percentage']:.2f}%)
- Empty tweets: {cleaning_stats['empty_removed']:,} rows ({cleaning_stats['empty_percentage']:.2f}%)

MEMORY USAGE:
- Original: {original_memory:.2f} MB
- Cleaned: {cleaned_memory:.2f} MB
- Reduction: {original_memory - cleaned_memory:.2f} MB ({(original_memory - cleaned_memory)/original_memory*100:.2f}%)

TWEET LENGTH STATISTICS:
Original Dataset:
- Mean length: {original_lengths.mean():.1f} characters
- Median length: {original_lengths.median():.1f} characters
- Min length: {original_lengths.min()} characters
- Max length: {original_lengths.max()} characters

Cleaned Dataset:
- Mean length: {cleaned_lengths.mean():.1f} characters
- Median length: {cleaned_lengths.median():.1f} characters
- Min length: {cleaned_lengths.min()} characters
- Max length: {cleaned_lengths.max()} characters

CLEANING STEPS APPLIED:
1. ✅ Removed rows with missing dates
2. ✅ Removed exact duplicates (same tweet and date/time)
3. ✅ Handled tweets containing NaN values
4. ✅ Removed stopwords (if NLTK available)
5. ✅ Removed emojis, mentions, and hashtags
6. ✅ Removed links and URLs
7. ✅ Removed special characters
8. ✅ Cleaned whitespace and lowercased

DATASET QUALITY:
- No missing dates: ✅
- No duplicates: ✅
- No NaN values: ✅
- Text preprocessed: ✅

"""
    
    # Save report to file
    with open('ChatGPT_cleaning_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Cleaning report saved as 'ChatGPT_cleaning_report.txt'")
    return report

def main():
    """Main function to run the complete cleaning process."""
    
    print("🧹 CHATGPT DATASET CLEANING")
    print("=" * 50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Store original for comparison
    original_df = df.copy()
    
    # Apply cleaning steps and collect statistics
    df, missing_dates_removed, missing_dates_percentage = remove_missing_dates(df)
    df, duplicates_removed, duplicates_percentage = remove_duplicates(df)
    df, nan_removed, nan_percentage = handle_nan_values(df)
    df, empty_removed, empty_percentage = preprocess_text(df)
    
    # Collect cleaning statistics
    cleaning_stats = {
        'missing_dates_removed': missing_dates_removed,
        'missing_dates_percentage': missing_dates_percentage,
        'duplicates_removed': duplicates_removed,
        'duplicates_percentage': duplicates_percentage,
        'nan_removed': nan_removed,
        'nan_percentage': nan_percentage,
        'empty_removed': empty_removed,
        'empty_percentage': empty_percentage
    }
    
    # Save cleaned data
    output_file = save_cleaned_data(df)
    
    # Create visualizations
    create_cleaning_visualizations(original_df, df, cleaning_stats)
    
    # Generate report
    report = generate_cleaning_report(original_df, df, cleaning_stats)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎉 CLEANING COMPLETED!")
    print("=" * 50)
    print(f"📊 Final Results:")
    print(f"   - Original: {len(original_df):,} rows")
    print(f"   - Cleaned: {len(df):,} rows")
    print(f"   - Removed: {len(original_df) - len(df):,} rows")
    print(f"   - Output file: {output_file}")
    print(f"   - Report: ChatGPT_cleaning_report.txt")
    print(f"   - Visualizations: ChatGPT_cleaning_visualizations.png")

if __name__ == "__main__":
    main()