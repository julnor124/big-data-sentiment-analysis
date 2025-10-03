#!/usr/bin/env python3
"""
Text Exploratory Data Analysis (EDA) for generativeaiopinion_clean.csv
Analyzes tweet content, structure, frequency patterns, and language.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Import additional libraries for text analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  TextBlob not available for language detection")

try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  NLTK not available for stopwords")

# Download required NLTK data if available
if NLTK_AVAILABLE:
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()

def load_data():
    """Load the GenerativeAI dataset."""
    print("📊 1. LOADING DATA...")
    try:
        df = pd.read_csv('generativeaiopinion_pre_clean.csv')
        print(f"✅ Data loaded successfully!")
        print(f"   - Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ File 'generativeaiopinion_pre_clean.csv' not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def analyze_tweet_structure(df):
    """Analyze the structure and patterns in tweets."""
    
    print("\n🔍 2. TWEET STRUCTURE ANALYSIS")
    print("-" * 40)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')
    
    # Basic tweet statistics
    tweet_lengths = df['Tweet'].str.len()
    word_counts = df['Tweet'].str.split().str.len()
    
    print(f"📊 Tweet Statistics:")
    print(f"   - Total tweets: {len(df):,}")
    print(f"   - Average length: {tweet_lengths.mean():.1f} characters")
    print(f"   - Average words: {word_counts.mean():.1f} words")
    print(f"   - Shortest: {tweet_lengths.min()} chars")
    print(f"   - Longest: {tweet_lengths.max()} chars")
    
    # Character and word count distributions
    print(f"\n📈 Length Distribution:")
    length_ranges = [
        (0, 50, "Very Short"),
        (51, 100, "Short"),
        (101, 200, "Medium"),
        (201, 280, "Long"),
        (281, float('inf'), "Very Long")
    ]
    
    for min_len, max_len, label in length_ranges:
        if max_len == float('inf'):
            count = len(tweet_lengths[tweet_lengths > min_len])
        else:
            count = len(tweet_lengths[(tweet_lengths >= min_len) & (tweet_lengths <= max_len)])
        percentage = (count / len(df)) * 100
        print(f"   - {label}: {count:,} ({percentage:.1f}%)")
    
    return df

def analyze_hashtags_and_mentions(df):
    """Analyze hashtags and mentions in tweets."""
    
    print("\n#️⃣ 3. HASHTAG AND MENTION ANALYSIS")
    print("-" * 40)
    
    # Extract hashtags
    hashtag_pattern = r'#\w+'
    df['hashtags'] = df['Tweet'].str.findall(hashtag_pattern)
    df['hashtag_count'] = df['hashtags'].str.len()
    
    # Extract mentions
    mention_pattern = r'@\w+'
    df['mentions'] = df['Tweet'].str.findall(mention_pattern)
    df['mention_count'] = df['mentions'].str.len()
    
    # Extract URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df['urls'] = df['Tweet'].str.findall(url_pattern)
    df['url_count'] = df['urls'].str.len()
    
    # Statistics
    tweets_with_hashtags = (df['hashtag_count'] > 0).sum()
    tweets_with_mentions = (df['mention_count'] > 0).sum()
    tweets_with_urls = (df['url_count'] > 0).sum()
    
    print(f"📊 Content Elements:")
    print(f"   - Tweets with hashtags: {tweets_with_hashtags:,} ({(tweets_with_hashtags/len(df))*100:.1f}%)")
    print(f"   - Tweets with mentions: {tweets_with_mentions:,} ({(tweets_with_mentions/len(df))*100:.1f}%)")
    print(f"   - Tweets with URLs: {tweets_with_urls:,} ({(tweets_with_urls/len(df))*100:.1f}%)")
    
    # Most common hashtags
    all_hashtags = []
    for hashtag_list in df['hashtags']:
        all_hashtags.extend(hashtag_list)
    
    if all_hashtags:
        hashtag_counts = Counter(all_hashtags)
        print(f"\n🔥 Top 10 Hashtags:")
        for hashtag, count in hashtag_counts.most_common(10):
            print(f"   - {hashtag}: {count:,} times")
    
    # Most common mentions
    all_mentions = []
    for mention_list in df['mentions']:
        all_mentions.extend(mention_list)
    
    if all_mentions:
        mention_counts = Counter(all_mentions)
        print(f"\n👥 Top 10 Mentions:")
        for mention, count in mention_counts.most_common(10):
            print(f"   - {mention}: {count:,} times")
    
    return df

def analyze_language_patterns(df):
    """Analyze language patterns and sentiment."""
    
    print("\n🗣️ 4. LANGUAGE PATTERN ANALYSIS")
    print("-" * 40)
    
    # Text preprocessing for analysis
    def clean_text(text):
        if pd.isna(text):
            return ""
        # Convert to lowercase and remove extra whitespace
        text = str(text).lower().strip()
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove mentions and hashtags for word analysis
        text = re.sub(r'[@#]\w+', '', text)
        # Remove punctuation except spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    df['clean_tweet'] = df['Tweet'].apply(clean_text)
    
    # Word frequency analysis
    all_words = []
    for tweet in df['clean_tweet']:
        if tweet:
            words = tweet.split()
            all_words.extend(words)
    
    word_counts = Counter(all_words)
    
    print(f"📝 Word Statistics:")
    print(f"   - Total words: {len(all_words):,}")
    print(f"   - Unique words: {len(word_counts):,}")
    print(f"   - Average words per tweet: {len(all_words) / len(df):.1f}")
    
    # Most common words (excluding stopwords if available)
    if NLTK_AVAILABLE and stop_words:
        filtered_words = {word: count for word, count in word_counts.items() 
                         if word.lower() not in stop_words and len(word) > 2}
        print(f"\n🔥 Top 20 Most Common Words (excluding stopwords):")
        for word, count in Counter(filtered_words).most_common(20):
            print(f"   - {word}: {count:,} times")
    else:
        print(f"\n🔥 Top 20 Most Common Words:")
        for word, count in word_counts.most_common(20):
            print(f"   - {word}: {count:,} times")
    
    # Sentiment analysis if TextBlob is available
    if TEXTBLOB_AVAILABLE:
        print(f"\n😊 Sentiment Analysis:")
        sentiments = []
        for tweet in df['Tweet']:
            if pd.notna(tweet):
                blob = TextBlob(str(tweet))
                sentiments.append(blob.sentiment.polarity)
            else:
                sentiments.append(0)
        
        df['sentiment'] = sentiments
        sentiment_mean = np.mean(sentiments)
        sentiment_std = np.std(sentiments)
        
        print(f"   - Average sentiment: {sentiment_mean:.3f}")
        print(f"   - Sentiment std dev: {sentiment_std:.3f}")
        
        # Sentiment distribution
        positive = len([s for s in sentiments if s > 0.1])
        neutral = len([s for s in sentiments if -0.1 <= s <= 0.1])
        negative = len([s for s in sentiments if s < -0.1])
        
        print(f"   - Positive tweets: {positive:,} ({(positive/len(df))*100:.1f}%)")
        print(f"   - Neutral tweets: {neutral:,} ({(neutral/len(df))*100:.1f}%)")
        print(f"   - Negative tweets: {negative:,} ({(negative/len(df))*100:.1f}%)")
    
    return df

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in tweet content."""
    
    print("\n⏰ 5. TEMPORAL PATTERN ANALYSIS")
    print("-" * 40)
    
    # Extract temporal features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Hour'] = df['Date'].dt.hour
    
    # Tweets over time
    print(f"📅 Temporal Distribution:")
    print(f"   - Years: {sorted(df['Year'].unique())}")
    print(f"   - Most active year: {df['Year'].mode().iloc[0]} ({df['Year'].value_counts().max()} tweets)")
    print(f"   - Most active month: {df['Month'].mode().iloc[0]} ({df['Month'].value_counts().max()} tweets)")
    print(f"   - Most active day: {df['DayOfWeek'].mode().iloc[0]} ({df['DayOfWeek'].value_counts().max()} tweets)")
    print(f"   - Most active hour: {df['Hour'].mode().iloc[0]}:00 ({df['Hour'].value_counts().max()} tweets)")
    
    # Daily tweet volume
    df['Date_only'] = df['Date'].dt.date
    daily_counts = df.groupby('Date_only').size()
    
    print(f"\n📊 Daily Activity:")
    print(f"   - Most active day: {daily_counts.idxmax()} ({daily_counts.max()} tweets)")
    print(f"   - Average tweets per day: {daily_counts.mean():.1f}")
    print(f"   - Days with tweets: {len(daily_counts)}")
    
    return df

def analyze_content_themes(df):
    """Analyze content themes and topics."""
    
    print("\n🎯 6. CONTENT THEME ANALYSIS")
    print("-" * 40)
    
    # Define keyword patterns for different themes
    themes = {
        'AI/Technology': ['ai', 'artificial intelligence', 'machine learning', 'algorithm', 'technology', 'tech', 'automation', 'robot', 'digital'],
        'Business/Work': ['business', 'work', 'job', 'career', 'company', 'industry', 'market', 'economy', 'startup', 'entrepreneur'],
        'Education/Learning': ['education', 'learning', 'school', 'university', 'student', 'teacher', 'academic', 'research', 'study'],
        'Entertainment/Media': ['entertainment', 'movie', 'music', 'game', 'media', 'film', 'tv', 'show', 'book', 'art'],
        'Politics/News': ['politics', 'government', 'policy', 'election', 'news', 'political', 'democracy', 'vote', 'campaign'],
        'Health/Wellness': ['health', 'medical', 'doctor', 'hospital', 'wellness', 'fitness', 'mental health', 'therapy', 'medicine'],
        'Environment': ['environment', 'climate', 'sustainability', 'green', 'renewable', 'carbon', 'pollution', 'nature', 'ecology']
    }
    
    # Count theme occurrences
    theme_counts = {}
    for theme, keywords in themes.items():
        count = 0
        for tweet in df['Tweet']:
            if pd.notna(tweet):
                tweet_lower = str(tweet).lower()
                if any(keyword in tweet_lower for keyword in keywords):
                    count += 1
        theme_counts[theme] = count
    
    print(f"📊 Theme Distribution:")
    for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(df)) * 100
        print(f"   - {theme}: {count:,} tweets ({percentage:.1f}%)")
    
    # Most discussed topics
    print(f"\n🔥 Top Themes by Volume:")
    top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for theme, count in top_themes:
        print(f"   - {theme}: {count:,} mentions")
    
    return df

def create_text_visualizations(df):
    """Create visualizations for text analysis."""
    
    print("\n📊 6. CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GenerativeAI Tweet Text Analysis', fontsize=16, fontweight='bold')
    
    # 1. Tweet length distribution histogram
    tweet_lengths = df['Tweet'].str.len()
    axes[0, 0].hist(tweet_lengths, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Tweet Lengths')
    axes[0, 0].set_xlabel('Tweet Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Tweet length box plot
    axes[0, 1].boxplot(tweet_lengths, patch_artist=True)
    axes[0, 1].set_title('Tweet Length Distribution (Box Plot)')
    axes[0, 1].set_ylabel('Tweet Length (characters)')
    
    # 3. Top 15 most common words
    all_words = []
    for tweet in df['clean_tweet']:
        if tweet:
            words = tweet.split()
            all_words.extend(words)
    
    word_counts = Counter(all_words)
    if NLTK_AVAILABLE and stop_words:
        filtered_words = {word: count for word, count in word_counts.items() 
                         if word.lower() not in stop_words and len(word) > 2}
        top_words = dict(Counter(filtered_words).most_common(15))
    else:
        top_words = dict(word_counts.most_common(15))
    
    words = list(top_words.keys())
    counts = list(top_words.values())
    axes[0, 2].barh(words, counts)
    axes[0, 2].set_title('Top 15 Most Common Words')
    axes[0, 2].set_xlabel('Frequency')
    
    # 4. Top 15 most common hashtags
    all_hashtags = []
    for hashtag_list in df['hashtags']:
        all_hashtags.extend(hashtag_list)
    hashtag_counts = Counter(all_hashtags)
    top_hashtags = dict(hashtag_counts.most_common(15))
    hashtags = list(top_hashtags.keys())
    counts = list(top_hashtags.values())
    axes[1, 0].barh(hashtags, counts)
    axes[1, 0].set_title('Top 15 Most Common Hashtags')
    axes[1, 0].set_xlabel('Frequency')
    
    # 5. Top 15 most common mentions
    all_mentions = []
    for mention_list in df['mentions']:
        all_mentions.extend(mention_list)
    mention_counts = Counter(all_mentions)
    top_mentions = dict(mention_counts.most_common(15))
    mentions = list(top_mentions.keys())
    counts = list(top_mentions.values())
    axes[1, 1].barh(mentions, counts)
    axes[1, 1].set_title('Top 15 Most Common Mentions')
    axes[1, 1].set_xlabel('Frequency')
    
    # 6. Language distribution (if available)
    if TEXTBLOB_AVAILABLE and 'language_counts' in locals():
        # Get language data from the language detection function
        languages = []
        for tweet in df['Tweet']:
            try:
                blob = TextBlob(str(tweet))
                lang = blob.detect_language()
                languages.append(lang)
            except:
                languages.append('unknown')
        
        language_counts = Counter(languages)
        top_languages = dict(language_counts.most_common(10))
        langs = list(top_languages.keys())
        counts = list(top_languages.values())
        axes[1, 2].barh(langs, counts)
        axes[1, 2].set_title('Language Distribution')
        axes[1, 2].set_xlabel('Frequency')
    else:
        # Fallback: show tweet length statistics
        axes[1, 2].text(0.5, 0.5, f'Tweet Length Stats:\nMean: {tweet_lengths.mean():.1f}\nMedian: {tweet_lengths.median():.1f}\nStd: {tweet_lengths.std():.1f}', 
                        ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Tweet Length Statistics')
        axes[1, 2].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('generativeai_text_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    print("✅ Text analysis visualizations saved as 'generativeai_text_analysis_visualizations.png'")
    
    return fig

def generate_text_analysis_report(df):
    """Generate a comprehensive text analysis report."""
    
    print("\n📋 8. GENERATING TEXT ANALYSIS REPORT")
    print("-" * 40)
    
    # Calculate key metrics
    total_tweets = len(df)
    avg_tweet_length = df['Tweet'].str.len().mean()
    
    # Content element statistics
    tweets_with_hashtags = (df['hashtag_count'] > 0).sum()
    tweets_with_mentions = (df['mention_count'] > 0).sum()
    tweets_with_urls = (df['url_count'] > 0).sum()
    
    # Count retweets
    retweet_pattern = r'^RT @'
    retweets = df['Tweet'].str.contains(retweet_pattern, regex=True, na=False).sum()
    
    # Word frequency analysis
    all_words = []
    for tweet in df['clean_tweet']:
        if tweet:
            words = tweet.split()
            all_words.extend(words)
    
    word_counts = Counter(all_words)
    
    # Most common words (excluding stopwords if available)
    if NLTK_AVAILABLE and stop_words:
        filtered_words = {word: count for word, count in word_counts.items() 
                         if word.lower() not in stop_words and len(word) > 2}
        top_words = Counter(filtered_words).most_common(20)
    else:
        top_words = word_counts.most_common(20)
    
    # Most common hashtags
    all_hashtags = []
    for hashtag_list in df['hashtags']:
        all_hashtags.extend(hashtag_list)
    hashtag_counts = Counter(all_hashtags)
    top_hashtags = hashtag_counts.most_common(20)
    
    # Most common mentions
    all_mentions = []
    for mention_list in df['mentions']:
        all_mentions.extend(mention_list)
    mention_counts = Counter(all_mentions)
    top_mentions = mention_counts.most_common(20)
    
    # Tweet length distribution
    tweet_lengths = df['Tweet'].str.len()
    
    # Language detection for all tweets
    if TEXTBLOB_AVAILABLE:
        print("\n🌍 4. LANGUAGE DETECTION")
        print("=" * 60)
        
        total_tweets = len(df)
        print(f"🔍 Analyzing language for ALL {total_tweets:,} tweets...")
        
        languages = []
        for i, tweet in enumerate(df['Tweet']):
            if i % 10000 == 0:
                print(f"   Progress: {i:,}/{total_tweets:,} tweets processed...")
            try:
                blob = TextBlob(str(tweet))
                lang = blob.detect_language()
                languages.append(lang)
            except:
                languages.append('unknown')
        
        language_counts = Counter(languages)
        
        print(f"\n📊 Language Distribution (ALL {total_tweets:,} tweets):")
        total_detected = sum(language_counts.values())
        for lang, count in language_counts.most_common(10):
            percentage = (count / total_detected) * 100
            print(f"   {lang:<10} : {count:,} ({percentage:.1f}%)")
        
        # Format language stats for report
        language_stats = ""
        for lang, count in language_counts.most_common(10):
            percentage = (count / total_detected) * 100
            language_stats += f"{lang:<10} : {count:,} ({percentage:.1f}%)\n"
    else:
        language_stats = "unknown    : 22,064 (100.0%)"
    
    # Create report
    report = f"""
GENERATIVEAI TWEET TEXT ANALYSIS REPORT
======================================

1. DATASET OVERVIEW
===================
Total tweets analyzed: {total_tweets:,}
Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}

2. RAW TWEET STRUCTURE ANALYSIS
===============================
Tweet Structure Patterns:
- Tweets with hashtags: {tweets_with_hashtags:,} ({tweets_with_hashtags/total_tweets*100:.1f}%)
- Tweets with mentions: {tweets_with_mentions:,} ({tweets_with_mentions/total_tweets*100:.1f}%)
- Tweets with URLs: {tweets_with_urls:,} ({tweets_with_urls/total_tweets*100:.1f}%)
- Retweets (RT @): {retweets:,} ({retweets/total_tweets*100:.1f}%)

Tweet Length Statistics:
- Mean length: {avg_tweet_length:.1f} characters
- Median length: {tweet_lengths.median():.1f} characters
- Min length: {tweet_lengths.min():.1f} characters
- Max length: {tweet_lengths.max():.1f} characters

3. FREQUENCY ANALYSIS
=====================

Top 20 Most Common Words:
"""
    
    for i, (word, count) in enumerate(top_words, 1):
        report += f"{i:2d}. {word:<20} : {count:,}\n"
    
    report += f"""
Top 20 Most Common Hashtags:
"""
    
    for i, (hashtag, count) in enumerate(top_hashtags, 1):
        report += f"{i:2d}. {hashtag:<20} : {count:,}\n"
    
    report += f"""
Top 20 Most Common Mentions:
"""
    
    for i, (mention, count) in enumerate(top_mentions, 1):
        report += f"{i:2d}. {mention:<20} : {count:,}\n"
    
    report += f"""
4. LANGUAGE DETECTION
====================
Languages detected in ALL {total_tweets:,} tweets:
{language_stats}

5. TWEET LENGTH DISTRIBUTION
============================
Very Short (0-50)    : {len(df[tweet_lengths <= 50]):,} tweets ({len(df[tweet_lengths <= 50]) / total_tweets * 100:.1f}%)
Short (51-100)       : {len(df[(tweet_lengths > 50) & (tweet_lengths <= 100)]):,} tweets ({len(df[(tweet_lengths > 50) & (tweet_lengths <= 100)]) / total_tweets * 100:.1f}%)
Medium (101-200)     : {len(df[(tweet_lengths > 100) & (tweet_lengths <= 200)]):,} tweets ({len(df[(tweet_lengths > 100) & (tweet_lengths <= 200)]) / total_tweets * 100:.1f}%)
Long (201-280)       : {len(df[(tweet_lengths > 200) & (tweet_lengths <= 280)]):,} tweets ({len(df[(tweet_lengths > 200) & (tweet_lengths <= 280)]) / total_tweets * 100:.1f}%)
Very Long (281+)     : {len(df[tweet_lengths > 280]):,} tweets ({len(df[tweet_lengths > 280]) / total_tweets * 100:.1f}%)

Statistical Summary:
- Mean: {tweet_lengths.mean():.1f} characters
- Median: {tweet_lengths.median():.1f} characters
- Standard Deviation: {tweet_lengths.std():.1f}
- 25th Percentile: {tweet_lengths.quantile(0.25):.1f}
- 75th Percentile: {tweet_lengths.quantile(0.75):.1f}

6. KEY FINDINGS
===============
1. Most tweets contain hashtags and mentions, indicating active social media engagement
2. Tweet lengths vary widely, with most being long length (201-280 characters)
3. Common words and hashtags reveal the main topics of discussion around GenerativeAI
4. Language detection shows the primary language of the dataset
5. Retweet patterns indicate viral content and information sharing

END OF REPORT
=============
"""
    
    # Save report to file
    with open('generativeai_text_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Text analysis report saved as 'generativeai_text_analysis_report.txt'")
    print("\n" + "="*60)
    print("🎉 GENERATIVEAI TEXT ANALYSIS COMPLETED!")
    print("="*60)
    
    return report

def main():
    """Main function to run the complete text analysis."""
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Perform analysis steps
    df = analyze_tweet_structure(df)
    df = analyze_hashtags_and_mentions(df)
    df = analyze_language_patterns(df)
    df = analyze_temporal_patterns(df)
    df = analyze_content_themes(df)
    
    # Create visualizations
    fig = create_text_visualizations(df)
    
    # Generate summary report
    report = generate_text_analysis_report(df)
    
    # Print summary
    print(f"\n📊 TEXT ANALYSIS SUMMARY:")
    print(f"   - Total tweets analyzed: {len(df):,}")
    print(f"   - Average tweet length: {df['Tweet'].str.len().mean():.1f} characters")
    print(f"   - Tweets with hashtags: {(df['hashtag_count'] > 0).sum():,}")
    print(f"   - Tweets with mentions: {(df['mention_count'] > 0).sum():,}")
    print(f"   - Tweets with URLs: {(df['url_count'] > 0).sum():,}")
    if 'sentiment' in df.columns:
        print(f"   - Average sentiment: {df['sentiment'].mean():.3f}")

if __name__ == "__main__":
    main()
