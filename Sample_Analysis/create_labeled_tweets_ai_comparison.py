#!/usr/bin/env python3
"""
Create Labeled Tweets AI Comparison
===================================

This script creates a sample of 400 random tweets with different emotions
from clean_tweets_ai.labeled.csv, finds their original tweets from tweets_ai.csv,
and creates a side-by-side comparison CSV.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_labeled_tweets_ai_comparison():
    """
    Create a sample of 400 tweets with different emotions and find their originals.
    """
    
    print("🎭 CREATING LABELED TWEETS AI COMPARISON")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # File paths
    labeled_file = 'Labeling/clean_tweets_ai.labeled.csv'
    original_file = 'tweets_ai.csv'
    output_file = 'labeled_tweets_ai_comparison.csv'
    
    print("📂 Loading datasets...")
    
    # Check if files exist
    files_to_check = [labeled_file, original_file]
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found.")
            return None
    
    # Load labeled dataset
    print("   - Loading labeled tweets dataset...")
    labeled_df = pd.read_csv(labeled_file)
    print(f"     Labeled dataset: {len(labeled_df):,} rows")
    print(f"     Columns: {list(labeled_df.columns)}")
    
    # Check emotion distribution
    emotion_counts = labeled_df['emotion_label'].value_counts()
    print(f"     Emotion distribution:")
    for emotion, count in emotion_counts.items():
        print(f"       {emotion}: {count:,} tweets")
    
    # Load original dataset (chunk by chunk due to size)
    print("   - Loading original tweets dataset...")
    print("     (This may take a moment due to file size...)")
    
    # Read original file in chunks to handle large size
    original_chunks = []
    chunk_size = 10000  # Read 10k rows at a time
    
    try:
        for chunk in pd.read_csv(original_file, chunksize=chunk_size):
            original_chunks.append(chunk)
            if len(original_chunks) % 10 == 0:  # Progress indicator
                print(f"     Loaded {len(original_chunks) * chunk_size:,} rows...")
    except Exception as e:
        print(f"❌ Error loading original file: {e}")
        return None
    
    # Combine all chunks
    original_df = pd.concat(original_chunks, ignore_index=True)
    print(f"     Original dataset: {len(original_df):,} rows")
    print(f"     Columns: {list(original_df.columns)}")
    
    print("\n🎯 Creating stratified sample...")
    
    # Create stratified sample to ensure different emotions
    sample_size = 400
    samples_per_emotion = sample_size // len(emotion_counts)
    remaining_samples = sample_size % len(emotion_counts)
    
    print(f"   - Target sample size: {sample_size}")
    print(f"   - Samples per emotion: {samples_per_emotion}")
    print(f"   - Remaining samples: {remaining_samples}")
    
    # Sample from each emotion
    sampled_tweets = []
    
    for i, (emotion, count) in enumerate(emotion_counts.items()):
        emotion_tweets = labeled_df[labeled_df['emotion_label'] == emotion]
        
        # Calculate how many to sample from this emotion
        if i < remaining_samples:
            n_samples = samples_per_emotion + 1
        else:
            n_samples = samples_per_emotion
        
        # Don't sample more than available
        n_samples = min(n_samples, len(emotion_tweets))
        
        if n_samples > 0:
            emotion_sample = emotion_tweets.sample(n=n_samples, random_state=42)
            sampled_tweets.append(emotion_sample)
            print(f"     {emotion}: {n_samples} samples")
    
    # Combine all samples
    sample_df = pd.concat(sampled_tweets, ignore_index=True)
    print(f"   - Total sampled tweets: {len(sample_df):,}")
    
    print("\n🔍 Finding original tweets...")
    
    # Create comparison data
    comparison_data = []
    found_matches = 0
    
    for idx, row in sample_df.iterrows():
        # Find original tweet by date and content similarity
        date_matches = original_df[original_df['date'] == row['Date']]
        
        if len(date_matches) > 0:
            # Find best match by content similarity
            best_match = None
            best_similarity = 0
            
            for _, orig_row in date_matches.iterrows():
                # Calculate similarity between cleaned and original tweet
                cleaned_words = set(str(row['Tweet']).lower().split())
                orig_words = set(str(orig_row['tweet']).lower().split())
                
                if len(cleaned_words) > 0 and len(orig_words) > 0:
                    intersection = len(cleaned_words.intersection(orig_words))
                    union = len(cleaned_words.union(orig_words))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = orig_row
            
            if best_match is not None and best_similarity > 0.1:
                comparison_data.append({
                    'Date': row['Date'],
                    'Emotion_Label': row['emotion_label'],
                    'Emotion_Probability': row['emotion_prob'],
                    'Original_Tweet': best_match['tweet'],
                    'Cleaned_Tweet': row['Tweet'],
                    'Similarity_Score': best_similarity,
                    'Original_Length': len(str(best_match['tweet'])),
                    'Cleaned_Length': len(str(row['Tweet'])),
                    'Length_Difference': len(str(best_match['tweet'])) - len(str(row['Tweet'])),
                    'Original_Language': best_match.get('language', 'unknown'),
                    'Original_Likes': best_match.get('likes_count', 0),
                    'Original_Retweets': best_match.get('retweets_count', 0),
                    'Original_Replies': best_match.get('replies_count', 0),
                    'Original_Hashtags': best_match.get('hashtags', '[]'),
                    'Original_URLs': best_match.get('urls', '[]')
                })
                found_matches += 1
        
        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"     Processed {idx + 1:,} tweets, found {found_matches:,} matches...")
    
    print(f"   - Found {found_matches:,} matches out of {len(sample_df):,} sampled tweets")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) == 0:
        print("❌ No matches found!")
        return None
    
    # Save the comparison dataset
    print(f"\n💾 Saving comparison dataset...")
    comparison_df.to_csv(output_file, index=False)
    
    print(f"✅ Comparison dataset saved as '{output_file}'")
    
    # Generate summary statistics
    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"   - Total matched tweets: {len(comparison_df):,}")
    print(f"   - Average similarity: {comparison_df['Similarity_Score'].mean():.3f}")
    print(f"   - Average original length: {comparison_df['Original_Length'].mean():.1f} characters")
    print(f"   - Average cleaned length: {comparison_df['Cleaned_Length'].mean():.1f} characters")
    print(f"   - Average length reduction: {comparison_df['Length_Difference'].mean():.1f} characters")
    
    # Emotion distribution in sample
    print(f"\n🎭 EMOTION DISTRIBUTION IN SAMPLE:")
    emotion_sample_counts = comparison_df['Emotion_Label'].value_counts()
    for emotion, count in emotion_sample_counts.items():
        print(f"   {emotion}: {count} tweets")
    
    # Language distribution
    if 'Original_Language' in comparison_df.columns:
        print(f"\n🌍 LANGUAGE DISTRIBUTION:")
        language_counts = comparison_df['Original_Language'].value_counts()
        for lang, count in language_counts.head(5).items():
            print(f"   {lang}: {count} tweets")
    
    # Engagement statistics
    print(f"\n📈 ENGAGEMENT STATISTICS:")
    print(f"   - Average likes: {comparison_df['Original_Likes'].mean():.1f}")
    print(f"   - Average retweets: {comparison_df['Original_Retweets'].mean():.1f}")
    print(f"   - Average replies: {comparison_df['Original_Replies'].mean():.1f}")
    
    # Show some examples
    print(f"\n📝 SAMPLE COMPARISONS:")
    print("-" * 100)
    
    for i, row in comparison_df.head(3).iterrows():
        print(f"\nExample {i+1} ({row['Emotion_Label']}):")
        print(f"  Original:  {str(row['Original_Tweet'])[:200]}...")
        print(f"  Cleaned:   {str(row['Cleaned_Tweet'])[:200]}...")
        print(f"  Emotion: {row['Emotion_Label']} (confidence: {row['Emotion_Probability']:.3f})")
        print(f"  Similarity: {row['Similarity_Score']:.3f}")
        print(f"  Length: {row['Original_Length']} → {row['Cleaned_Length']} ({row['Length_Difference']:+d})")
        print(f"  Engagement: {row['Original_Likes']} likes, {row['Original_Retweets']} retweets")
        print("-" * 100)
    
    return comparison_df

def create_high_quality_sample():
    """
    Create a smaller high-quality sample for detailed analysis.
    """
    
    print("\n🔍 CREATING HIGH-QUALITY SAMPLE")
    print("=" * 40)
    
    comparison_file = 'labeled_tweets_ai_comparison.csv'
    if not os.path.exists(comparison_file):
        print(f"❌ Error: {comparison_file} not found.")
        return None
    
    comparison_df = pd.read_csv(comparison_file)
    
    # Filter for high-quality matches (high similarity)
    high_quality = comparison_df[comparison_df['Similarity_Score'] > 0.3]
    
    # Create a balanced sample across emotions
    emotion_samples = []
    samples_per_emotion = 20  # 20 samples per emotion
    
    for emotion in comparison_df['Emotion_Label'].unique():
        emotion_data = high_quality[high_quality['Emotion_Label'] == emotion]
        if len(emotion_data) > 0:
            sample_size = min(samples_per_emotion, len(emotion_data))
            emotion_sample = emotion_data.sample(n=sample_size, random_state=42)
            emotion_samples.append(emotion_sample)
    
    if emotion_samples:
        high_quality_sample = pd.concat(emotion_samples, ignore_index=True)
    else:
        high_quality_sample = high_quality.head(50)
    
    sample_file = 'high_quality_labeled_tweets_ai_sample.csv'
    high_quality_sample.to_csv(sample_file, index=False)
    
    print(f"✅ High-quality sample saved as '{sample_file}'")
    print(f"   - Sample size: {len(high_quality_sample):,} high-quality matches")
    print(f"   - Average similarity: {high_quality_sample['Similarity_Score'].mean():.3f}")
    
    # Show emotion distribution
    print(f"\n🎭 HIGH-QUALITY SAMPLE EMOTION DISTRIBUTION:")
    emotion_counts = high_quality_sample['Emotion_Label'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count} tweets")
    
    return high_quality_sample

def main():
    """Main function to create labeled tweets AI comparison."""
    print("🎭 LABELED TWEETS AI COMPARISON TOOL")
    print("=" * 50)
    print("This tool creates a sample of 400 tweets with different emotions")
    print("from the labeled dataset and finds their original versions.")
    print("=" * 50)
    
    # Create the main comparison
    comparison_df = create_labeled_tweets_ai_comparison()
    if comparison_df is None:
        print("❌ Failed to create comparison dataset")
        return
    
    # Create high-quality sample
    sample_df = create_high_quality_sample()
    if sample_df is None:
        print("❌ Failed to create high-quality sample")
        return
    
    print("\n" + "=" * 60)
    print("🎉 LABELED TWEETS AI COMPARISON COMPLETED!")
    print("=" * 60)
    print("✅ Created emotion-stratified sample")
    print("✅ Found original tweets for comparison")
    print("✅ Created side-by-side comparison")
    print()
    print("📁 Output files:")
    print("   - labeled_tweets_ai_comparison.csv (full sample)")
    print("   - high_quality_labeled_tweets_ai_sample.csv (high-quality subset)")
    print()
    print("🔍 You can now:")
    print("   - See original tweets vs cleaned versions")
    print("   - Analyze emotion distribution")
    print("   - Compare text changes by emotion")
    print("   - Study cleaning impact on different emotions")
    print("   - Analyze engagement patterns by emotion")

if __name__ == "__main__":
    main()
