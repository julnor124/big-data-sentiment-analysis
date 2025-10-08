#!/usr/bin/env python3
"""
Create Emotion Sample Comparison
===============================

This script creates a sample of 400 random tweets with different emotions
from AfterChatGPT.labeled.csv, finds their original tweets from source files,
and creates a side-by-side comparison CSV.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_emotion_sample_comparison():
    """
    Create a sample of 400 tweets with different emotions and find their originals.
    """
    
    print("🎭 CREATING EMOTION SAMPLE COMPARISON")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # File paths
    labeled_file = 'Labeling/AfterChatGPT.labeled.csv'
    original_chatgpt = 'EDA_ChatGPT/ChatGPT_pre_clean.csv'
    original_genai = 'EDA_GenAI/generativeaiopinion_pre_clean.csv'
    output_file = 'emotion_sample_comparison.csv'
    
    print("📂 Loading datasets...")
    
    # Check if files exist
    files_to_check = [labeled_file, original_chatgpt, original_genai]
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found.")
            return None
    
    # Load labeled dataset
    print("   - Loading labeled dataset...")
    labeled_df = pd.read_csv(labeled_file)
    print(f"     Labeled dataset: {len(labeled_df):,} rows")
    
    # Check emotion distribution
    emotion_counts = labeled_df['emotion_label'].value_counts()
    print(f"     Emotion distribution:")
    for emotion, count in emotion_counts.items():
        print(f"       {emotion}: {count:,} tweets")
    
    # Load original datasets
    print("   - Loading original ChatGPT dataset...")
    orig_chatgpt_df = pd.read_csv(original_chatgpt)
    print(f"     Original ChatGPT: {len(orig_chatgpt_df):,} rows")
    
    print("   - Loading original GenerativeAI dataset...")
    orig_genai_df = pd.read_csv(original_genai)
    print(f"     Original GenerativeAI: {len(orig_genai_df):,} rows")
    
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
        # Determine which original dataset to search
        if row['Source'] == 'ChatGPT':
            original_df = orig_chatgpt_df
        else:
            original_df = orig_genai_df
        
        # IMPROVED: Use timestamp matching with ±5 minute window
        target_time = row['Date']
        time_start = target_time - pd.Timedelta(minutes=5)
        time_end = target_time + pd.Timedelta(minutes=5)
        
        # Find tweets within time window
        time_matches = original_df[
            (original_df['Date'] >= time_start) &
            (original_df['Date'] <= time_end)
        ]
        
        if len(time_matches) > 0:
            # Find best match by content similarity + time proximity
            best_match = None
            best_similarity = 0
            best_time_diff = None
            
            for _, orig_row in time_matches.iterrows():
                # Calculate time difference
                time_diff = abs((orig_row['Date'] - target_time).total_seconds())
                
                # Calculate content similarity
                cleaned_words = set(row['Tweet'].lower().split())
                orig_words = set(orig_row['Tweet'].lower().split())
                
                if len(cleaned_words) > 0 and len(orig_words) > 0:
                    intersection = len(cleaned_words.intersection(orig_words))
                    union = len(cleaned_words.union(orig_words))
                    similarity = intersection / union if union > 0 else 0
                    
                    # Combined score: 70% content similarity + 30% time proximity
                    time_score = 1 / (1 + time_diff / 300)  # Decay over 5 minutes
                    combined_score = 0.7 * similarity + 0.3 * time_score
                    
                    if combined_score > best_similarity:
                        best_similarity = combined_score
                        best_match = orig_row
                        best_time_diff = time_diff
            
            if best_match is not None and best_similarity > 0.1:
                comparison_data.append({
                    'Date': row['Date'],
                    'Source': row['Source'],
                    'Emotion_Label': row['emotion_label'],
                    'Emotion_Probability': row['emotion_prob'],
                    'Original_Tweet': best_match['Tweet'],
                    'Cleaned_Tweet': row['Tweet'],
                    'Similarity_Score': best_similarity,
                    'Time_Difference_Seconds': best_time_diff,
                    'Time_Proximity_Score': 1 / (1 + best_time_diff / 300),
                    'Original_Length': len(best_match['Tweet']),
                    'Cleaned_Length': len(row['Tweet']),
                    'Length_Difference': len(best_match['Tweet']) - len(row['Tweet'])
                })
                found_matches += 1
    
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
    
    # Show some examples
    print(f"\n📝 SAMPLE COMPARISONS:")
    print("-" * 100)
    
    for i, row in comparison_df.head(3).iterrows():
        print(f"\nExample {i+1} ({row['Source']} - {row['Emotion_Label']}):")
        print(f"  Original:  {row['Original_Tweet'][:200]}...")
        print(f"  Cleaned:   {row['Cleaned_Tweet'][:200]}...")
        print(f"  Emotion: {row['Emotion_Label']} (confidence: {row['Emotion_Probability']:.3f})")
        print(f"  Similarity: {row['Similarity_Score']:.3f}")
        print(f"  Length: {row['Original_Length']} → {row['Cleaned_Length']} ({row['Length_Difference']:+d})")
        print("-" * 100)
    
    return comparison_df

def create_high_quality_sample():
    """
    Create a smaller high-quality sample for detailed analysis.
    """
    
    print("\n🔍 CREATING HIGH-QUALITY SAMPLE")
    print("=" * 40)
    
    comparison_file = 'emotion_sample_comparison.csv'
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
    
    sample_file = 'high_quality_emotion_sample.csv'
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
    """Main function to create emotion sample comparison."""
    print("🎭 EMOTION SAMPLE COMPARISON TOOL")
    print("=" * 50)
    print("This tool creates a sample of 400 tweets with different emotions")
    print("and finds their original versions for side-by-side comparison.")
    print("=" * 50)
    
    # Create the main comparison
    comparison_df = create_emotion_sample_comparison()
    if comparison_df is None:
        print("❌ Failed to create comparison dataset")
        return
    
    # Create high-quality sample
    sample_df = create_high_quality_sample()
    if sample_df is None:
        print("❌ Failed to create high-quality sample")
        return
    
    print("\n" + "=" * 60)
    print("🎉 EMOTION SAMPLE COMPARISON COMPLETED!")
    print("=" * 60)
    print("✅ Created emotion-stratified sample")
    print("✅ Found original tweets for comparison")
    print("✅ Created side-by-side comparison")
    print()
    print("📁 Output files:")
    print("   - emotion_sample_comparison.csv (full sample)")
    print("   - high_quality_emotion_sample.csv (high-quality subset)")
    print()
    print("🔍 You can now:")
    print("   - See original tweets vs cleaned versions")
    print("   - Analyze emotion distribution")
    print("   - Compare text changes by emotion")
    print("   - Study cleaning impact on different emotions")

if __name__ == "__main__":
    main()
