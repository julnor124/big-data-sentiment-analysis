#!/usr/bin/env python3
"""
Improved Matching Algorithm with Timestamps
===========================================

This script demonstrates how to use timestamps for more precise matching
between cleaned and original tweets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

def improved_timestamp_matching(cleaned_df, original_df, time_window_minutes=5):
    """
    Improved matching using timestamps with configurable time window.
    
    Args:
        cleaned_df: DataFrame with cleaned tweets
        original_df: DataFrame with original tweets
        time_window_minutes: Time window for matching (default: 5 minutes)
    
    Returns:
        DataFrame with matched tweets and similarity scores
    """
    
    print(f"🔍 IMPROVED TIMESTAMP MATCHING")
    print(f"Time window: ±{time_window_minutes} minutes")
    print("=" * 50)
    
    # Convert date columns to datetime
    cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
    original_df['created_at'] = pd.to_datetime(original_df['created_at'])
    
    comparison_data = []
    found_matches = 0
    total_processed = 0
    
    for idx, row in cleaned_df.iterrows():
        total_processed += 1
        
        # Define time window
        target_time = row['Date']
        time_start = target_time - timedelta(minutes=time_window_minutes)
        time_end = target_time + timedelta(minutes=time_window_minutes)
        
        # Find tweets within time window
        time_matches = original_df[
            (original_df['created_at'] >= time_start) &
            (original_df['created_at'] <= time_end)
        ]
        
        if len(time_matches) > 0:
            # Find best match by content similarity
            best_match = None
            best_similarity = 0
            best_time_diff = None
            
            for _, orig_row in time_matches.iterrows():
                # Calculate time difference
                time_diff = abs((orig_row['created_at'] - target_time).total_seconds())
                
                # Calculate content similarity
                similarity = calculate_text_similarity(
                    row['Tweet'], 
                    orig_row['tweet']
                )
                
                # Combined score: similarity + time proximity
                time_score = 1 / (1 + time_diff / 60)  # Decay over minutes
                combined_score = 0.7 * similarity + 0.3 * time_score
                
                if combined_score > best_similarity:
                    best_similarity = combined_score
                    best_match = orig_row
                    best_time_diff = time_diff
            
            if best_match is not None and best_similarity > 0.1:
                comparison_data.append({
                    'Date': row['Date'],
                    'Original_Time': best_match['created_at'],
                    'Time_Difference_Seconds': best_time_diff,
                    'Original_Tweet': best_match['tweet'],
                    'Cleaned_Tweet': row['Tweet'],
                    'Similarity_Score': best_similarity,
                    'Original_Length': len(str(best_match['tweet'])),
                    'Cleaned_Length': len(str(row['Tweet'])),
                    'Length_Difference': len(str(best_match['tweet'])) - len(str(row['Tweet'])),
                    'Time_Proximity_Score': 1 / (1 + best_time_diff / 60)
                })
                found_matches += 1
        
        # Progress indicator
        if total_processed % 100 == 0:
            print(f"   Processed {total_processed:,} tweets, found {found_matches:,} matches...")
    
    print(f"✅ Found {found_matches:,} matches out of {total_processed:,} tweets")
    print(f"   Match rate: {found_matches/total_processed*100:.1f}%")
    
    return pd.DataFrame(comparison_data)

def calculate_text_similarity(text1, text2):
    """
    Calculate Jaccard similarity between two texts.
    """
    if pd.isna(text1) or pd.isna(text2):
        return 0
    
    # Convert to lowercase and split into words
    words1 = set(str(text1).lower().split())
    words2 = set(str(text2).lower().split())
    
    if len(words1) == 0 and len(words2) == 0:
        return 1.0
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

def analyze_timestamp_precision(cleaned_df, original_df):
    """
    Analyze the precision improvement from using timestamps.
    """
    print("\n📊 TIMESTAMP PRECISION ANALYSIS")
    print("=" * 40)
    
    # Convert to datetime
    cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
    original_df['created_at'] = pd.to_datetime(original_df['created_at'])
    
    # Analyze date-only matching
    date_only_matches = 0
    timestamp_matches = 0
    
    for idx, row in cleaned_df.head(100).iterrows():  # Sample first 100
        target_date = row['Date'].date()
        
        # Date-only matching
        date_matches = original_df[original_df['created_at'].dt.date == target_date]
        date_only_matches += len(date_matches)
        
        # Timestamp matching (±5 minutes)
        time_start = row['Date'] - timedelta(minutes=5)
        time_end = row['Date'] + timedelta(minutes=5)
        time_matches = original_df[
            (original_df['created_at'] >= time_start) &
            (original_df['created_at'] <= time_end)
        timestamp_matches += len(time_matches)
    
    print(f"Date-only matching: {date_only_matches:,} candidates")
    print(f"Timestamp matching: {timestamp_matches:,} candidates")
    print(f"Precision improvement: {date_only_matches/timestamp_matches:.1f}x more precise")

def create_optimized_matching_script():
    """
    Create an optimized version of the matching scripts.
    """
    
    print("\n🚀 OPTIMIZED MATCHING RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = """
    RECOMMENDED IMPROVEMENTS:
    
    1. TIMESTAMP MATCHING:
       - Use ±5 minute time windows for high precision
       - Combine content similarity + time proximity scores
       - Handle timezone differences properly
    
    2. ADAPTIVE TIME WINDOWS:
       - Start with ±5 minutes
       - Expand to ±15 minutes if no matches
       - Fall back to date-only if still no matches
    
    3. MULTI-CRITERIA SCORING:
       - 70% content similarity
       - 30% time proximity
       - Bonus for exact timestamp matches
    
    4. PERFORMANCE OPTIMIZATIONS:
       - Index timestamp columns
       - Use vectorized operations
       - Parallel processing for large datasets
    
    5. ERROR HANDLING:
       - Handle missing timestamps
       - Timezone conversion
       - Invalid date formats
    """
    
    print(recommendations)

def main():
    """Main function to demonstrate improved matching."""
    
    print("🕐 IMPROVED TIMESTAMP MATCHING ALGORITHM")
    print("=" * 60)
    print("This demonstrates how to use timestamps for more precise matching.")
    print("=" * 60)
    
    # Show recommendations
    create_optimized_matching_script()
    
    print("\n💡 IMPLEMENTATION STEPS:")
    print("1. Update your matching scripts to use timestamps")
    print("2. Implement adaptive time windows")
    print("3. Add time proximity scoring")
    print("4. Test with your datasets")
    print("5. Compare match quality vs. current approach")

if __name__ == "__main__":
    main()
