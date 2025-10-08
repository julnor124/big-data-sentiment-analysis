#!/usr/bin/env python3
"""
Analyze Emotion Labeling Quality
===============================

This script helps you evaluate the quality of emotion labeling
by analyzing the comparison data and providing quality metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_emotion_quality():
    """
    Analyze the quality of emotion labeling from comparison data.
    """
    
    print("🎭 EMOTION LABELING QUALITY ANALYSIS")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # File paths
    comparison_file = 'emotion_sample_comparison_improved.csv'
    high_quality_file = 'high_quality_emotion_sample_improved.csv'
    
    print("📂 Loading comparison data...")
    
    # Check if files exist
    if not pd.io.common.file_exists(comparison_file):
        print(f"❌ Error: {comparison_file} not found.")
        print("   Run the improved emotion sample script first.")
        return None
    
    # Load data
    comparison_df = pd.read_csv(comparison_file)
    print(f"   Loaded {len(comparison_df):,} comparison records")
    
    if pd.io.common.file_exists(high_quality_file):
        high_quality_df = pd.read_csv(high_quality_file)
        print(f"   Loaded {len(high_quality_df):,} high-quality records")
    else:
        high_quality_df = None
        print("   High-quality file not found")
    
    print("\n📊 EMOTION LABELING QUALITY METRICS")
    print("=" * 50)
    
    # 1. Overall Quality Metrics
    print("1. OVERALL QUALITY METRICS:")
    print(f"   - Total samples: {len(comparison_df):,}")
    print(f"   - Average similarity: {comparison_df['Similarity_Score'].mean():.3f}")
    print(f"   - Average emotion probability: {comparison_df['Emotion_Probability'].mean():.3f}")
    print(f"   - High confidence labels (>0.8): {len(comparison_df[comparison_df['Emotion_Probability'] > 0.8]):,} ({len(comparison_df[comparison_df['Emotion_Probability'] > 0.8])/len(comparison_df)*100:.1f}%)")
    print(f"   - Medium confidence labels (0.5-0.8): {len(comparison_df[(comparison_df['Emotion_Probability'] >= 0.5) & (comparison_df['Emotion_Probability'] <= 0.8)]):,} ({len(comparison_df[(comparison_df['Emotion_Probability'] >= 0.5) & (comparison_df['Emotion_Probability'] <= 0.8)])/len(comparison_df)*100:.1f}%)")
    print(f"   - Low confidence labels (<0.5): {len(comparison_df[comparison_df['Emotion_Probability'] < 0.5]):,} ({len(comparison_df[comparison_df['Emotion_Probability'] < 0.5])/len(comparison_df)*100:.1f}%)")
    
    # 2. Emotion Distribution Analysis
    print("\n2. EMOTION DISTRIBUTION ANALYSIS:")
    emotion_counts = comparison_df['Emotion_Label'].value_counts()
    for emotion, count in emotion_counts.items():
        percentage = count / len(comparison_df) * 100
        avg_prob = comparison_df[comparison_df['Emotion_Label'] == emotion]['Emotion_Probability'].mean()
        print(f"   {emotion}: {count} tweets ({percentage:.1f}%) - Avg confidence: {avg_prob:.3f}")
    
    # 3. Quality by Emotion
    print("\n3. QUALITY BY EMOTION:")
    for emotion in comparison_df['Emotion_Label'].unique():
        emotion_data = comparison_df[comparison_df['Emotion_Label'] == emotion]
        print(f"   {emotion}:")
        print(f"     - Count: {len(emotion_data)}")
        print(f"     - Avg similarity: {emotion_data['Similarity_Score'].mean():.3f}")
        print(f"     - Avg confidence: {emotion_data['Emotion_Probability'].mean():.3f}")
        print(f"     - High confidence: {len(emotion_data[emotion_data['Emotion_Probability'] > 0.8])} ({len(emotion_data[emotion_data['Emotion_Probability'] > 0.8])/len(emotion_data)*100:.1f}%)")
    
    # 4. Text Length Analysis
    print("\n4. TEXT LENGTH ANALYSIS:")
    print(f"   - Average original length: {comparison_df['Original_Length'].mean():.1f} characters")
    print(f"   - Average cleaned length: {comparison_df['Cleaned_Length'].mean():.1f} characters")
    print(f"   - Average length reduction: {comparison_df['Length_Difference'].mean():.1f} characters")
    print(f"   - Length reduction percentage: {(comparison_df['Length_Difference'].mean() / comparison_df['Original_Length'].mean()) * 100:.1f}%")
    
    # 5. Time Precision Analysis
    if 'Time_Difference_Seconds' in comparison_df.columns:
        print("\n5. TIME PRECISION ANALYSIS:")
        print(f"   - Average time difference: {comparison_df['Time_Difference_Seconds'].mean():.1f} seconds")
        print(f"   - Perfect matches (0s): {len(comparison_df[comparison_df['Time_Difference_Seconds'] == 0])} ({len(comparison_df[comparison_df['Time_Difference_Seconds'] == 0])/len(comparison_df)*100:.1f}%)")
        print(f"   - Within 1 minute: {len(comparison_df[comparison_df['Time_Difference_Seconds'] <= 60])} ({len(comparison_df[comparison_df['Time_Difference_Seconds'] <= 60])/len(comparison_df)*100:.1f}%)")
    
    # 6. Quality Recommendations
    print("\n6. QUALITY RECOMMENDATIONS:")
    
    high_confidence_pct = len(comparison_df[comparison_df['Emotion_Probability'] > 0.8]) / len(comparison_df) * 100
    avg_similarity = comparison_df['Similarity_Score'].mean()
    avg_confidence = comparison_df['Emotion_Probability'].mean()
    
    if high_confidence_pct > 70:
        print("   ✅ EXCELLENT: High confidence labels (>70%)")
    elif high_confidence_pct > 50:
        print("   ✅ GOOD: Moderate confidence labels (50-70%)")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT: Low confidence labels (<50%)")
    
    if avg_similarity > 0.5:
        print("   ✅ EXCELLENT: High text similarity (>0.5)")
    elif avg_similarity > 0.3:
        print("   ✅ GOOD: Moderate text similarity (0.3-0.5)")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT: Low text similarity (<0.3)")
    
    if avg_confidence > 0.7:
        print("   ✅ EXCELLENT: High emotion confidence (>0.7)")
    elif avg_confidence > 0.5:
        print("   ✅ GOOD: Moderate emotion confidence (0.5-0.7)")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT: Low emotion confidence (<0.5)")
    
    # 7. Sample Review
    print("\n7. SAMPLE REVIEW - Check these examples:")
    print("=" * 60)
    
    # Show examples for each emotion
    for emotion in comparison_df['Emotion_Label'].unique():
        emotion_data = comparison_df[comparison_df['Emotion_Label'] == emotion]
        # Get highest confidence example
        best_example = emotion_data.loc[emotion_data['Emotion_Probability'].idxmax()]
        
        print(f"\n{emotion.upper()} (Confidence: {best_example['Emotion_Probability']:.3f}):")
        print(f"  Original:  {str(best_example['Original_Tweet'])[:150]}...")
        print(f"  Cleaned:   {str(best_example['Cleaned_Tweet'])[:150]}...")
        print(f"  Similarity: {best_example['Similarity_Score']:.3f}")
        print("-" * 60)
    
    # 8. Quality Score
    print("\n8. OVERALL QUALITY SCORE:")
    quality_score = (
        (high_confidence_pct / 100) * 0.4 +  # 40% weight on confidence
        (avg_similarity) * 0.3 +              # 30% weight on similarity
        (avg_confidence) * 0.3                # 30% weight on average confidence
    ) * 100
    
    print(f"   Quality Score: {quality_score:.1f}/100")
    
    if quality_score > 80:
        print("   🎉 EXCELLENT: Emotion labeling is very reliable!")
    elif quality_score > 60:
        print("   ✅ GOOD: Emotion labeling is generally reliable")
    elif quality_score > 40:
        print("   ⚠️  MODERATE: Emotion labeling needs some improvement")
    else:
        print("   ❌ POOR: Emotion labeling needs significant improvement")
    
    return comparison_df

def create_quality_report():
    """
    Create a detailed quality report.
    """
    
    print("\n📋 CREATING QUALITY REPORT...")
    
    # This would create a detailed report file
    report_content = """
EMOTION LABELING QUALITY REPORT
==============================

This report helps you evaluate the quality of emotion labeling
in your dataset. Use these metrics to assess reliability:

1. HIGH CONFIDENCE LABELS (>0.8): Most reliable
2. MEDIUM CONFIDENCE LABELS (0.5-0.8): Generally reliable  
3. LOW CONFIDENCE LABELS (<0.5): May need review

RECOMMENDATIONS:
- Focus on high-confidence labels for analysis
- Review low-confidence labels manually
- Consider re-labeling if quality score < 60
- Use similarity scores to validate matches
    """
    
    with open('emotion_quality_report.txt', 'w') as f:
        f.write(report_content)
    
    print("✅ Quality report saved as 'emotion_quality_report.txt'")

def main():
    """Main function to analyze emotion quality."""
    
    print("🎭 EMOTION LABELING QUALITY ANALYZER")
    print("=" * 50)
    print("This tool helps you evaluate the quality of emotion labeling")
    print("by analyzing similarity scores, confidence levels, and more.")
    print("=" * 50)
    
    # Analyze quality
    comparison_df = analyze_emotion_quality()
    
    if comparison_df is not None:
        # Create quality report
        create_quality_report()
        
        print("\n" + "=" * 60)
        print("🎉 EMOTION QUALITY ANALYSIS COMPLETED!")
        print("=" * 60)
        print("✅ Analyzed emotion labeling quality")
        print("✅ Generated quality metrics")
        print("✅ Provided recommendations")
        print("✅ Created quality report")
        print()
        print("🔍 You can now:")
        print("   - Review high-confidence labels")
        print("   - Check low-confidence labels manually")
        print("   - Use quality scores for filtering")
        print("   - Make decisions about data reliability")

if __name__ == "__main__":
    main()
