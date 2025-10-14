#!/usr/bin/env python3
"""
Calculate Weighted Accuracy/Error Rates
========================================
Adjusts for class imbalance by weighting errors based on actual emotion distributions.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def get_actual_emotion_distribution(labeled_file):
    """Get the actual emotion distribution from the full labeled dataset"""
    df = pd.read_csv(labeled_file)
    emotion_counts = df['emotion_label'].value_counts()
    emotion_proportions = emotion_counts / len(df)
    return emotion_counts, emotion_proportions

def calculate_weighted_metrics(sample_file, evaluation_excel, labeled_full_file, dataset_name):
    """
    Calculate weighted accuracy metrics
    
    Args:
        sample_file: CSV with the balanced sample (400 tweets)
        evaluation_excel: Excel file with rows marked as incorrect
        labeled_full_file: Full labeled dataset to get true distribution
        dataset_name: Name of the dataset for reporting
    """
    
    print(f"\n{'='*80}")
    print(f"WEIGHTED ACCURACY ANALYSIS: {dataset_name}")
    print(f"{'='*80}")
    
    # Load sample data
    sample_df = pd.read_csv(sample_file)
    print(f"\nSample size: {len(sample_df)} tweets")
    
    # Load evaluation (incorrect rows)
    eval_df = pd.read_excel(evaluation_excel)
    print(f"Rows marked as incorrect: {len(eval_df)}")
    
    # Get emotion distribution in sample
    sample_emotion_counts = sample_df['Emotion_Label'].value_counts()
    print(f"\nEmotion distribution in SAMPLE (balanced):")
    for emotion, count in sorted(sample_emotion_counts.items()):
        pct = (count / len(sample_df)) * 100
        print(f"  {emotion}: {count} tweets ({pct:.1f}%)")
    
    # Get emotion distribution in evaluation (errors)
    # Handle different column names
    emotion_col = 'Emotion' if 'Emotion' in eval_df.columns else 'Emotion_Label'
    
    # Filter out non-string values (like datetime objects that might be in the column)
    eval_df_clean = eval_df[eval_df[emotion_col].apply(lambda x: isinstance(x, str))].copy()
    
    eval_emotion_counts = eval_df_clean[emotion_col].value_counts()
    print(f"\nEmotion distribution in ERRORS (after cleaning {len(eval_df) - len(eval_df_clean)} invalid rows):")
    for emotion, count in sorted(eval_emotion_counts.items()):
        pct = (count / len(eval_df_clean)) * 100
        print(f"  {emotion}: {count} errors ({pct:.1f}%)")
    
    # Get actual emotion distribution from full dataset
    actual_counts, actual_proportions = get_actual_emotion_distribution(labeled_full_file)
    print(f"\nActual emotion distribution in FULL DATASET:")
    for emotion, count in sorted(actual_counts.items()):
        pct = actual_proportions[emotion] * 100
        print(f"  {emotion}: {count:,} tweets ({pct:.1f}%)")
    
    # Calculate per-emotion error rates
    print(f"\n{'='*80}")
    print("PER-EMOTION ERROR RATES (from balanced sample):")
    print(f"{'='*80}")
    
    emotion_error_rates = {}
    for emotion in sample_emotion_counts.index:
        # Number of this emotion in sample
        emotion_sample_count = sample_emotion_counts.get(emotion, 0)
        
        # Number of errors for this emotion
        emotion_error_count = eval_emotion_counts.get(emotion, 0)
        
        # Error rate for this emotion
        if emotion_sample_count > 0:
            error_rate = emotion_error_count / emotion_sample_count
        else:
            error_rate = 0
        
        emotion_error_rates[emotion] = error_rate
        
        accuracy_rate = 1 - error_rate
        print(f"  {emotion}:")
        print(f"    Sample: {emotion_sample_count} tweets")
        print(f"    Errors: {emotion_error_count}")
        print(f"    Error rate: {error_rate*100:.1f}%")
        print(f"    Accuracy: {accuracy_rate*100:.1f}%")
    
    # Calculate UNWEIGHTED metrics (traditional)
    print(f"\n{'='*80}")
    print("UNWEIGHTED METRICS (traditional, assumes balanced data):")
    print(f"{'='*80}")
    
    total_errors = len(eval_df)
    total_sample = len(sample_df)
    unweighted_error_rate = total_errors / total_sample
    unweighted_accuracy = 1 - unweighted_error_rate
    
    print(f"  Total errors: {total_errors}/{total_sample}")
    print(f"  Error rate: {unweighted_error_rate*100:.1f}%")
    print(f"  Accuracy: {unweighted_accuracy*100:.1f}%")
    
    # Calculate WEIGHTED metrics (accounts for class imbalance)
    print(f"\n{'='*80}")
    print("WEIGHTED METRICS (adjusted for actual class distribution):")
    print(f"{'='*80}")
    
    weighted_error_rate = 0
    weighted_accuracy = 0
    
    print(f"\nCalculation breakdown:")
    for emotion in emotion_error_rates.keys():
        error_rate = emotion_error_rates[emotion]
        actual_weight = actual_proportions.get(emotion, 0)
        
        contribution_to_error = error_rate * actual_weight
        contribution_to_accuracy = (1 - error_rate) * actual_weight
        
        weighted_error_rate += contribution_to_error
        weighted_accuracy += contribution_to_accuracy
        
        print(f"  {emotion}:")
        print(f"    Error rate: {error_rate*100:.1f}%")
        print(f"    Actual proportion in dataset: {actual_weight*100:.1f}%")
        print(f"    Weighted contribution to error: {contribution_to_error*100:.2f}%")
        print(f"    Weighted contribution to accuracy: {contribution_to_accuracy*100:.2f}%")
    
    print(f"\n  FINAL WEIGHTED METRICS:")
    print(f"    Weighted Error Rate: {weighted_error_rate*100:.2f}%")
    print(f"    Weighted Accuracy: {weighted_accuracy*100:.2f}%")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON:")
    print(f"{'='*80}")
    print(f"  Unweighted Accuracy: {unweighted_accuracy*100:.1f}%")
    print(f"  Weighted Accuracy:   {weighted_accuracy*100:.1f}%")
    print(f"  Difference:          {(weighted_accuracy - unweighted_accuracy)*100:.1f} percentage points")
    
    if weighted_accuracy > unweighted_accuracy:
        print(f"\n  ✓ The model performs BETTER than the unweighted accuracy suggests")
        print(f"    (The model is better at predicting common emotions like neutral)")
    elif weighted_accuracy < unweighted_accuracy:
        print(f"\n  ✗ The model performs WORSE than the unweighted accuracy suggests")
        print(f"    (The model struggles more with common emotions)")
    else:
        print(f"\n  = The weighted and unweighted accuracies are the same")
    
    return {
        'dataset': dataset_name,
        'sample_size': total_sample,
        'total_errors': total_errors,
        'unweighted_accuracy': unweighted_accuracy,
        'unweighted_error_rate': unweighted_error_rate,
        'weighted_accuracy': weighted_accuracy,
        'weighted_error_rate': weighted_error_rate,
        'emotion_error_rates': emotion_error_rates,
        'actual_proportions': actual_proportions.to_dict()
    }

def generate_report(results_list):
    """Generate a comprehensive report"""
    
    report = f"""
WEIGHTED ACCURACY ANALYSIS REPORT
==================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
--------
This analysis adjusts accuracy metrics for class imbalance by weighting errors
based on the actual emotion distribution in the full dataset.

WHY WEIGHTED METRICS?
---------------------
- The evaluation sample has balanced emotions (equal numbers of each)
- The actual dataset is heavily imbalanced (much more neutral than disgust)
- Traditional accuracy gives equal weight to all emotions
- Weighted accuracy reflects real-world performance better

METHODOLOGY
-----------
For each emotion:
1. Calculate error rate from balanced sample
2. Get actual proportion in full dataset
3. Weight the error rate by actual proportion
4. Sum weighted errors across all emotions

Formula: Weighted Error Rate = Σ(error_rate_i × proportion_i)
where i = each emotion category

"""
    
    for results in results_list:
        report += f"""
{'='*80}
DATASET: {results['dataset']}
{'='*80}

Sample Evaluation:
- Sample size: {results['sample_size']} tweets (balanced)
- Incorrect labels: {results['total_errors']}

Unweighted Metrics (traditional):
- Accuracy: {results['unweighted_accuracy']*100:.1f}%
- Error rate: {results['unweighted_error_rate']*100:.1f}%

Weighted Metrics (adjusted for class imbalance):
- Accuracy: {results['weighted_accuracy']*100:.2f}%
- Error rate: {results['weighted_error_rate']*100:.2f}%

Difference: {(results['weighted_accuracy'] - results['unweighted_accuracy'])*100:.1f} percentage points

Per-Emotion Analysis:
"""
        
        for emotion in sorted(results['emotion_error_rates'].keys()):
            error_rate = results['emotion_error_rates'][emotion]
            proportion = results['actual_proportions'].get(emotion, 0)
            report += f"""
  {emotion.upper()}:
    Error rate in sample: {error_rate*100:.1f}%
    Accuracy in sample: {(1-error_rate)*100:.1f}%
    Actual proportion in dataset: {proportion*100:.1f}%
    Weighted contribution: {error_rate * proportion * 100:.2f}% to total error
"""
    
    # Overall summary
    report += f"""

OVERALL FINDINGS
----------------
"""
    
    for results in results_list:
        diff = (results['weighted_accuracy'] - results['unweighted_accuracy']) * 100
        direction = "better" if diff > 0 else "worse" if diff < 0 else "same"
        
        report += f"""
{results['dataset']}:
  - Unweighted accuracy: {results['unweighted_accuracy']*100:.1f}%
  - Weighted accuracy: {results['weighted_accuracy']*100:.2f}%
  - Model performs {direction} than unweighted metrics suggest ({abs(diff):.1f}pp difference)
"""
    
    report += f"""

INTERPRETATION
--------------

The weighted metrics provide a more realistic view of model performance because:

1. REFLECTS REAL USAGE: Weights errors by how often each emotion appears
2. PRIORITIZES COMMON EMOTIONS: Better performance on frequent emotions (neutral)
   matters more than rare ones (disgust)
3. PRODUCTION ACCURACY: Shows expected performance on real data distribution

RECOMMENDATIONS
---------------

1. If weighted accuracy > unweighted:
   ✓ Model is good at common emotions
   ✓ Acceptable for production use
   ✗ May struggle with rare emotions - monitor these

2. If weighted accuracy < unweighted:
   ✗ Model struggles with common emotions
   ⚠ Consider improving model for frequent categories
   ✓ Good at rare emotions (but less impactful)

3. For balanced performance:
   - Use class weights during training
   - Oversample rare emotions
   - Use stratified sampling for evaluation
"""
    
    return report

def main():
    """Main analysis function"""
    
    print("="*80)
    print("WEIGHTED ACCURACY CALCULATOR")
    print("="*80)
    print("Adjusting accuracy metrics for class imbalance")
    print()
    
    results = []
    
    # Analysis 1: Tweets AI dataset
    print("\n" + "="*80)
    print("ANALYZING: Tweets AI Dataset (Pre-ChatGPT)")
    print("="*80)
    
    result1 = calculate_weighted_metrics(
        sample_file='Sample_Analysis/labeled_tweets_ai_comparison.csv',
        evaluation_excel='Evaluation_ Tweets_AI_Labeling .xlsx',
        labeled_full_file='Labeling/clean_tweets_ai.labeled.csv',
        dataset_name='Tweets AI (Pre-ChatGPT)'
    )
    results.append(result1)
    
    # Analysis 2: AfterChatGPT dataset
    print("\n" + "="*80)
    print("ANALYZING: AfterChatGPT Dataset")
    print("="*80)
    
    result2 = calculate_weighted_metrics(
        sample_file='Sample_Analysis/emotion_sample_comparison.csv',
        evaluation_excel='Evaluation_ AfterChatGPT_Labeling.xlsx',
        labeled_full_file='Labeling/AfterChatGPT.labeled.csv',
        dataset_name='AfterChatGPT (Post-Launch)'
    )
    results.append(result2)
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    
    report = generate_report(results)
    
    with open('Sample_Analysis/weighted_accuracy_report.txt', 'w') as f:
        f.write(report)
    
    print("\n✅ Report saved as 'Sample_Analysis/weighted_accuracy_report.txt'")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for result in results:
        print(f"\n{result['dataset']}:")
        print(f"  Unweighted Accuracy: {result['unweighted_accuracy']*100:.1f}%")
        print(f"  Weighted Accuracy:   {result['weighted_accuracy']*100:.2f}%")
        print(f"  Difference:          {(result['weighted_accuracy'] - result['unweighted_accuracy'])*100:.1f}pp")

if __name__ == "__main__":
    main()

