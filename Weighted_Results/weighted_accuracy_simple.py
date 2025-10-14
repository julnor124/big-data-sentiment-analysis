#!/usr/bin/env python3
"""
Simple Weighted Accuracy Calculator
====================================
Calculate weighted accuracy when you have error counts per emotion.
"""

import pandas as pd
import json

def calculate_weighted_accuracy():
    """
    Calculate weighted accuracy metrics with manual input
    """
    
    print("="*80)
    print("WEIGHTED ACCURACY CALCULATOR - MANUAL INPUT VERSION")
    print("="*80)
    print()
    
    # Get actual distributions from labeled datasets
    print("Loading actual emotion distributions from datasets...")
    
    tweets_ai_df = pd.read_csv('Labeling/clean_tweets_ai.labeled.csv')
    afterchatgpt_df = pd.read_csv('Labeling/AfterChatGPT.labeled.csv')
    
    tweets_ai_dist = tweets_ai_df['emotion_label'].value_counts()
    tweets_ai_prop = (tweets_ai_dist / len(tweets_ai_df)).to_dict()
    
    afterchatgpt_dist = afterchatgpt_df['emotion_label'].value_counts()  
    afterchatgpt_prop = (afterchatgpt_dist / len(afterchatgpt_df)).to_dict()
    
    print("\n" + "="*80)
    print("TWEETS AI DATASET (Pre-ChatGPT)")
    print("="*80)
    print(f"\nActual distribution in full dataset ({len(tweets_ai_df):,} tweets):")
    for emotion in sorted(tweets_ai_prop.keys()):
        count = tweets_ai_dist[emotion]
        prop = tweets_ai_prop[emotion]
        print(f"  {emotion}: {count:,} ({prop*100:.1f}%)")
    
    print("\n" + "-"*80)
    print("MANUAL INPUT - Error counts per emotion from your evaluation:")
    print("-"*80)
    print("\nFör varje känsla, ange antalet FEL (errors) ni hittade i samplet (57 per känsla):")
    print()
    
    tweets_ai_errors = {}
    emotions_list = sorted(tweets_ai_prop.keys())
    
    for emotion in emotions_list:
        while True:
            try:
                errors = int(input(f"  {emotion}: "))
                if 0 <= errors <= 57:
                    tweets_ai_errors[emotion] = errors
                    break
                else:
                    print(f"    Felaktigt värde. Ange mellan 0 och 57.")
            except ValueError:
                print(f"    Felaktigt värde. Ange ett heltal.")
    
    # Calculate for Tweets AI
    print("\n" + "="*80)
    print("CALCULATION - TWEETS AI")
    print("="*80)
    
    total_sample = 57 * len(emotions_list)  # Balanced sample
    total_errors = sum(tweets_ai_errors.values())
    
    print(f"\nSample: {total_sample} tweets (balanced)")
    print(f"Total errors: {total_errors}")
    print(f"Unweighted error rate: {total_errors/total_sample*100:.1f}%")
    print(f"Unweighted accuracy: {(1-total_errors/total_sample)*100:.1f}%")
    
    # Weighted calculation
    print(f"\nWeighted calculation:")
    weighted_error = 0
    
    for emotion in emotions_list:
        errors = tweets_ai_errors[emotion]
        sample_count = 57
        error_rate = errors / sample_count
        actual_weight = tweets_ai_prop[emotion]
        contribution = error_rate * actual_weight
        weighted_error += contribution
        
        print(f"\n  {emotion}:")
        print(f"    Errors: {errors}/{sample_count}")
        print(f"    Error rate: {error_rate*100:.1f}%")
        print(f"    Actual proportion: {actual_weight*100:.1f}%")
        print(f"    Weighted contribution: {contribution*100:.2f}%")
    
    weighted_accuracy = 1 - weighted_error
    
    print(f"\n{'='*80}")
    print(f"TWEETS AI RESULTS:")
    print(f"{'='*80}")
    print(f"  Unweighted Accuracy: {(1-total_errors/total_sample)*100:.1f}%")
    print(f"  Weighted Accuracy:   {weighted_accuracy*100:.2f}%")
    print(f"  Difference:          {(weighted_accuracy - (1-total_errors/total_sample))*100:.1f}pp")
    
    # Now AfterChatGPT dataset
    print("\n\n" + "="*80)
    print("AFTERCHATGPT DATASET (Post-Launch)")
    print("="*80)
    print(f"\nActual distribution in full dataset ({len(afterchatgpt_df):,} tweets):")
    for emotion in sorted(afterchatgpt_prop.keys()):
        count = afterchatgpt_dist[emotion]
        prop = afterchatgpt_prop[emotion]
        print(f"  {emotion}: {count:,} ({prop*100:.1f}%)")
    
    print("\n" + "-"*80)
    print("MANUAL INPUT - Error counts per emotion from your evaluation:")
    print("-"*80)
    print("\nFör varje känsla, ange antalet FEL (errors) ni hittade i samplet (57 per känsla):")
    print()
    
    afterchatgpt_errors = {}
    emotions_list2 = sorted(afterchatgpt_prop.keys())
    
    for emotion in emotions_list2:
        while True:
            try:
                errors = int(input(f"  {emotion}: "))
                if 0 <= errors <= 57:
                    afterchatgpt_errors[emotion] = errors
                    break
                else:
                    print(f"    Felaktigt värde. Ange mellan 0 och 57.")
            except ValueError:
                print(f"    Felaktigt värde. Ange ett heltal.")
    
    # Calculate for AfterChatGPT
    print("\n" + "="*80)
    print("CALCULATION - AFTERCHATGPT")
    print("="*80)
    
    total_sample2 = 57 * len(emotions_list2)
    total_errors2 = sum(afterchatgpt_errors.values())
    
    print(f"\nSample: {total_sample2} tweets (balanced)")
    print(f"Total errors: {total_errors2}")
    print(f"Unweighted error rate: {total_errors2/total_sample2*100:.1f}%")
    print(f"Unweighted accuracy: {(1-total_errors2/total_sample2)*100:.1f}%")
    
    # Weighted calculation
    print(f"\nWeighted calculation:")
    weighted_error2 = 0
    
    for emotion in emotions_list2:
        errors = afterchatgpt_errors[emotion]
        sample_count = 57
        error_rate = errors / sample_count
        actual_weight = afterchatgpt_prop[emotion]
        contribution = error_rate * actual_weight
        weighted_error2 += contribution
        
        print(f"\n  {emotion}:")
        print(f"    Errors: {errors}/{sample_count}")
        print(f"    Error rate: {error_rate*100:.1f}%")
        print(f"    Actual proportion: {actual_weight*100:.1f}%")
        print(f"    Weighted contribution: {contribution*100:.2f}%")
    
    weighted_accuracy2 = 1 - weighted_error2
    
    print(f"\n{'='*80}")
    print(f"AFTERCHATGPT RESULTS:")
    print(f"{'='*80}")
    print(f"  Unweighted Accuracy: {(1-total_errors2/total_sample2)*100:.1f}%")
    print(f"  Weighted Accuracy:   {weighted_accuracy2*100:.2f}%")
    print(f"  Difference:          {(weighted_accuracy2 - (1-total_errors2/total_sample2))*100:.1f}pp")
    
    # Final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\nTweets AI (Pre-ChatGPT):")
    print(f"  Unweighted: {(1-total_errors/total_sample)*100:.1f}%")
    print(f"  Weighted:   {weighted_accuracy*100:.2f}%")
    
    print(f"\nAfterChatGPT (Post-Launch):")
    print(f"  Unweighted: {(1-total_errors2/total_sample2)*100:.1f}%")
    print(f"  Weighted:   {weighted_accuracy2*100:.2f}%")
    
    # Save results
    results = {
        'tweets_ai': {
            'total_sample': total_sample,
            'total_errors': total_errors,
            'unweighted_accuracy': (1-total_errors/total_sample),
            'weighted_accuracy': weighted_accuracy,
            'error_counts': tweets_ai_errors,
            'actual_proportions': tweets_ai_prop
        },
        'afterchatgpt': {
            'total_sample': total_sample2,
            'total_errors': total_errors2,
            'unweighted_accuracy': (1-total_errors2/total_sample2),
            'weighted_accuracy': weighted_accuracy2,
            'error_counts': afterchatgpt_errors,
            'actual_proportions': afterchatgpt_prop
        }
    }
    
    with open('Sample_Analysis/weighted_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to 'Sample_Analysis/weighted_accuracy_results.json'")

if __name__ == "__main__":
    calculate_weighted_accuracy()

