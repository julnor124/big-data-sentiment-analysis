#!/usr/bin/env python3
"""
Example data analysis script to test the virtual environment setup.
This script demonstrates basic data analysis capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def main():
    print("Virtual environment is working correctly!")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Create sample data
    data = {
        'text': [
            'I love this product!',
            'This is terrible.',
            'It is okay, nothing special.',
            'Amazing quality and fast delivery!',
            'Waste of money.'
        ],
        'rating': [5, 1, 3, 5, 1]
    }
    
    df = pd.DataFrame(data)
    print("\nSample DataFrame:")
    print(df)
    
    # Basic sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    
    sentiments = []
    for text in df['text']:
        scores = analyzer.polarity_scores(text)
        sentiments.append(scores['compound'])
    
    df['sentiment_score'] = sentiments
    print("\nDataFrame with sentiment scores:")
    print(df)
    
    # Create a simple visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='rating', y='sentiment_score')
    plt.title('Sentiment Scores vs Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Sentiment Score')
    plt.tight_layout()
    plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved as 'sentiment_analysis.png'")
    
    print("\n✅ All packages are working correctly!")
    print("You're ready to start your data analysis project!")

if __name__ == "__main__":
    main()
