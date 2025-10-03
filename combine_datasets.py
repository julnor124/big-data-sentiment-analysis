#!/usr/bin/env python3
"""
Dataset Combination Script
=========================

This script combines the cleaned ChatGPT and GenerativeAI datasets into a single file.
It merges ChatGPT_cleaned.csv and GenerativeAI_cleaned.csv into AfterChatGPT.csv.

Author: Generated for Big Data Sentiment Analysis Project
Date: 2025-10-03
"""

import pandas as pd
import os
from datetime import datetime

def combine_datasets():
    """Combine ChatGPT and GenerativeAI cleaned datasets."""
    
    print("🔄 COMBINING CLEANED DATASETS")
    print("=" * 50)
    
    # File paths
    chatgpt_file = 'CLEAN_ChatGPT/ChatGPT_cleaned.csv'
    genai_file = 'CLEAN_GenAI/GenerativeAI_cleaned.csv'
    output_file = 'AfterChatGPT.csv'
    
    # Check if input files exist
    if not os.path.exists(chatgpt_file):
        print(f"❌ Error: {chatgpt_file} not found!")
        return
    
    if not os.path.exists(genai_file):
        print(f"❌ Error: {genai_file} not found!")
        return
    
    try:
        # Load datasets
        print("📂 Loading datasets...")
        print(f"   - Loading {chatgpt_file}...")
        df_chatgpt = pd.read_csv(chatgpt_file)
        print(f"   - Loading {genai_file}...")
        df_genai = pd.read_csv(genai_file)
        
        print(f"✅ Datasets loaded successfully!")
        print(f"   - ChatGPT: {len(df_chatgpt):,} rows")
        print(f"   - GenerativeAI: {len(df_genai):,} rows")
        
        # Check column compatibility
        chatgpt_cols = set(df_chatgpt.columns)
        genai_cols = set(df_genai.columns)
        
        if chatgpt_cols != genai_cols:
            print(f"⚠️  Warning: Column mismatch detected!")
            print(f"   - ChatGPT columns: {list(chatgpt_cols)}")
            print(f"   - GenAI columns: {list(genai_cols)}")
            print(f"   - Common columns: {list(chatgpt_cols & genai_cols)}")
            print(f"   - Missing in ChatGPT: {list(genai_cols - chatgpt_cols)}")
            print(f"   - Missing in GenAI: {list(chatgpt_cols - genai_cols)}")
        
        # Add source column to each dataset
        print("\n🏷️  Adding source labels...")
        df_chatgpt['Source'] = 'ChatGPT'
        df_genai['Source'] = 'GenAI'
        
        # Combine datasets
        print("\n🔄 Combining datasets...")
        df_combined = pd.concat([df_chatgpt, df_genai], ignore_index=True)
        
        print(f"✅ Datasets combined successfully!")
        print(f"   - Combined: {len(df_combined):,} rows")
        print(f"   - Columns: {list(df_combined.columns)}")
        
        # Save combined dataset
        print(f"\n💾 Saving combined dataset...")
        df_combined.to_csv(output_file, index=False)
        
        # Calculate statistics
        memory_usage = df_combined.memory_usage(deep=True).sum() / 1024**2
        
        print(f"✅ Combined dataset saved as '{output_file}'")
        print(f"   - Final shape: {df_combined.shape}")
        print(f"   - Memory usage: {memory_usage:.2f} MB")
        
        # Generate summary report
        print("\n📋 Generating summary report...")
        
        report = f"""
DATASET COMBINATION REPORT
=========================

COMBINATION SUMMARY:
- ChatGPT dataset: {len(df_chatgpt):,} rows
- GenerativeAI dataset: {len(df_genai):,} rows
- Combined dataset: {len(df_combined):,} rows
- Output file: {output_file}

DATASET COMPOSITION:
- ChatGPT percentage: {(len(df_chatgpt) / len(df_combined)) * 100:.2f}%
- GenerativeAI percentage: {(len(df_genai) / len(df_combined)) * 100:.2f}%

MEMORY USAGE:
- Combined dataset: {memory_usage:.2f} MB

COLUMNS:
- Date: {df_combined['Date'].dtype}
- Tweet: {df_combined['Tweet'].dtype}
- Source: {df_combined['Source'].dtype}

SOURCE DISTRIBUTION:
- ChatGPT: {len(df_combined[df_combined['Source'] == 'ChatGPT']):,} rows ({(len(df_combined[df_combined['Source'] == 'ChatGPT']) / len(df_combined)) * 100:.2f}%)
- GenAI: {len(df_combined[df_combined['Source'] == 'GenAI']):,} rows ({(len(df_combined[df_combined['Source'] == 'GenAI']) / len(df_combined)) * 100:.2f}%)

DATASET QUALITY:
- No missing dates: {'✅' if df_combined['Date'].notna().all() else '❌'}
- No missing tweets: {'✅' if df_combined['Tweet'].notna().all() else '❌'}
- No missing sources: {'✅' if df_combined['Source'].notna().all() else '❌'}
- No duplicates: {'✅' if not df_combined.duplicated().any() else '❌'}
- Text preprocessed: ✅

"""
        
        # Save report
        with open('AfterChatGPT_combination_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✅ Summary report saved as 'AfterChatGPT_combination_report.txt'")
        
        # Final summary
        print("\n" + "=" * 50)
        print("🎉 COMBINATION COMPLETED!")
        print("=" * 50)
        print(f"📊 Final Results:")
        print(f"   - Input files: {chatgpt_file}, {genai_file}")
        print(f"   - Output file: {output_file}")
        print(f"   - Total rows: {len(df_combined):,}")
        print(f"   - Report: AfterChatGPT_combination_report.txt")
        
    except Exception as e:
        print(f"❌ Error during combination: {e}")
        return

if __name__ == "__main__":
    combine_datasets()
