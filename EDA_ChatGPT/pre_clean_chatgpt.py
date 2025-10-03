#!/usr/bin/env python3
"""
Script to clean ChatGPT.csv by keeping only 'Date' and 'Tweet' columns.
"""

import pandas as pd
import os

def pre_clean_chatgpt_csv():
    """Clean ChatGPT.csv by keeping only Date and Tweet columns."""
    
    print("🧹 Cleaning ChatGPT.csv...")
    
    chatgpt_file = 'ChatGPT.csv'
    if os.path.exists(chatgpt_file):
        print(f"📄 Processing {chatgpt_file}...")
        try:
            # Read the file
            df_chatgpt = pd.read_csv(chatgpt_file)
            print(f"   Original shape: {df_chatgpt.shape}")
            print(f"   Original columns: {list(df_chatgpt.columns)}")
            
            # Check if required columns exist
            required_cols = ['Date', 'Tweet']
            missing_cols = [col for col in required_cols if col not in df_chatgpt.columns]
            
            if missing_cols:
                print(f"   ⚠️  Missing columns: {missing_cols}")
                print(f"   Available columns: {list(df_chatgpt.columns)}")
            else:
                # Keep only required columns
                df_chatgpt_clean = df_chatgpt[required_cols].copy()
                print(f"   Cleaned shape: {df_chatgpt_clean.shape}")
                
                # Save cleaned file
                output_file = 'ChatGPT_clean.csv'
                df_chatgpt_clean.to_csv(output_file, index=False)
                print(f"   ✅ Saved cleaned file: {output_file}")
                
                # Show first few rows
                print(f"   First few rows:")
                print(df_chatgpt_clean.head())
                
        except Exception as e:
            print(f"   ❌ Error processing {chatgpt_file}: {e}")
    else:
        print(f"   ⚠️  File {chatgpt_file} not found")
    
    print("\n🎉 ChatGPT.csv cleaning completed!")

if __name__ == "__main__":
    pre_clean_chatgpt_csv()
