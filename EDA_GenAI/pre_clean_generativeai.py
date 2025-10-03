#!/usr/bin/env python3
"""
Script to clean generativeaiopinion_with_dates.csv by keeping only 'date' and 'full_text' columns.
"""

import pandas as pd
import os

def pre_clean_generativeai_csv():
    """Clean generativeaiopinion_pre_clean.csv by keeping only date and full_text columns."""
    
    print("🧹 Cleaning generativeaiopinion_pre_clean.csv...")
    
    genai_file = 'generativeaiopinion_pre_clean.csv'
    if os.path.exists(genai_file):
        print(f"📄 Processing {genai_file}...")
        try:
            # Read the file
            df_genai = pd.read_csv(genai_file)
            print(f"   Original shape: {df_genai.shape}")
            print(f"   Original columns: {list(df_genai.columns)}")
            
            # Check if required columns exist
            required_cols = ['date', 'full_text']
            missing_cols = [col for col in required_cols if col not in df_genai.columns]
            
            if missing_cols:
                print(f"   ⚠️  Missing columns: {missing_cols}")
                print(f"   Available columns: {list(df_genai.columns)}")
            else:
                # Keep only required columns and rename to match ChatGPT format
                df_genai_clean = df_genai[required_cols].copy()
                
                # Rename columns to match ChatGPT format (Date, Tweet)
                df_genai_clean = df_genai_clean.rename(columns={
                    'date': 'Date',
                    'full_text': 'Tweet'
                })
                
                print(f"   Cleaned shape: {df_genai_clean.shape}")
                
                # Save cleaned file
                output_file = 'generativeaiopinion_clean.csv'
                df_genai_clean.to_csv(output_file, index=False)
                print(f"   ✅ Saved cleaned file: {output_file}")
                
                # Show first few rows
                print(f"   First few rows:")
                print(df_genai_clean.head())
                
        except Exception as e:
            print(f"   ❌ Error processing {genai_file}: {e}")
    else:
        print(f"   ⚠️  File {genai_file} not found")
    
    print("\n🎉 generativeaiopinion_pre_clean.csv cleaning completed!")

if __name__ == "__main__":
    pre_clean_generativeai_csv()
