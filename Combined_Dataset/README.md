# Combined Dataset Folder
## Main Dataset and Source Files

This folder contains the main combined dataset and all related source files for the big data sentiment analysis project.

## 📁 Files in this folder:

### **Main Combined Dataset:**
- `AfterChatGPT.csv` - Final combined dataset (59MB, 490K+ tweets)
- `AfterChatGPT_combination_report.txt` - Detailed combination report

### **Source Datasets:**
- `generativeaiopinion.csv` - Original GenerativeAI dataset (6.5MB, 22K+ tweets)
- `tweets_ai.csv` - Original AI tweets dataset (380MB, 893K+ tweets)

### **Scripts:**
- `combine_datasets.py` - Script to combine ChatGPT and GenerativeAI datasets

## 📊 Dataset Information:

### **AfterChatGPT.csv (Main Combined Dataset):**
- **Size**: 490,457 tweets
- **Sources**: 
  - ChatGPT: 469,169 tweets (95.7%)
  - GenerativeAI: 21,288 tweets (4.3%)
- **Columns**: Date, Tweet, Source
- **Date Range**: 2022-11-30 to 2024-11-15
- **Purpose**: Main dataset for sentiment analysis

### **generativeaiopinion.csv (Source):**
- **Size**: 22,066 tweets
- **Topic**: Generative AI opinions and discussions
- **Date Range**: 2024-08-01 to 2024-11-21
- **Language**: Primarily English
- **Source**: Social media posts about Generative AI

### **tweets_ai.csv (Source):**
- **Size**: 893,076 tweets
- **Topic**: Artificial Intelligence discussions
- **Date Range**: 2017-01-31 to 2021-01-27
- **Language**: Primarily English (96.7%)
- **Source**: AI-related social media posts

## 🔄 Dataset Creation Process:

### **Step 1: Data Collection**
- Collected tweets from multiple sources
- Focused on AI-related topics
- Time period: 2017-2024

### **Step 2: Data Cleaning**
- Removed duplicates
- Handled missing values
- Standardized date formats
- Text preprocessing

### **Step 3: Dataset Combination**
- Combined ChatGPT and GenerativeAI datasets
- Added source identification
- Maintained data quality
- Generated combination report

## 📈 Dataset Statistics:

### **Combined Dataset (AfterChatGPT.csv):**
- **Total tweets**: 490,457
- **ChatGPT percentage**: 95.7%
- **GenerativeAI percentage**: 4.3%
- **Memory usage**: ~60MB
- **No missing values**: ✅
- **No duplicates**: ✅
- **Text preprocessed**: ✅

### **Source Dataset Comparison:**
| Dataset | Size | Percentage | Topic Focus |
|---------|------|------------|-------------|
| ChatGPT | 469K tweets | 95.7% | ChatGPT discussions |
| GenerativeAI | 21K tweets | 4.3% | Generative AI opinions |
| **Total** | **490K tweets** | **100%** | **AI sentiment analysis** |

## 🎯 Usage Examples:

### **Load combined dataset:**
```python
import pandas as pd

# Load main combined dataset
df = pd.read_csv('AfterChatGPT.csv')

# Check dataset info
print(f"Total tweets: {len(df):,}")
print(f"Sources: {df['Source'].value_counts()}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
```

### **Filter by source:**
```python
# ChatGPT tweets only
chatgpt_tweets = df[df['Source'] == 'ChatGPT']

# GenerativeAI tweets only
genai_tweets = df[df['Source'] == 'GenerativeAI']
```

### **Time-based analysis:**
```python
# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Group by month
monthly_tweets = df.groupby(df['Date'].dt.to_period('M')).size()
```

## 🔄 To recreate the combined dataset:
```bash
cd /Users/julianordqvist/Documents/GitHub/big-data-sentiment-analysis/Combined_Dataset
python3 combine_datasets.py
```

## 📝 Notes:

- The combined dataset is ready for sentiment analysis
- Source identification allows for comparative analysis
- All data has been cleaned and preprocessed
- The dataset spans multiple years for trend analysis
- Suitable for machine learning and NLP tasks

## 🎯 Research Applications:

1. **Sentiment Analysis**: Analyze public opinion on AI topics
2. **Comparative Studies**: Compare ChatGPT vs GenerativeAI discussions
3. **Trend Analysis**: Track sentiment changes over time
4. **Topic Modeling**: Identify key themes and discussions
5. **Machine Learning**: Train models for sentiment classification
6. **Social Media Research**: Study AI discourse patterns
