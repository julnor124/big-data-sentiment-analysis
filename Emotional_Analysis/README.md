# Emotional Analysis Folder
## Sentiment and Emotion Analysis Files

This folder contains all files related to emotional analysis, sentiment detection, and emotion labeling for the big data sentiment analysis project.

## 📁 Files in this folder:

### **Labeled Datasets:**
- `AfterChatGPT.labeled.csv` - Combined ChatGPT and GenerativeAI dataset with emotion labels
- `clean_tweets_ai.labeled.csv` - AI tweets dataset with emotion labels

### **Analysis Scripts:**
- `emotion_timeline_analysis.py` - Script for analyzing emotion trends over time

### **Analysis Outputs:**
- `emotion_timeline_analysis_report.txt` - Detailed report of emotion analysis
- `emotion_timeline_analysis.png` - Visualization of emotion trends over time

## 📊 Dataset Information:

### **AfterChatGPT.labeled.csv:**
- **Size**: ~490K tweets
- **Emotions**: 7 categories (neutral, joy, anger, sadness, fear, surprise, disgust)
- **Sources**: ChatGPT and GenerativeAI tweets
- **Columns**: Date, Tweet, Source, emotion_label, emotion_prob

### **clean_tweets_ai.labeled.csv:**
- **Size**: ~490K tweets
- **Emotions**: 7 categories (neutral, joy, anger, sadness, fear, surprise, disgust)
- **Source**: AI-related tweets
- **Columns**: Date, Tweet, emotion_label, emotion_prob

## 🎭 Emotion Categories:

1. **Neutral** - Factual, informative, no strong emotion
2. **Joy** - Happy, excited, positive sentiment
3. **Anger** - Frustrated, annoyed, negative sentiment
4. **Sadness** - Disappointed, melancholic, negative sentiment
5. **Fear** - Worried, anxious, concerned
6. **Surprise** - Unexpected, amazed, shocked
7. **Disgust** - Repulsed, offended, negative sentiment

## 📈 Analysis Capabilities:

### **Timeline Analysis:**
- Emotion trends over time
- Seasonal patterns
- Event-driven emotion spikes
- Long-term sentiment evolution

### **Emotion Distribution:**
- Overall emotion frequency
- Source-specific emotion patterns
- Confidence score analysis
- Emotion correlation analysis

## 🔍 Usage Examples:

### **Load labeled dataset:**
```python
import pandas as pd

# Load AfterChatGPT labeled dataset
df = pd.read_csv('AfterChatGPT.labeled.csv')

# Check emotion distribution
print(df['emotion_label'].value_counts())

# Filter by emotion
joy_tweets = df[df['emotion_label'] == 'joy']
```

### **Analyze emotion confidence:**
```python
# High confidence emotions
high_conf = df[df['emotion_prob'] > 0.8]

# Low confidence emotions (might need review)
low_conf = df[df['emotion_prob'] < 0.5]
```

### **Time-based analysis:**
```python
# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Group by month and emotion
monthly_emotions = df.groupby([df['Date'].dt.to_period('M'), 'emotion_label']).size()
```

## 📊 Key Statistics:

### **AfterChatGPT Dataset:**
- **Total tweets**: 490,457
- **ChatGPT tweets**: ~470K (95.7%)
- **GenerativeAI tweets**: ~21K (4.3%)
- **Date range**: 2022-11-30 to 2024-11-15

### **Tweets AI Dataset:**
- **Total tweets**: 490,118
- **Date range**: 2017-01-31 to 2021-01-27
- **Language**: Primarily English (96.7%)

## 🎯 Research Applications:

1. **Sentiment Analysis**: Understanding public opinion on AI topics
2. **Emotion Trends**: Tracking how emotions change over time
3. **Event Analysis**: Correlating emotions with specific events
4. **Source Comparison**: Comparing emotions between different AI topics
5. **Confidence Analysis**: Identifying uncertain emotion classifications

## 📝 Notes:

- All emotion labels are generated using machine learning models
- Confidence scores (emotion_prob) indicate model certainty
- High confidence scores (>0.8) are generally more reliable
- Low confidence scores (<0.5) may need manual review
- The datasets are ready for further analysis and visualization

## 🔄 To regenerate analysis:
```bash
cd /Users/julianordqvist/Documents/GitHub/big-data-sentiment-analysis/Emotional_Analysis
python3 emotion_timeline_analysis.py
```
