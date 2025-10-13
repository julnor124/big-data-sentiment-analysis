# Sample Analysis Folder
## Labeling Review and Comparison Files

This folder contains scripts and CSV files for reviewing emotion labeling accuracy by comparing original tweets with their cleaned versions.

## 📁 Files in this folder:

### **Scripts:**
- `create_emotion_sample_comparison.py` - Creates 400-tweet sample from AfterChatGPT.labeled.csv
- `create_labeled_tweets_ai_comparison.py` - Creates 400-tweet sample from clean_tweets_ai.labeled.csv

### **Sample CSV Files:**

#### **AfterChatGPT Dataset Samples:**
- `emotion_sample_comparison.csv` - Full sample (400 tweets) from AfterChatGPT.labeled.csv
- `high_quality_emotion_sample.csv` - High-quality subset (~140 tweets) with better similarity scores

#### **Tweets AI Dataset Samples:**
- `labeled_tweets_ai_comparison.csv` - Full sample (400 tweets) from clean_tweets_ai.labeled.csv  
- `high_quality_labeled_tweets_ai_sample.csv` - High-quality subset (~140 tweets) with better similarity scores

## 🔍 How to Review Labeling:

### **1. Open the CSV files in Excel or any spreadsheet application**

### **2. Key columns to review:**
- **`Emotion_Label`** - The detected emotion (neutral, joy, anger, etc.)
- **`Emotion_Probability`** - Confidence score (0-1, higher = more confident)
- **`Original_Tweet`** - Raw tweet before cleaning
- **`Cleaned_Tweet`** - Processed tweet after cleaning
- **`Similarity_Score`** - How well original and cleaned tweets match

### **3. Review process:**
1. **Read the original tweet** to understand the context
2. **Check the emotion label** - does it seem correct?
3. **Look at the probability** - is the model confident?
4. **Compare with cleaned version** - what was removed/changed?
5. **Note any mislabelings** for potential model improvement

### **4. Focus on:**
- **High-confidence labels** (probability > 0.8) that seem wrong
- **Low-confidence labels** (probability < 0.5) that might be uncertain
- **Edge cases** where emotion could be ambiguous
- **Context-dependent emotions** that might be misclassified

## 📊 Sample Statistics:

### **AfterChatGPT Dataset:**
- **Total samples**: 400 tweets
- **Emotions**: 7 different emotions (neutral, joy, anger, sadness, fear, surprise, disgust)
- **Success rate**: ~98% (390+ matches found)
- **Average similarity**: ~43% between original and cleaned

### **Tweets AI Dataset:**
- **Total samples**: 400 tweets  
- **Emotions**: 7 different emotions
- **Success rate**: ~97% (390+ matches found)
- **Average similarity**: ~42% between original and cleaned

## 🎯 Review Guidelines:

### **Emotion Categories:**
- **Neutral**: Factual, informative, no strong emotion
- **Joy**: Happy, excited, positive sentiment
- **Anger**: Frustrated, annoyed, negative sentiment
- **Sadness**: Disappointed, melancholic, negative sentiment
- **Fear**: Worried, anxious, concerned
- **Surprise**: Unexpected, amazed, shocked
- **Disgust**: Repulsed, offended, negative sentiment

### **Common Issues to Look For:**
1. **Sarcasm misclassified** as joy instead of anger
2. **Concern misclassified** as fear instead of sadness
3. **Excitement misclassified** as joy instead of surprise
4. **Neutral content** over-confidently labeled as emotional
5. **Context-dependent emotions** that depend on background knowledge

## 📝 Notes:
- Use the high-quality samples for detailed review (better similarity scores)
- Focus on tweets where you can clearly see the original context
- Consider that some emotions might be subtle or context-dependent
- The similarity score helps identify if the original and cleaned tweets are actually the same content

## 🔄 To regenerate samples:
```bash
cd /Users/julianordqvist/Documents/GitHub/big-data-sentiment-analysis/Sample_Analysis
python3 create_emotion_sample_comparison.py
python3 create_labeled_tweets_ai_comparison.py
```
