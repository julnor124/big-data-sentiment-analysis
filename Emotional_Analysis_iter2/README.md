# Emotional Analysis Iteration 2
## Sentiment and Emotion Analysis - Second Iteration

This folder contains emotional analysis for the iteration 2 datasets, comparing emotions across different time periods related to AI and ChatGPT discourse.

## 📁 Files in this folder:

### **Input Datasets (from Sampling_iter2):**
- `../Sampling_iter2/postlaunch.labeled.csv` - Combined ChatGPT and GenerativeAI post-launch dataset with emotion labels (~500K tweets)
- `../Sampling_iter2/tweets_ai_downsampled.labeled.csv` - Downsampled AI tweets dataset with emotion labels (~494K tweets)

### **Analysis Scripts:**
- `emotion_timeline_analysis.py` - Script for analyzing emotion trends over time periods

### **Analysis Outputs:**
- `emotion_timeline_analysis_report.txt` - Detailed report of emotion analysis findings
- `emotion_timeline_analysis.png` - Comprehensive visualizations of emotion trends

## 📊 Dataset Information:

### **postlaunch.labeled.csv:**
- **Size**: ~500K tweets
- **Emotions**: 7 categories (neutral, joy, anger, sadness, fear, surprise, disgust)
- **Source**: Combined from chatgpt_cleaned_it2.csv + generativeaiopinion_pre_clean.csv
- **Date Range**: 2022-2023+ (post-ChatGPT launch)
- **Columns**: Date, Tweet, emotion_label, emotion_prob

### **tweets_ai_downsampled.labeled.csv:**
- **Size**: ~494K tweets
- **Emotions**: 7 categories (neutral, joy, anger, sadness, fear, surprise, disgust)
- **Source**: Downsampled AI-related tweets (pre-ChatGPT)
- **Date Range**: 2017-2021
- **Columns**: Date, Tweet, emotion_label, emotion_prob

## 🎭 Emotion Categories:

1. **Neutral** - Factual, informative, no strong emotion
2. **Joy** - Happy, excited, positive sentiment
3. **Anger** - Frustrated, annoyed, negative sentiment
4. **Sadness** - Disappointed, melancholic, negative sentiment
5. **Fear** - Worried, anxious, concerned
6. **Surprise** - Unexpected, amazed, shocked
7. **Disgust** - Repulsed, offended, negative sentiment

## 📈 Analysis Focus:

### **Time Period Comparison:**
1. **Pre-ChatGPT Era (2017-2021)**: 
   - Source: tweets_ai_downsampled.labeled.csv
   - AI discourse before ChatGPT launch
   
2. **Right after ChatGPT (2022)**:
   - Source: postlaunch.labeled.csv (filtered for 2022)
   - Immediate reactions to ChatGPT launch
   
3. **Established ChatGPT Era (2023+)**:
   - Source: postlaunch.labeled.csv (filtered for 2023+)
   - ChatGPT as an established technology

### **Key Analysis Areas:**
- Emotion distribution changes across time periods
- Statistical significance testing (Chi-square)
- Visualization of emotion trends
- Identification of key emotional shifts
- Comparison of pre/post ChatGPT sentiment

## 🔍 Key Differences from Iteration 1:

### **Dataset Changes:**
- **Iteration 1**: Used AfterChatGPT.labeled.csv + clean_tweets_ai.labeled.csv
- **Iteration 2**: Uses postlaunch.labeled.csv + tweets_ai_downsampled.labeled.csv

### **Benefits of Iteration 2:**
- Downsampled pre-ChatGPT data for better balance
- Combined post-launch datasets for comprehensive coverage
- Same emotion labeling methodology for consistency
- Focus on the most relevant time periods

## 🚀 Usage:

### **Run the analysis:**
```bash
cd /Users/julianordqvist/Documents/GitHub/big-data-sentiment-analysis
python3 Emotional_Analysis_iter2/emotion_timeline_analysis.py
```

### **Expected outputs:**
1. Console output with emotion distributions and statistics
2. `emotion_timeline_analysis.png` with comprehensive visualizations
3. `emotion_timeline_analysis_report.txt` with detailed findings

## 📊 Visualization Features:

The analysis generates a comprehensive visualization with 9 subplots:

1. **Emotion Distribution (Counts)**: Bar chart showing tweet counts per emotion by period
2. **Emotion Distribution (Percentages)**: Bar chart showing percentage distribution
3. **Stacked Distribution**: Stacked bar showing relative emotion proportions
4. **Heatmap**: Color-coded emotion intensity across periods
5. **Trend Lines**: Line chart showing emotion evolution over time
6. **Pie Charts (3)**: One for each time period showing emotion breakdown

## 📝 Statistical Analysis:

### **Chi-Square Test:**
- Tests if emotion distributions differ significantly across periods
- Provides p-value for statistical significance
- Identifies which emotions changed most dramatically

### **Percentage Point Changes:**
- Calculates exact changes in emotion percentages
- Ranks emotions by magnitude of change
- Shows direction of change (increase/decrease)

## 🎯 Research Questions Answered:

1. **How did emotions change after ChatGPT's launch?**
   - Compare pre-2022 vs 2022 vs 2023+ distributions
   
2. **Which emotions increased/decreased the most?**
   - Statistical analysis of emotion shifts
   
3. **Are the changes statistically significant?**
   - Chi-square test validates findings
   
4. **What does this tell us about public perception?**
   - Insights into AI discourse evolution

## 📋 Key Metrics Reported:

- Total tweets per period
- Emotion counts and percentages per period
- Chi-square statistic and p-value
- Top 5 emotion changes (by magnitude)
- Detailed emotion breakdowns

## 🔄 Comparison with Iteration 1:

You can compare results from Emotional_Analysis_iter2 with Emotional_Analysis to see:
- How dataset choices affect conclusions
- Consistency of emotion trends across different samples
- Impact of downsampling on analysis quality
- Validation of key findings

## 📌 Notes:

- All emotion labels are machine-generated
- Confidence scores (emotion_prob) indicate model certainty
- High confidence (>0.8) labels are more reliable
- Analysis uses the same emotion classification model across all periods
- Date ranges are automatically filtered from the datasets

## 🔬 Next Steps:

After running this analysis, you can:
1. Compare with Iteration 1 results for validation
2. Explore specific emotion categories in detail
3. Correlate findings with real-world events
4. Use insights for further research or reporting
5. Filter by confidence scores for refined analysis

---

**Generated**: October 2025  
**Project**: Big Data Sentiment Analysis  
**Iteration**: 2

