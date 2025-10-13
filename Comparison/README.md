# Comparison Folder
## Emotional Analysis Iteration Comparison

This folder contains a comprehensive comparison between the two iterations of emotional timeline analysis to validate findings and assess robustness.

## 📁 Files in this folder:

### **Analysis Scripts:**
- `compare_emotional_analyses.py` - Comparison script that analyzes both iterations

### **Comparison Outputs:**
- `emotional_analysis_comparison_report.txt` - Detailed comparison report with metrics
- `emotional_analysis_comparison.png` - Visual comparison with 12 subplots

## 📊 What's Being Compared:

### **Iteration 1:**
- **Pre-ChatGPT**: clean_tweets_ai.labeled.csv (490,118 tweets)
- **2022**: AfterChatGPT.labeled.csv filtered for 2022 (80,499 tweets)
- **2023+**: AfterChatGPT.labeled.csv filtered for 2023+ (409,929 tweets)
- **Total**: 980,546 tweets

### **Iteration 2:**
- **Pre-ChatGPT**: tweets_ai_downsampled.labeled.csv (494,227 tweets)
- **2022**: postlaunch.labeled.csv filtered for 2022 (81,374 tweets)
- **2023+**: postlaunch.labeled.csv filtered for 2023+ (418,291 tweets)
- **Total**: 993,892 tweets

## 🎯 Key Metrics:

### **Overall Agreement:**
- **Correlation**: 0.9751 (very strong positive relationship)
- **Mean Absolute Error (MAE)**: 4.88 percentage points
- **RMSE**: 7.18 percentage points
- **Average Agreement**: 95.1%

### **Agreement by Period:**
- Pre-ChatGPT (2017-2021): 94.1%
- Right after ChatGPT (2022): 95.9%
- Established ChatGPT (2023+): 95.3%

### **Agreement by Emotion:**
- **Highest Agreement** (>98%): anger, disgust, fear, sadness
- **Good Agreement** (>90%): joy, surprise
- **Lower Agreement**: neutral (84.7%) - but still good!

## 📈 Major Findings:

### **1. Highly Consistent Results:**
- 97.5% correlation shows both analyses tell the same story
- Key emotion trends are validated across both iterations

### **2. Main Discrepancies:**

#### **Neutral Emotion:**
- Largest difference (15.3pp average across periods)
- Iter1: Lower neutral percentages
- Iter2: Higher neutral percentages
- Both show same trend: neutral decreases after ChatGPT

#### **Joy & Surprise:**
- Iter1 shows higher joy/surprise in pre-ChatGPT period
- Iter2 shows lower joy/surprise in pre-ChatGPT period
- Both show same trend: joy/surprise increase after ChatGPT

### **3. Validated Key Trends:**
Both iterations confirm:
- ✅ Neutral sentiment **decreases** after ChatGPT launch
- ✅ Joy and surprise **increase** after ChatGPT launch
- ✅ Fear and sadness show **moderate increases**
- ✅ Emotional discourse becomes more varied post-ChatGPT

## 📊 Visualization Breakdown:

The comparison visualization includes 12 subplots:

1. **Subplots 1-3**: Side-by-side emotion distributions for each period
2. **Subplot 4**: Heatmap of differences (Iter2 - Iter1)
3. **Subplot 5**: Correlation scatter plot with regression line
4. **Subplot 6**: Agreement by emotion (bar chart)
5. **Subplot 7**: Dataset size comparison
6. **Subplot 8**: Empty (placeholder)
7. **Subplot 9**: Empty (placeholder)
8. **Subplots 10-12**: Emotion difference bars for each period

## 🔍 What This Means:

### **Validation:**
✅ The main findings are **robust** - they hold true regardless of dataset choice
✅ The emotional shift after ChatGPT's launch is **confirmed** in both analyses
✅ 95%+ agreement shows the conclusions are **reliable**

### **Dataset Impact:**
- Iteration 1 used full datasets
- Iteration 2 used downsampled pre-ChatGPT + combined post-launch
- Different data sources yield highly consistent results

### **Confidence:**
- Can confidently report the key emotional trends
- The 15.3pp neutral difference suggests some sampling bias but doesn't change overall conclusions
- Joy/surprise differences are smaller and directionally consistent

## 📝 How to Interpret:

### **Agreement Scores:**
- **>95%**: Excellent agreement - very reliable
- **90-95%**: Good agreement - reliable
- **80-90%**: Moderate agreement - generally reliable
- **<80%**: Lower agreement - needs context

### **Color Coding in Visualizations:**
- **Green**: High agreement (>90%)
- **Orange**: Moderate agreement (80-90%)
- **Red**: Lower agreement (<80%)

## 🚀 Usage:

### **Run the comparison:**
```bash
cd /Users/julianordqvist/Documents/GitHub/big-data-sentiment-analysis
python3 Comparison/compare_emotional_analyses.py
```

### **Review the results:**
1. Check `emotional_analysis_comparison_report.txt` for detailed metrics
2. Open `emotional_analysis_comparison.png` for visual comparison
3. Focus on high-agreement emotions for strongest conclusions

## 💡 Recommendations:

1. **For Research/Reporting:**
   - Use findings from either iteration with confidence
   - Focus on emotions with >90% agreement for strongest claims
   - Mention the 95.1% overall agreement as validation

2. **For Further Analysis:**
   - Investigate why neutral shows the largest difference
   - Consider the impact of downsampling on emotion detection
   - Explore if pre-processing differences affect results

3. **For Presentations:**
   - Highlight the 97.5% correlation as strong validation
   - Show that key trends (neutral decrease, joy/surprise increase) are robust
   - Use the comparison visualization to demonstrate reliability

## 🎓 Statistical Interpretation:

### **Correlation of 0.9751:**
- Near-perfect positive relationship
- Both iterations rank emotions almost identically
- Validates the emotion labeling model's consistency

### **MAE of 4.88pp:**
- On average, emotion percentages differ by ~5 points
- This is very good for social media emotion analysis
- Shows minor variations don't change overall patterns

### **95.1% Agreement:**
- Extremely high consistency across 21 comparisons (7 emotions × 3 periods)
- Only 4.9% average difference
- Strongly supports the validity of both analyses

---

**Generated**: October 2025  
**Purpose**: Validate emotional timeline analysis findings  
**Result**: 95.1% agreement confirms key trends are robust

