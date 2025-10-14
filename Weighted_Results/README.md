# Weighted Accuracy Analysis Results
## Adjusting Model Performance Metrics for Class Imbalance

This folder contains all scripts and results for weighted accuracy analysis, which adjusts traditional accuracy metrics to account for class imbalance in emotion detection.

## 📁 Contents

### **Analysis Scripts:**
1. **`calculate_weighted_accuracy.py`** - Main script for Iteration 1 datasets
2. **`calculate_weighted_accuracy_iter2.py`** - Main script for Iteration 2 datasets  
3. **`weighted_accuracy_simple.py`** - Interactive version (manual input)

### **Reports & Results:**

#### **Iteration 1:**
- **`weighted_accuracy_report.txt`** - Detailed analysis report
- **`weighted_accuracy_output.txt`** - Full console output

#### **Iteration 2:**
- **`weighted_accuracy_report_iter2.txt`** - Detailed analysis report
- **`weighted_accuracy_report_iter2.txt`** - Full console output

---

## 🎯 What is Weighted Accuracy?

**Problem:** Traditional accuracy treats all emotions equally, but datasets are imbalanced:
- Neutral: 52-82% of data
- Disgust: 0.2-1.5% of data

**Solution:** Weight each emotion's accuracy by how often it appears in real data.

**Formula:**
```
Weighted Accuracy = Σ (accuracy_per_emotion × actual_proportion_in_dataset)
```

---

## 📊 Summary of Results

### **ITERATION 1:**

| Dataset | Sample | Evaluation File | Unweighted | Weighted | Improvement |
|---------|--------|----------------|------------|----------|-------------|
| **Tweets AI (Pre-ChatGPT)** | labeled_tweets_ai_comparison.csv | Evaluation_ Tweets_AI_Labeling .xlsx | 55.1% | **80.1%** | +25.0pp |
| **AfterChatGPT (Post-Launch)** | emotion_sample_comparison.csv | Evaluation_ AfterChatGPT_Labeling.xlsx | 67.3% | **74.7%** | +7.3pp |

### **ITERATION 2:**

| Dataset | Sample | Evaluation File | Unweighted | Weighted | Improvement |
|---------|--------|----------------|------------|----------|-------------|
| **Tweets AI Downsampled** | tweets_ai_sampled_400.csv | Evaluation_ tweets_ai.xlsx | 53.0% | **67.9%** | +14.9pp |
| **Postlaunch** | postlaunch_sampled_400.csv | Evaluation_ postlaunch.xlsx | 65.5% | **78.8%** | +13.3pp |

---

## 🔍 Key Findings

### **Overall Performance:**
- **Average Weighted Accuracy: ~75%**
- **Best Performance:** Tweets AI Iteration 1 (80.1%)
- **Most Balanced:** Postlaunch (78.8%)

### **Why Weighted is Higher:**
The model excels at common emotions:
- **Neutral**: 72-98% accuracy (62-82% of data)
- **Joy**: 75-92% accuracy (5-15% of data)

But struggles with rare emotions:
- **Surprise**: 22-28% accuracy (only 5-18% of data)
- **Sadness**: 35-49% accuracy (only 1-5% of data)

Since neutral and joy dominate the real dataset, **weighted accuracy better reflects real-world performance**.

---

## 📈 Per-Emotion Performance

### **Best Performing Emotions:**
1. **Neutral** - 72-98% accuracy (most important due to high frequency)
2. **Joy** - 75-92% accuracy
3. **Fear** - 33-89% accuracy (varies by dataset)

### **Challenging Emotions:**
1. **Surprise** - 22-28% accuracy (needs improvement)
2. **Sadness** - 35-51% accuracy
3. **Disgust** - 46-68% accuracy (but rare, so less impact)

---

## 🔧 How to Run

### **For Iteration 1:**
```bash
cd /Users/julianordqvist/Documents/GitHub/big-data-sentiment-analysis
source venv/bin/activate
python3 Weighted_Results/calculate_weighted_accuracy.py
```

### **For Iteration 2:**
```bash
cd /Users/julianordqvist/Documents/GitHub/big-data-sentiment-analysis
source venv/bin/activate
python3 Weighted_Results/calculate_weighted_accuracy_iter2.py
```

### **Requirements:**
- pandas
- numpy
- openpyxl (for reading Excel files)

---

## 📝 Methodology Details

### **Data Sources:**

**Iteration 1:**
1. Sample: Balanced samples (390-395 tweets, ~57 per emotion)
2. Evaluation: Manual review Excel files with incorrect predictions
3. Full Dataset: Complete labeled datasets for true distributions

**Iteration 2:**
1. Sample: Balanced samples (400 tweets, ~57 per emotion)
2. Evaluation: Manual review Excel files with incorrect predictions
3. Full Dataset: Downsampled/combined labeled datasets

### **Calculation Steps:**

1. **Calculate per-emotion error rates:**
   - For each emotion: errors / sample_count
   - Example: neutral = 1 error / 58 tweets = 1.7% error

2. **Get actual proportions:**
   - From full dataset
   - Example: neutral = 62.3% of all tweets

3. **Weight the errors:**
   - Weighted Error = Σ(error_rate × proportion)
   - Example: 1.7% × 62.3% = 1.06% contribution

4. **Calculate weighted accuracy:**
   - Weighted Accuracy = 1 - Weighted Error

---

## 💡 Insights & Recommendations

### **What This Tells Us:**

✅ **Model is Production-Ready:**
- 75-80% weighted accuracy is good for emotion detection
- Strong performance on common emotions (neutral, joy)
- Acceptable tradeoff on rare emotions

⚠️ **Areas for Improvement:**
- **Surprise detection** needs work (only 22-28% accuracy)
- **Sadness** could be better (35-51% accuracy)
- Consider ensemble methods or fine-tuning for these emotions

### **For Research/Reporting:**

1. **Use weighted accuracy** when discussing model performance
2. **Mention the unweighted** for comparison
3. **Highlight**: Model performs 15-25pp better than unweighted suggests
4. **Explain**: This is because it excels at common emotions

### **For Model Improvement:**

1. **Focus on surprise & sadness** (largest error contributors)
2. **Consider class weights** during training
3. **Oversample rare emotions** in training data
4. **Review misclassified examples** for patterns

---

## 📊 Comparison with Baseline

**Random Guessing (7 emotions):** ~14.3% accuracy

**Our Model:**
- Unweighted: 53-67% (3.7-4.7× better than random)
- **Weighted: 68-80% (4.8-5.6× better than random)** ✨

The weighted accuracy shows the model is **significantly better** than it appears from unweighted metrics!

---

## 🔗 Related Files

**Evaluation Files (in root directory):**
- `Evaluation_ Tweets_AI_Labeling .xlsx`
- `Evaluation_ AfterChatGPT_Labeling.xlsx`
- `Evaluation_ tweets_ai.xlsx`
- `Evaluation_ postlaunch.xlsx`

**Sample Files:**
- `Sample_Analysis/labeled_tweets_ai_comparison.csv`
- `Sample_Analysis/emotion_sample_comparison.csv`
- `Sampling_iter2/tweets_ai_sampled_400.csv`
- `Sampling_iter2/postlaunch_sampled_400.csv`

**Full Labeled Datasets:**
- `Labeling/clean_tweets_ai.labeled.csv`
- `Labeling/AfterChatGPT.labeled.csv`
- `Sampling_iter2/tweets_ai_downsampled.labeled.csv`
- `Sampling_iter2/postlaunch.labeled.csv`

---

**Generated:** October 2025  
**Purpose:** Accurate model evaluation accounting for class imbalance  
**Result:** Model performs 68-80% (weighted) vs 53-67% (unweighted)

