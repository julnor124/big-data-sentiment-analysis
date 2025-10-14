# Statistical Significance Analysis
## Validating Model Performance with Statistical Tests

This folder contains comprehensive statistical analyses to validate the weighted accuracy results and test for significant differences.

## 📁 Contents

### **Analysis Scripts:**
- **`statistical_significance_analysis.py`** - Main statistical testing script
- **`population_inference_report.py`** - Population accuracy estimation
- **`create_statistical_visualizations.py`** - Visualization generator

### **Reports:**
- **`statistical_significance_report.txt`** - Comprehensive statistical report
- **`population_inference_report.txt`** - Population inference report
- **`statistical_output.txt`** - Console output from significance analysis
- **`population_inference_output.txt`** - Console output from population inference

### **Visualizations:**
- **`statistical_analysis_visualization.png`** - 12 subplots with comprehensive metrics
- **`statistical_summary.png`** - Text-based summary figure

---

## 🎯 What We Test

### **1. Confidence Intervals**
- **Bootstrap CI** for weighted accuracy (more robust)
- **Normal approximation CI** for unweighted accuracy
- **95% confidence level** used throughout

### **2. Chi-Square Test**
- **Question:** Do error rates differ significantly across emotions?
- **Result:** YES - all datasets show p < 0.001
- **Conclusion:** Validates need for weighted accuracy

### **3. Binomial Test**
- **Question:** Is model better than random guessing (14.3%)?
- **Baseline:** 1/7 = 14.3% (random for 7 emotions)
- **Result:** All models p < 0.001 (highly significant)
- **Conclusion:** Model is 3.7-5.6× better than random

### **4. Effect Size (Cohen's h)**
- **Question:** How meaningful is weighted vs unweighted difference?
- **Ranges:** Small (<0.2), Medium (0.2-0.5), Large (>0.5)
- **Results:** Small to Large effects found

### **5. Two-Proportion Z-Test**
- **Question:** Are iterations significantly different?
- **Compares:** Iter1 vs Iter2 for same dataset types
- **Results:** Pre-ChatGPT differs, Post-ChatGPT doesn't

---

## 📊 Key Statistical Results

### **CONFIDENCE INTERVALS (95%):**

| Dataset | Weighted Accuracy | 95% CI | Precision |
|---------|-------------------|---------|-----------|
| **Tweets AI (Iter1)** | 80.09% | [76.92%, 82.76%] | ±2.92% ⭐ |
| **AfterChatGPT (Iter1)** | 74.68% | [68.13%, 80.74%] | ±6.31% |
| **Tweets AI (Iter2)** | 67.93% | [58.05%, 77.04%] | ±9.49% |
| **Postlaunch (Iter2)** | 78.84% | [72.43%, 84.53%] | ±6.05% |

**Interpretation:**
- ✅ Narrower CI = more reliable estimate
- ✅ All well above random baseline (14.3%)
- ✅ Tweets AI Iter1 has best precision (±2.92%)

---

### **CHI-SQUARE TEST RESULTS:**

| Dataset | χ² Statistic | p-value | Significant? |
|---------|--------------|---------|--------------|
| Tweets AI (Iter1) | 116.20 | 1.02e-22 | ✅ YES |
| AfterChatGPT (Iter1) | 37.50 | 1.41e-06 | ✅ YES |
| Tweets AI (Iter2) | 48.95 | 7.62e-09 | ✅ YES |
| Postlaunch (Iter2) | 69.49 | 5.19e-13 | ✅ YES |

**Interpretation:**
- ✅ **Error rates DIFFER significantly across emotions**
- ✅ Validates the need for weighted accuracy
- ✅ Cannot treat all emotions equally

---

### **BINOMIAL TEST (vs Random):**

| Dataset | Model Accuracy | p-value | Improvement |
|---------|---------------|---------|-------------|
| Tweets AI (Iter1) | 56.2% | 1.82e-82 | **3.9× better** |
| AfterChatGPT (Iter1) | 67.3% | 4.04e-127 | **4.7× better** |
| Tweets AI (Iter2) | 53.0% | 1.05e-73 | **3.7× better** |
| Postlaunch (Iter2) | 65.5% | 8.58e-121 | **4.6× better** |

**Baseline:** 14.3% (random guessing for 7 emotions)

**Interpretation:**
- ✅ **ALL models HIGHLY significant** (p < 0.001)
- ✅ Models perform 3.7-4.7× better than random
- ✅ Results are NOT due to chance

---

### **EFFECT SIZES (Weighted vs Unweighted):**

| Dataset | Difference | Cohen's h | Effect Size |
|---------|------------|-----------|-------------|
| Tweets AI (Iter1) | +23.9pp | 0.522 | **LARGE** ⭐ |
| AfterChatGPT (Iter1) | +7.3pp | 0.162 | Small |
| Tweets AI (Iter2) | +14.9pp | 0.307 | Medium |
| Postlaunch (Iter2) | +13.3pp | 0.300 | Medium |

**Interpretation:**
- ✅ Large effect for Tweets AI Iter1 (class imbalance matters most)
- ✅ Medium effects confirm weighted accuracy is meaningful
- ✅ Smaller effect for AfterChatGPT (more balanced dataset)

---

### **ITERATION COMPARISONS:**

#### **Pre-ChatGPT: Iter1 vs Iter2**
- **Iter1:** 80.09% ± 2.92%
- **Iter2:** 67.93% ± 9.49%
- **Difference:** 12.16pp
- **Z-statistic:** 3.893
- **p-value:** 0.0001 ✅ **SIGNIFICANT**
- **Conclusion:** Iter1 significantly outperforms Iter2

#### **Post-ChatGPT: Iter1 vs Iter2**
- **Iter1:** 74.68% ± 6.31%
- **Iter2:** 78.84% ± 6.05%
- **Difference:** -4.16pp
- **Z-statistic:** -1.388
- **p-value:** 0.165 ✗ **NOT SIGNIFICANT**
- **Conclusion:** No significant difference

#### **Iter1: Pre vs Post ChatGPT**
- **Pre:** 80.09%
- **Post:** 74.68%
- **Difference:** 5.41pp
- **p-value:** 0.0701 ✗ **NOT SIGNIFICANT** (marginal)
- **Conclusion:** Slight but not significant difference

---

## 🔬 Statistical Methodology

### **Tests Used:**

1. **Bootstrap Confidence Intervals (10,000 iterations)**
   - Non-parametric method
   - No distribution assumptions
   - Robust for weighted metrics

2. **Chi-Square Test of Independence**
   - Tests if error rates vary by emotion
   - H₀: Error rates are equal across emotions
   - H₁: Error rates differ by emotion

3. **Binomial Test (Exact)**
   - Tests if model > random baseline
   - H₀: Accuracy = 14.3% (random)
   - H₁: Accuracy > 14.3%

4. **Cohen's h (Effect Size)**
   - Measures practical significance
   - |h| < 0.2: small, 0.2-0.5: medium, >0.5: large
   - Complements p-values

5. **Two-Proportion Z-Test**
   - Compares two accuracy rates
   - Tests if difference is significant
   - Used for iteration comparisons

### **Significance Levels:**
- **p < 0.001:** Highly significant (⭐⭐⭐)
- **p < 0.01:** Very significant (⭐⭐)
- **p < 0.05:** Significant (⭐)
- **p ≥ 0.05:** Not significant

---

## 💡 Key Insights

### **✅ What We Can Say with Confidence:**

1. **Model is Statistically Valid:**
   - Significantly better than random (p < 0.001)
   - All datasets show consistent results
   - Not due to chance or luck

2. **Weighted Accuracy is Necessary:**
   - Chi-square shows errors differ by emotion (p < 0.001)
   - Effect sizes confirm meaningful differences
   - Reflects real-world performance better

3. **Results are Reliable:**
   - Confidence intervals are reasonably narrow
   - Bootstrap validation confirms robustness
   - Consistent across multiple tests

4. **Iteration 1 Pre-ChatGPT is Best:**
   - 80.09% ± 2.92% (most precise)
   - Significantly better than Iter2 (p < 0.001)
   - Large effect size (Cohen's h = 0.522)

### **⚠️ Important Caveats:**

1. **Post-ChatGPT Iterations are Equivalent:**
   - No significant difference (p = 0.165)
   - Either can be used
   - ~75-79% weighted accuracy range

2. **Pre vs Post ChatGPT:**
   - Difference is marginal (p = 0.0701)
   - Not quite significant at α = 0.05
   - Suggests similar model performance

3. **Sample Size Matters:**
   - Iter2 has wider CIs (smaller effective sample)
   - Iter1 more precise due to larger balanced sample
   - Consider when choosing dataset

---

## 📈 Practical Implications

### **For Reporting:**

✅ **CAN report:**
- "Model achieves 68-80% weighted accuracy"
- "Performance significantly exceeds random baseline (p < 0.001)"
- "Error rates differ significantly by emotion (χ² p < 0.001)"
- "Weighted accuracy 7-24pp higher than unweighted (medium-large effect)"

✅ **Should include:**
- Confidence intervals for transparency
- Statistical significance tests
- Effect sizes for practical significance

### **For Model Selection:**

**Best Performance:**
- Use **Tweets AI Iter1** for pre-ChatGPT (80.09% ± 2.92%)
- Use **Postlaunch Iter2** for post-ChatGPT (78.84% ± 6.05%)

**Most Reliable:**
- Narrower CI = more reliable
- Tweets AI Iter1 has best precision

### **For Future Work:**

1. **Focus on High-Error Emotions:**
   - Chi-square confirms different error rates
   - Target surprise, sadness for improvement
   - Consider emotion-specific strategies

2. **Use Weighted Metrics:**
   - Statistically validated approach
   - Reflects real distribution
   - More meaningful for production

3. **Consider Sample Size:**
   - Larger samples → narrower CIs
   - Balance vs precision tradeoff
   - Bootstrap for robust estimates

---

## 🔧 How to Run

```bash
cd /Users/julianordqvist/Documents/GitHub/big-data-sentiment-analysis
source venv/bin/activate
python3 Statistical_Analysis/statistical_significance_analysis.py
```

### **Requirements:**
- pandas
- numpy
- scipy
- openpyxl

### **Output Files:**
- `statistical_significance_report.txt` - Detailed report
- `statistical_output.txt` - Console output with all tests

---

## 📚 Statistical Interpretation Guide

### **P-Values:**
- **p < 0.001:** "highly significant" - very strong evidence
- **p < 0.01:** "very significant" - strong evidence
- **p < 0.05:** "significant" - good evidence
- **p < 0.10:** "marginally significant" - weak evidence
- **p ≥ 0.10:** "not significant" - insufficient evidence

### **Confidence Intervals:**
- **Narrow CI:** More precise estimate, more reliable
- **Wide CI:** Less precise, need more data
- **Non-overlapping CIs:** Likely significant difference
- **Overlapping CIs:** May not be significantly different

### **Effect Sizes (Cohen's h):**
- **0.2:** "Small" - detectable but minor practical impact
- **0.5:** "Medium" - moderate practical significance
- **0.8:** "Large" - substantial practical importance

---

## 🎓 References & Methods

### **Statistical Tests:**
- **Bootstrap:** Efron & Tibshirani (1993)
- **Chi-square:** Pearson (1900)
- **Binomial:** Clopper & Pearson (1934)
- **Effect size:** Cohen (1988)
- **Two-proportion z-test:** Standard statistical test

### **Significance Testing:**
- α = 0.05 (standard significance level)
- Two-tailed tests (unless specified)
- Multiple comparison aware (be cautious)

---

**Generated:** October 2025  
**Purpose:** Statistical validation of weighted accuracy results  
**Conclusion:** Model performance is statistically significant and robust

