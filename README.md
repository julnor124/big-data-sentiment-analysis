# Big Data Sentiment Analysis Project

A comprehensive sentiment analysis project for analyzing ChatGPT and Generative AI opinions from social media data. This project includes data preprocessing, exploratory data analysis, text cleaning, emotion classification, and statistical validation workflows.

## Project Overview

This project analyzes emotion and sentiment from multiple datasets across different time periods:
- **ChatGPT Dataset**: Social media posts about ChatGPT
- **GenerativeAI Dataset**: Social media posts about Generative AI
- **Tweets AI Dataset**: Additional AI-related social media data

The analysis pipeline includes data cleaning, preprocessing, EDA, emotion classification using Hugging Face models, statistical validation, and temporal analysis of emotional trends.

## Datasets

### Input Datasets
- `ChatGPT.csv` - Original ChatGPT social media data
- `generativeaiopinion.csv` - Original Generative AI opinion data (22K+ rows)
- `tweets_ai.csv` - Additional AI-related social media data

### Processed Datasets
- `ChatGPT_pre_clean.csv` - Preprocessed ChatGPT data (Date, Tweet columns)
- `generativeaiopinion_pre_clean.csv` - Preprocessed Generative AI data (Date, Tweet columns)
- `ChatGPT_cleaned.csv` - Fully cleaned ChatGPT data (469K+ rows)
- `GenerativeAI_cleaned.csv` - Fully cleaned Generative AI data (21K+ rows)
- `AfterChatGPT.csv` - Combined cleaned dataset (490K+ rows with Source column)

### Emotion-Labeled Datasets
- `AfterChatGPT.labeled.csv` - Emotion-labeled ChatGPT data using Hugging Face model
- `clean_tweets_ai.labeled.csv` - Emotion-labeled Tweets AI data
- `postlaunch.labeled.csv` - Emotion-labeled post-ChatGPT launch data
- `tweets_ai_downsampled.labeled.csv` - Downsampled emotion-labeled Tweets AI data

## Quick Start

### Prerequisites
- Python 3.8+ (Python 3.13.1 recommended)
- Git
- Terminal/Command Prompt

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd big-data-sentiment-analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   # venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Upgrade pip
   pip install --upgrade pip
   
   # Install all required packages
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   source venv/bin/activate
   python -c "import pandas, numpy, matplotlib, seaborn, transformers, torch; print('All packages installed successfully!')"
   ```

## Project Structure

```
big-data-sentiment-analysis/
├── venv/                                    # Virtual environment (not tracked in git)
├── requirements.txt                         # Python dependencies
├── activate_env.sh                         # Environment activation script
├── README.md                               # This file
├── .gitignore                              # Git ignore file
│
├── # Original Datasets
├── ChatGPT.csv                             # Original ChatGPT dataset (1.6M+ rows)
├── generativeaiopinion.csv                 # Original Generative AI dataset (22K+ rows)
├── tweets_ai.csv                           # Additional AI-related social media data
│
├── # Combined Output
├── AfterChatGPT.csv                        # Final combined cleaned dataset (490K+ rows)
├── AfterChatGPT_combination_report.txt     # Combination report
│
├── # Data Preprocessing and Cleaning
├── EDA_ChatGPT/                           # ChatGPT exploratory data analysis
├── EDA_GenAI/                             # Generative AI exploratory data analysis
├── EDA_tweets_ai/                         # Tweets AI exploratory data analysis
├── CLEAN_ChatGPT/                         # ChatGPT data cleaning
├── CLEAN_GenAI/                           # Generative AI data cleaning
├── clean_tweets_ai/                       # Tweets AI data cleaning
├── Clean_iter2/                           # Iteration 2 data cleaning
│
├── # Emotion Analysis
├── Emotional_Analysis/                    # Original emotion timeline analysis
├── Emotional_Analysis_iter2/              # Iteration 2 emotion timeline analysis
├── Comparison/                            # Comparison between emotion analyses
│
├── # Sampling and Labeling
├── Sampling_iter2/                        # Iteration 2 sampling and labeling
├── Labeling/                              # Manual emotion labeling results
├── Sample_Analysis/                       # Sample quality analysis and validation
│
├── # Statistical Analysis
├── Weighted_Results/                      # Weighted accuracy calculations
├── Statistical_Analysis/                  # Statistical significance testing
│
├── # Evaluation Files
├── Evaluation_ Tweets_AI_Labeling .xlsx   # Manual evaluation Iteration 1
├── Evaluation_ AfterChatGPT_Labeling.xlsx # Manual evaluation Iteration 1
├── Evaluation_ tweets_ai.xlsx            # Manual evaluation Iteration 2
├── Evaluation_ postlaunch.xlsx            # Manual evaluation Iteration 2
```

## Analysis Pipeline

### 1. Data Preprocessing
```bash
# Preprocess ChatGPT data
cd EDA_ChatGPT
python pre_clean_chatgpt.py

# Preprocess Generative AI data
cd ../EDA_GenAI
python pre_clean_generativeai.py

# Preprocess Tweets AI data
cd ../EDA_tweets_ai
python pre_cleaning.py
```

### 2. Exploratory Data Analysis
```bash
# Run general EDA for ChatGPT
cd EDA_ChatGPT
python general_eda.py
python text_eda.py

# Run general EDA for Generative AI
cd ../EDA_GenAI
python general_eda.py
python text_eda.py

# Run EDA for Tweets AI
cd ../EDA_tweets_ai
python eda_tweets_ai.py
```

### 3. Advanced Data Cleaning
```bash
# Clean ChatGPT dataset
cd CLEAN_ChatGPT
python cleaning_chatgpt.py

# Clean Generative AI dataset
cd ../CLEAN_GenAI
python GenAI_cleaning.py

# Clean Tweets AI dataset
cd ../clean_tweets_ai
python clean.py
```

### 4. Dataset Combination
```bash
# Combine cleaned datasets
cd ..
python combine_datasets.py
```

### 5. Emotion Classification
```bash
# Label emotions using Hugging Face model
cd Labeling
python emotion_labeling.py
```

### 6. Emotion Timeline Analysis
```bash
# Analyze emotion trends over time
cd Emotional_Analysis
python emotion_timeline_analysis.py

# Analyze iteration 2 datasets
cd ../Emotional_Analysis_iter2
python emotion_timeline_analysis.py
```

### 7. Statistical Validation
```bash
# Calculate weighted accuracy
cd Sample_Analysis
python calculate_weighted_accuracy.py
python calculate_weighted_accuracy_iter2.py

# Perform statistical significance testing
cd ../Statistical_Analysis
python statistical_significance_analysis.py
python population_inference_report.py
```

### 8. Comparison Analysis
```bash
# Compare emotion analyses
cd Comparison
python compare_emotional_analyses.py
```

## 🧹 Data Cleaning Features

### Cleaning Steps Applied
1. **Remove missing dates** - Eliminate rows with null dates
2. **Remove duplicates** - Remove exact duplicates (same tweet and date/time)
3. **Handle NaN values** - Remove tweets with missing content
4. **Remove stopwords** - Filter out common stopwords (if NLTK available)
5. **Remove emojis, mentions, and hashtags** - Clean social media artifacts
6. **Remove links** - Strip URLs and links
7. **Remove special characters** - Keep only alphanumeric and basic punctuation
8. **Clean whitespace and lowercase** - Normalize text formatting

### Text Processing Features
- **Emoji removal** using `emoji` library with regex fallback
- **Hashtag removal** with comprehensive patterns
- **Mention removal** (@username patterns)
- **Link removal** (http, https, www, t.co patterns)
- **Special character filtering** (preserves basic punctuation for sentiment)
- **Whitespace normalization** and lowercasing

## 📈 Analysis Features

### General EDA
- Dataset format inspection
- Data type analysis
- Missing value analysis
- Duplicate detection
- Date distribution analysis
- Tweet length analysis
- Memory usage tracking

### Text Analysis
- Tweet structure analysis
- Word frequency analysis
- Hashtag and mention analysis
- Language detection (using TextBlob)
- Tweet length distribution
- Content theme analysis

### Visualizations
- Dataset size comparisons
- Missing value heatmaps
- Date distribution plots
- Tweet length histograms and box plots
- Word/hashtag/mention frequency charts
- Language distribution plots
- Cleaning process visualizations

## Dependencies

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Basic plotting
- **seaborn** - Statistical data visualization

### NLP and ML Libraries
- **nltk** - Natural language processing (stopwords)
- **textblob** - Language detection and sentiment analysis
- **transformers** - Hugging Face transformers for emotion classification
- **torch** - PyTorch for deep learning
- **scikit-learn** - Machine learning algorithms and statistical tests

### Statistical Analysis Libraries
- **scipy** - Statistical functions and tests
- **statsmodels** - Advanced statistical modeling
- **openpyxl** - Excel file reading/writing

### Text Processing Libraries
- **emoji** - Emoji handling (optional)
- **re** - Regular expressions for text cleaning

### Visualization Libraries
- **plotly** - Interactive visualizations (optional)

## Dataset Statistics

### Final Combined Dataset (`AfterChatGPT.csv`)
- **Total rows**: 490,457
- **Columns**: Date, Tweet, Source
- **Memory usage**: ~124 MB
- **Date range**: 2017-2023
- **Sources**: ChatGPT (469K rows), GenAI (21K rows)

### Data Quality
- ✅ No missing dates
- ✅ No duplicates
- ✅ No NaN values
- ✅ Cleaned text (no emojis, hashtags, mentions, links)
- ✅ Normalized punctuation and whitespace
- ✅ Lowercased text

## Emotion Analysis Results

### Model Performance
- **Model**: `j-hartmann/emotion-english-distilroberta-base`
- **Weighted Accuracy**: 68-80% across all datasets
- **Unweighted Accuracy**: 53-67% (traditional accuracy)
- **Statistical Significance**: All p < 0.001 (highly significant)

### Per-Emotion Accuracy Ranges
- **Neutral**: 72-98% (highest performance)
- **Joy**: 72-93% (strong performance)
- **Anger**: 46-77% (moderate performance)
- **Disgust**: 46-77% (moderate performance)
- **Surprise**: 22-78% (variable performance)
- **Sadness**: 33-51% (challenging emotion)
- **Fear**: 22-89% (highly variable)

### Temporal Emotion Trends
- **Pre-ChatGPT (2017-2022)**: Dominated by neutral and joy
- **Post-ChatGPT (2022-2023)**: Increased anger and fear
- **Statistical Validation**: Chi-square tests confirm significant temporal differences (p < 0.001)

## Statistical Validation

### Weighted Accuracy Results
- **Tweets AI (Iteration 1)**: 80.09% ± 2.92%
- **AfterChatGPT (Iteration 1)**: 74.68% ± 6.31%
- **Tweets AI (Iteration 2)**: 67.93% ± 9.49%
- **Postlaunch (Iteration 2)**: 78.84% ± 6.05%

### Statistical Tests Performed
- **Chi-square tests**: Significant differences across emotions (p < 0.001)
- **Binomial tests**: Model significantly outperforms random baseline (p < 0.001)
- **Two-proportion Z-tests**: Iteration comparisons
- **Confidence intervals**: 95% CI for population estimates
- **Effect size calculations**: Cohen's h for practical significance

## Usage Examples

### Activating the Environment
```bash
# Option 1: Manual activation
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate    # Windows

# Option 2: Use the activation script
./activate_env.sh  # macOS/Linux
```

### Running the Complete Pipeline
```bash
# Activate environment
source venv/bin/activate

# Run preprocessing
cd EDA_ChatGPT && python pre_clean_chatgpt.py
cd ../EDA_GenAI && python pre_clean_generativeai.py
cd ../EDA_tweets_ai && python pre_cleaning.py

# Run EDA
cd ../EDA_ChatGPT && python general_eda.py && python text_eda.py
cd ../EDA_GenAI && python general_eda.py && python text_eda.py
cd ../EDA_tweets_ai && python eda_tweets_ai.py

# Run cleaning
cd ../CLEAN_ChatGPT && python cleaning_chatgpt.py
cd ../CLEAN_GenAI && python GenAI_cleaning.py
cd ../clean_tweets_ai && python clean.py

# Combine datasets
cd .. && python combine_datasets.py

# Label emotions
cd Labeling && python emotion_labeling.py

# Analyze emotions
cd ../Emotional_Analysis && python emotion_timeline_analysis.py
cd ../Emotional_Analysis_iter2 && python emotion_timeline_analysis.py

# Statistical validation
cd ../Sample_Analysis && python calculate_weighted_accuracy.py
cd ../Statistical_Analysis && python statistical_significance_analysis.py
```

### Accessing Results
```bash
# View emotion analysis reports
cat Emotional_Analysis/emotion_timeline_analysis_report.txt
cat Emotional_Analysis_iter2/emotion_timeline_analysis_report.txt

# View statistical results
cat Statistical_Analysis/statistical_significance_report.txt
cat Weighted_Results/weighted_accuracy_report.txt

# View comparison analysis
cat Comparison/emotional_analysis_comparison_report.txt
```

## 🤝 Contributing

1. Always work in the virtual environment
2. Install new packages and update `requirements.txt`:
   ```bash
   pip freeze > requirements.txt
   ```
3. Follow the project structure
4. Test your code before committing
5. Update documentation for new features

## 📝 Notes

- The virtual environment is not tracked in git (see `.gitignore`)
- Each team member needs to create their own virtual environment
- Always activate the environment before working on the project
- Keep `requirements.txt` updated when adding new dependencies
- The cleaning pipeline preserves punctuation for sentiment analysis
- All scripts generate detailed reports and visualizations

## 🔍 Troubleshooting

### Common Issues

1. **Python not found**
   - Make sure Python 3.8+ is installed
   - Try `python3` instead of `python`

2. **Permission denied on activation script**
   ```bash
   chmod +x activate_env.sh
   ```

3. **Package installation fails**
   - Make sure virtual environment is activated
   - Try upgrading pip: `pip install --upgrade pip`
   - Check internet connection

4. **NLTK data not found**
   - The scripts will automatically download required NLTK data
   - If issues persist, manually download: `python -m nltk.downloader stopwords punkt`

5. **Emoji library not available**
   - Scripts will use regex fallback for emoji removal
   - Install emoji library for better coverage: `pip install emoji`

### Getting Help

- Check Python version: `python --version`
- Check installed packages: `pip list`
- Verify virtual environment: `which python` (should show venv path)
- Check script outputs for detailed error messages

---

## Project Status

✅ **Data Preprocessing** - Complete  
✅ **Exploratory Data Analysis** - Complete  
✅ **Advanced Data Cleaning** - Complete  
✅ **Dataset Combination** - Complete  
✅ **Emotion Classification** - Complete  
✅ **Emotion Timeline Analysis** - Complete  
✅ **Statistical Validation** - Complete  
✅ **Model Performance Evaluation** - Complete  
✅ **Temporal Trend Analysis** - Complete  

The project has successfully completed a comprehensive emotion analysis pipeline with statistical validation. All analyses are complete and documented with detailed reports and visualizations.