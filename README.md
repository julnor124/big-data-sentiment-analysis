# Big Data Sentiment Analysis Project

A comprehensive sentiment analysis project for analyzing ChatGPT and Generative AI opinions from social media data. This project includes data preprocessing, exploratory data analysis, text cleaning, and dataset combination workflows.

## 🎯 Project Overview

This project analyzes sentiment from two datasets:
- **ChatGPT Dataset**: Social media posts about ChatGPT
- **GenerativeAI Dataset**: Social media posts about Generative AI

The analysis pipeline includes data cleaning, preprocessing, EDA, and sentiment analysis preparation.

## 📊 Datasets

### Input Datasets
- `ChatGPT.csv` - Original ChatGPT social media data (1.6M+ rows)
- `generativeaiopinion.csv` - Original Generative AI opinion data (22K+ rows)

### Processed Datasets
- `ChatGPT_pre_clean.csv` - Preprocessed ChatGPT data (Date, Tweet columns)
- `generativeaiopinion_pre_clean.csv` - Preprocessed Generative AI data (Date, Tweet columns)
- `ChatGPT_cleaned.csv` - Fully cleaned ChatGPT data (469K+ rows)
- `GenerativeAI_cleaned.csv` - Fully cleaned Generative AI data (21K+ rows)
- `AfterChatGPT.csv` - Combined cleaned dataset (490K+ rows with Source column)

## 🚀 Quick Start

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
   python -c "import pandas, numpy, matplotlib, seaborn; print('✅ All packages installed successfully!')"
   ```

## 📁 Project Structure

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
│
├── # Combined Output
├── AfterChatGPT.csv                        # Final combined cleaned dataset (490K+ rows)
├── AfterChatGPT_combination_report.txt     # Combination report
│
├── # ChatGPT Analysis Pipeline
├── EDA_ChatGPT/
│   ├── pre_clean_chatgpt.py               # Preprocessing script
│   ├── ChatGPT_pre_clean.csv              # Preprocessed data (Date, Tweet)
│   ├── general_eda.py                     # General EDA analysis
│   ├── chatgpt_general_eda_report.txt     # General EDA report
│   ├── chatgpt_general_eda_visualizations.png # General EDA plots
│   ├── text_eda.py                        # Text-specific EDA
│   ├── chatgpt_text_analysis_report.txt   # Text analysis report
│   └── chatgpt_text_analysis_visualizations.png # Text analysis plots
│
├── CLEAN_ChatGPT/
│   ├── cleaning_chatgpt.py                # Advanced cleaning script
│   ├── ChatGPT_cleaned.csv                # Final cleaned dataset
│   ├── ChatGPT_cleaning_report.txt        # Cleaning report
│   └── ChatGPT_cleaning_visualizations.png # Cleaning visualizations
│
├── # Generative AI Analysis Pipeline
├── EDA_GenAI/
│   ├── pre_clean_generativeai.py          # Preprocessing script
│   ├── generativeaiopinion_pre_clean.csv  # Preprocessed data (Date, Tweet)
│   ├── general_eda.py                     # General EDA analysis
│   ├── generativeai_general_eda_report.txt # General EDA report
│   ├── generativeai_general_eda_visualizations.png # General EDA plots
│   ├── text_eda.py                        # Text-specific EDA
│   ├── generativeai_text_analysis_report.txt # Text analysis report
│   └── generativeai_text_analysis_visualizations.png # Text analysis plots
│
└── CLEAN_GenAI/
    ├── GenAI_cleaning.py                  # Advanced cleaning script
    ├── GenerativeAI_cleaned.csv           # Final cleaned dataset
    ├── GenerativeAI_cleaning_report.txt   # Cleaning report
    └── GenerativeAI_cleaning_visualizations.png # Cleaning visualizations
```

## 🔄 Analysis Pipeline

### 1. Data Preprocessing
```bash
# Preprocess ChatGPT data
cd EDA_ChatGPT
python pre_clean_chatgpt.py

# Preprocess Generative AI data
cd ../EDA_GenAI
python pre_clean_generativeai.py
```

### 2. Exploratory Data Analysis
```bash
# Run general EDA for ChatGPT
cd EDA_ChatGPT
python general_eda.py

# Run text EDA for ChatGPT
python text_eda.py

# Run general EDA for Generative AI
cd ../EDA_GenAI
python general_eda.py

# Run text EDA for Generative AI
python text_eda.py
```

### 3. Advanced Data Cleaning
```bash
# Clean ChatGPT dataset
cd CLEAN_ChatGPT
python cleaning_chatgpt.py

# Clean Generative AI dataset
cd ../CLEAN_GenAI
python GenAI_cleaning.py
```

### 4. Dataset Combination
```bash
# Combine cleaned datasets
cd ..
python combine_datasets.py
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

## 🛠️ Dependencies

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Basic plotting
- **seaborn** - Statistical data visualization

### NLP Libraries
- **nltk** - Natural language processing (stopwords)
- **textblob** - Language detection and sentiment analysis
- **emoji** - Emoji handling (optional)
- **re** - Regular expressions for text cleaning

### Optional Libraries
- **plotly** - Interactive visualizations
- **scikit-learn** - Machine learning algorithms
- **transformers** - Hugging Face transformers
- **torch** - PyTorch for deep learning

## 📊 Dataset Statistics

### Final Combined Dataset (`AfterChatGPT.csv`)
- **Total rows**: 490,457
- **Columns**: Date, Tweet, Source
- **Memory usage**: ~124 MB
- **Date range**: 2023-2024
- **Sources**: ChatGPT (469K rows), GenAI (21K rows)

### Data Quality
- ✅ No missing dates
- ✅ No duplicates
- ✅ No NaN values
- ✅ Cleaned text (no emojis, hashtags, mentions, links)
- ✅ Normalized punctuation and whitespace
- ✅ Lowercased text

## 🔧 Usage Examples

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

# Run EDA
cd ../EDA_ChatGPT && python general_eda.py && python text_eda.py
cd ../EDA_GenAI && python general_eda.py && python text_eda.py

# Run cleaning
cd ../CLEAN_ChatGPT && python cleaning_chatgpt.py
cd ../CLEAN_GenAI && python GenAI_cleaning.py

# Combine datasets
cd .. && python combine_datasets.py
```

### Accessing Results
```bash
# View final combined dataset
head AfterChatGPT.csv

# View cleaning reports
cat CLEAN_ChatGPT/ChatGPT_cleaning_report.txt
cat CLEAN_GenAI/GenerativeAI_cleaning_report.txt

# View combination report
cat AfterChatGPT_combination_report.txt
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

## 📊 Project Status

✅ **Data Preprocessing** - Complete  
✅ **Exploratory Data Analysis** - Complete  
✅ **Advanced Data Cleaning** - Complete  
✅ **Dataset Combination** - Complete  
🔄 **Sentiment Analysis** - Ready for implementation  

The project is now ready for sentiment analysis implementation using the cleaned `AfterChatGPT.csv` dataset.