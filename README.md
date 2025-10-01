# Big Data Sentiment Analysis Project

A comprehensive data analysis project for sentiment analysis using Python, featuring machine learning, NLP, and data visualization tools.

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
   python example_analysis.py
   ```

## What's Included

### Core Data Analysis
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scipy** - Scientific computing

### Visualization
- **matplotlib** - Basic plotting
- **seaborn** - Statistical data visualization
- **plotly** - Interactive visualizations

### Machine Learning & NLP
- **scikit-learn** - Machine learning algorithms
- **nltk** - Natural language processing
- **spacy** - Advanced NLP
- **transformers** - Hugging Face transformers
- **torch** - PyTorch for deep learning

### Sentiment Analysis
- **textblob** - Simple sentiment analysis
- **vaderSentiment** - VADER sentiment analysis

### Development Tools
- **jupyter** - Jupyter notebooks
- **ipykernel** - Jupyter kernel support

### Data Sources
- **tweepy** - Twitter API
- **requests** - HTTP requests

### Utilities
- **python-dotenv** - Environment variables
- **tqdm** - Progress bars

## Usage

### Activating the Environment

**Option 1: Manual activation**
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate    # Windows
```

**Option 2: Use the activation script**
```bash
./activate_env.sh  # macOS/Linux
```

### Running Jupyter Lab
```bash
# Make sure virtual environment is activated
source venv/bin/activate
jupyter lab
```

### Deactivating the Environment
```bash
deactivate
```

## 📁 Project Structure

```
big-data-sentiment-analysis/
├── venv/                    # Virtual environment (not tracked in git)
├── requirements.txt         # Python dependencies
├── activate_env.sh         # Environment activation script
├── example_analysis.py     # Example analysis script
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## 🔧 Troubleshooting

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

4. **Jupyter not found**
   - Ensure virtual environment is activated
   - Reinstall jupyter: `pip install jupyter`

### Getting Help

- Check Python version: `python --version`
- Check installed packages: `pip list`
- Verify virtual environment: `which python` (should show venv path)

## Example Usage

The project includes an example script (`example_analysis.py`) that demonstrates:
- Basic data manipulation with pandas
- Sentiment analysis using VADER
- Data visualization with matplotlib/seaborn
- Saving plots to files

Run it to verify everything is working:
```bash
source venv/bin/activate
python example_analysis.py
```

## 🤝 Contributing

1. Always work in the virtual environment
2. Install new packages and update `requirements.txt`:
   ```bash
   pip freeze > requirements.txt
   ```
3. Follow the project structure
4. Test your code before committing

## 📝 Notes

- The virtual environment is not tracked in git (see `.gitignore`)
- Each team member needs to create their own virtual environment
- Always activate the environment before working on the project
- Keep `requirements.txt` updated when adding new dependencies

---