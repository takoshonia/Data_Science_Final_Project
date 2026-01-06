# Quick Start Guide - Urban Pulse Project

## 🚀 Getting Started in 5 Steps

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Download the Dataset

1. Go to [Kaggle - Metro Interstate Traffic Volume](https://www.kaggle.com/datasets) or [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
2. Download the dataset CSV file
3. Place it in `data/raw/` directory
4. Rename it to `Metro_Interstate_Traffic_Volume.csv` (or update the path in notebooks)

### Step 3: Run Jupyter Notebooks in Order

Open Jupyter Notebook:
```bash
jupyter notebook
```

Then run notebooks in this order:

1. **`notebooks/01_data_exploration.ipynb`**
   - Loads and inspects raw data
   - Generates initial data quality report

2. **`notebooks/02_data_preprocessing.ipynb`**
   - Cleans missing values
   - Handles outliers
   - Creates derived features
   - Saves processed data

3. **`notebooks/03_eda_visualization.ipynb`**
   - Creates 7+ visualizations
   - Performs statistical analysis
   - Discovers patterns

4. **`notebooks/04_machine_learning.ipynb`**
   - Trains Logistic Regression model
   - Trains Decision Tree model
   - Compares and evaluates models

### Step 4: Review Results

- Check `reports/figures/` for all visualizations
- Review model outputs in the ML notebook
- Check `data/processed/` for cleaned data

### Step 5: Customize (Optional)

- Adjust preprocessing parameters in `02_data_preprocessing.ipynb`
- Modify visualization functions in `src/visualization.py`
- Tune ML model parameters in `04_machine_learning.ipynb`

## 📝 Important Notes

### If Dataset Column Names Differ

If your dataset has different column names, update:
- `src/data_processing.py` - preprocessing functions
- `notebooks/02_data_preprocessing.ipynb` - column references
- `notebooks/03_eda_visualization.ipynb` - visualization calls

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
- **Solution**: Make sure you're running notebooks from the `notebooks/` directory, or update the `sys.path.append()` in each notebook

**Issue**: File not found errors
- **Solution**: Check that data file is in `data/raw/` with correct name

**Issue**: Import errors
- **Solution**: Make sure all dependencies are installed: `pip install -r requirements.txt`

## 🎯 Expected Outputs

After running all notebooks, you should have:

- ✅ Cleaned dataset in `data/processed/traffic_cleaned.csv`
- ✅ 8+ visualizations in `reports/figures/`
- ✅ Trained models with evaluation metrics
- ✅ Model comparison results
- ✅ Feature importance analysis

## 📊 Project Structure

```
DS_Final_Project/
├── data/
│   ├── raw/                    # Put your CSV here
│   ├── processed/              # Cleaned data (generated)
│   └── DATA_DICTIONARY.md      # Column descriptions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda_visualization.ipynb
│   └── 04_machine_learning.ipynb
├── src/
│   ├── data_processing.py       # Cleaning functions
│   ├── visualization.py        # Plotting functions
│   └── models.py              # ML model functions
├── reports/
│   ├── figures/                # Generated plots
│   └── results/                # Model outputs
├── README.md
├── CONTRIBUTIONS.md
├── QUICKSTART.md               # This file
└── requirements.txt
```

## 🆘 Need Help?

1. Check the `README.md` for detailed documentation
2. Review `DATA_DICTIONARY.md` for column information
3. Check notebook markdown cells for explanations
4. Review function docstrings in `src/` modules

## ✅ Checklist Before Submission

- [ ] All notebooks run without errors
- [ ] Dataset is properly cleaned
- [ ] 5+ visualizations created
- [ ] 2 ML models trained and compared
- [ ] Results documented
- [ ] README.md updated with your information
- [ ] CONTRIBUTIONS.md filled out
- [ ] Code is well-commented
- [ ] requirements.txt is complete

Good luck with your project! 🎉

