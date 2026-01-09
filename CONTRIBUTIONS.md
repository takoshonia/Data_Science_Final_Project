# Team Contributions

## Project: Urban Pulse – Predicting City Traffic Stress

### Team Members
- Barbare Pantskhava
- Tamar Shonia

---

## Contribution Breakdown

### [Team Member 1 Name]
**Role**: Data Engineer / Preprocessing Specialist

**Contributions**:
- Set up project structure and repository
- Implemented data preprocessing pipeline (`src/data_processing.py`)
- Created data cleaning functions with error handling for file loading and input validation
- Developed feature engineering logic (rush hour detection, day type classification, traffic stress levels)
- Handled missing values and outlier detection using IQR method
- Created data quality reports and documentation
- Created `notebooks/01_data_exploration.ipynb`
- Created `notebooks/02_data_preprocessing.ipynb`
- Wrote data dictionary documentation (`data/DATA_DICTIONARY.md`)
- Created static visualization functions (`src/visualization.py`)
- Created `notebooks/03_eda_visualization.ipynb`
- Performed statistical analysis and correlation studies
- Identified key patterns and insights from exploratory data analysis

**Files Modified/Created**:
- `src/data_processing.py`
- `src/visualization.py`
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_data_preprocessing.ipynb`
- `notebooks/03_eda_visualization.ipynb`
- `data/DATA_DICTIONARY.md`
- `README.md` (data processing and dataset sections)
- `reports/figures/` (static visualization outputs)

---

### [Team Member 2 Name]
**Role**: ML Engineer / Visualization Specialist

**Contributions**:
- Created interactive visualization functions using Plotly (`src/interactive_visualization.py`)
- Created `notebooks/05_interactive_visualizations.ipynb`
- Implemented Logistic Regression model
- Implemented Decision Tree model
- Implemented Random Forest model (bonus third model)
- Conducted comprehensive model evaluation and comparison
- Created model utility functions (`src/models.py`)
- Created `notebooks/04_machine_learning.ipynb`
- Generated model performance reports and metrics
- Developed Streamlit interactive dashboard (`dashboard.py`)
- Created comprehensive README.md with results and methodology
- Created project documentation (`PROJECT_SUMMARY.md`, `QUICKSTART.md`)
- Ensured code quality and PEP 8 compliance

**Files Modified/Created**:
- `src/interactive_visualization.py`
- `src/models.py`
- `dashboard.py`
- `notebooks/04_machine_learning.ipynb`
- `notebooks/05_interactive_visualizations.ipynb`
- `README.md` (main documentation)
- `PROJECT_SUMMARY.md`
- `QUICKSTART.md`
- `reports/figures/` (interactive visualization outputs)
- `reports/results/` (model outputs)

---

## Collaboration Notes

### Communication
- **Meetings**: Regular team meetings to coordinate work and review progress
- **Tools**: Regular communication for integration and code sharing
- **Timeline**: Followed project timeline as outlined in guidelines

### Code Review Process
- All code was reviewed by the other team member before finalizing
- Ensured consistency in coding style and documentation
- Verified integration between data processing and analysis components

### Workflow
- **Phase 1**: Team Member 1 focused on data exploration and preprocessing pipeline
- **Phase 2**: Team Member 1 created static visualizations while Team Member 2 began ML model development
- **Phase 3**: Team Member 2 completed ML models, interactive visualizations, and dashboard
- **Integration**: Both members collaborated on final documentation and project integration

### Challenges Overcome
- Coordinated data preprocessing output format to ensure compatibility with visualization and modeling functions
- Integrated static and interactive visualization modules to work with same data pipeline
- Ensured consistent feature engineering between preprocessing and model training

