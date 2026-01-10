# NHL-Data-project-ELIAS-WIJKSTROM
## reasearch question
Can we predict playoff qualification and playoff outcomes based on regular season performance metrics using machine learning?


## Data

- **Source**: NHL team-level statistics from Natural Stat Trick  
- **Scope**: Regular season team performance metrics across multiple NHL seasons  
- **Features**: Shot-based metrics, expected goals, scoring chances, and danger-based statistics  
- **Target variables**:
  - `made_playoffs` (binary classification)
  - `round_reached` (multi-class classification)

The dataset used for modeling has been cleaned and preprocessed prior to analysis.


## Methods

The following machine learning models are used in this project:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Models are trained to predict playoff qualification and, conditional on qualification, playoff rounds reached. Model performance is evaluated using accuracy metrics and confusion matrices.

---

## Running the Project

### Environment Setup

This project uses a Conda environment. To create and activate the environment, run:

conda env create -f environment.yml
conda activate NHL-Data-project

### usage
python main.py

Expected output: Accuracy of prediction models tables, confusion matrices


## Project Structure

```
NHL-Data-project-ELIAS-WIJKSTROM/
├── main.py                 # Main entry point
├── project_report.pdf      # Final report 
├── project_report.md       # Markdown report source
├── environment.yml         # Conda environment
├── README.md               # Project overview
├── data/
│   ├── raw/                # Raw input data
│   └── clean/              # Cleaned dataset
├── notebooks/
│   └── EDA.ipynb           # Exploratory analysis
├── src/
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── models.py           # Model definitions
│   └── evaluation.py       # Evaluation metrics and plots
└── results/
                            # Generated plots and tables
              
```

## requirements

- python=3.11
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter



