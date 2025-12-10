# Student Performance Indicator - End-to-End Machine Learning Project

A comprehensive MLOps project that predicts student math scores based on various demographic and educational factors. This project demonstrates a complete machine learning pipeline from data ingestion to model deployment.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
- [Components](#components)
- [Model Training](#model-training)
- [Logging](#logging)
- [Error Handling](#error-handling)

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline to predict student math scores using various features such as gender, ethnicity, parental education level, lunch type, and test preparation course. The project follows MLOps best practices with modular code structure, comprehensive logging, and error handling.

## 📝 Problem Statement

The goal of this project is to understand and predict how student performance (math scores) is affected by various factors including:
- **Gender**: Male/Female
- **Race/Ethnicity**: Group A, B, C, D, E
- **Parental Level of Education**: Various education levels
- **Lunch Type**: Standard/Free or Reduced
- **Test Preparation Course**: Completed/None
- **Reading Score**: Numerical feature
- **Writing Score**: Numerical feature

**Target Variable**: `math_score` (predicted value)

## 📊 Dataset

- **Source**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)
- **Size**: 1000 rows × 8 columns
- **Location**: `notebook/data/stud.csv`

### Dataset Features

**Categorical Features:**
- `gender`: Student's gender
- `race_ethnicity`: Student's race/ethnicity group
- `parental_level_of_education`: Parent's education level
- `lunch`: Type of lunch (standard/free or reduced)
- `test_preparation_course`: Whether test prep course was completed

**Numerical Features:**
- `reading_score`: Student's reading score
- `writing_score`: Student's writing score
- `math_score`: Student's math score (target variable)

## 📁 Project Structure

```
mlproject/
├── artifacts/                 # Generated files (models, preprocessors, data splits)
│   ├── preprocessor.pkl      # Saved preprocessing pipeline
│   ├── model.pkl             # Trained model
│   ├── train.csv             # Training dataset
│   ├── test.csv              # Testing dataset
│   └── raw.csv               # Raw dataset
├── logs/                      # Log files with timestamps
├── notebook/                  # Jupyter notebooks for EDA and model training
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   ├── 2. MODEL TRAINING.ipynb
│   └── data/
│       └── stud.csv
├── src/                       # Source code
│   ├── components/            # Core ML components
│   │   ├── data_ingestion.py      # Data loading and train-test split
│   │   ├── data_transformation.py # Data preprocessing pipeline
│   │   └── model_trainer.py       # Model training and evaluation
│   ├── pipeline/             # Prediction and training pipelines
│   │   ├── train_pipeline.py # End-to-end training pipeline
│   │   └── pred_pipeline.py  # Prediction pipeline
│   ├── exception.py          # Custom exception handling
│   ├── logger.py             # Logging configuration
│   └── utils.py              # Utility functions
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup configuration
└── README.md                # Project documentation
```

## 🔧 Technologies Used

- **Python 3.x**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **CatBoost**: Gradient boosting framework
- **XGBoost**: Extreme gradient boosting
- **Flask**: Web framework (for deployment)
- **Dill**: Object serialization
- **Matplotlib & Seaborn**: Data visualization (in notebooks)

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd mlproject
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

## 💻 Usage

### Running the Data Ingestion Pipeline

```bash
python src/components/data_ingestion.py
```

This will:
- Load the raw data from `notebook/data/stud.csv`
- Split the data into train and test sets (80/20 split)
- Save the processed data to `artifacts/` directory

### Running the Data Transformation Pipeline

The data transformation is automatically triggered when running data ingestion. It will:
- Create preprocessing pipelines for numerical and categorical features
- Apply transformations to training and testing data
- Save the preprocessor object to `artifacts/preprocessor.pkl`

### Training the Model

```bash
python src/pipeline/train_pipeline.py
```

This will execute the complete training pipeline:
1. Data ingestion
2. Data transformation
3. Model training and evaluation
4. Model saving

### Making Predictions

```bash
python src/pipeline/pred_pipeline.py
```

## 🔄 Project Workflow

```
Raw Data (stud.csv)
    ↓
Data Ingestion
    ├── Load data
    ├── Train-test split (80/20)
    └── Save to artifacts/
    ↓
Data Transformation
    ├── Numerical Pipeline
    │   ├── Imputation (median)
    │   └── Standard Scaling
    ├── Categorical Pipeline
    │   ├── Imputation (most_frequent)
    │   ├── One-Hot Encoding
    │   └── Standard Scaling
    └── Save preprocessor
    ↓
Model Training
    ├── Train multiple models
    ├── Evaluate performance
    ├── Select best model
    └── Save model
    ↓
Prediction Pipeline
    └── Load model & preprocessor
    └── Make predictions
```

## 🧩 Components

### 1. Data Ingestion (`data_ingestion.py`)

- **Class**: `DataIngestion`
- **Purpose**: Load raw data and split into train/test sets
- **Output**: Train and test CSV files in `artifacts/` directory

### 2. Data Transformation (`data_transformation.py`)

- **Class**: `DataTransformation`
- **Purpose**: Create and apply preprocessing pipelines
- **Features**:
  - Numerical features: Median imputation + Standard scaling
  - Categorical features: Most frequent imputation + One-hot encoding + Scaling
- **Output**: Preprocessed arrays and saved preprocessor object

### 3. Model Trainer (`model_trainer.py`)

- **Class**: `ModelTrainer`
- **Purpose**: Train, evaluate, and select the best model
- **Models**: Multiple algorithms including CatBoost, XGBoost, and others
- **Output**: Trained model saved as `artifacts/model.pkl`

### 4. Logging (`logger.py`)

- **Purpose**: Configure logging for the entire project
- **Features**:
  - Timestamped log files in `logs/` directory
  - Detailed logging with line numbers and timestamps
  - INFO level logging by default

### 5. Exception Handling (`exception.py`)

- **Class**: `CustomException`
- **Purpose**: Custom exception handling with detailed error messages
- **Features**: Includes file name, line number, and error message

### 6. Utilities (`utils.py`)

- **Function**: `save_object()`
- **Purpose**: Save Python objects (models, preprocessors) using dill
- **Usage**: Serializes objects to `.pkl` files

## 📈 Model Training

The project supports training multiple machine learning models:
- **CatBoost**: Gradient boosting with categorical features support
- **XGBoost**: Extreme gradient boosting
- **Other algorithms**: As configured in the model trainer

The best model is selected based on evaluation metrics (typically R² score or RMSE) and saved for production use.

## 📝 Logging

All operations are logged with timestamps:
- Log files are stored in `logs/` directory
- Format: `DD_MM_YYYY_HH_MM_SS.log`
- Includes: Timestamp, line number, module name, log level, and message

## ⚠️ Error Handling

The project uses custom exception handling:
- All exceptions are caught and wrapped in `CustomException`
- Error messages include:
  - Python script name
  - Line number where error occurred
  - Detailed error message

## 👤 Author

**Varadaraj**
- Email: varadaraj.kamisetty@gmail.com

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset source: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)
- Built following MLOps best practices

## 📚 Additional Notes

- The `setup.py` file allows the project to be installed as a package, making it reusable across different environments
- All artifacts (models, preprocessors, data splits) are saved in the `artifacts/` directory
- The project follows a modular structure for easy maintenance and scalability

---

**Note**: Make sure to update the data path in `data_ingestion.py` if your dataset location differs from the default path.
