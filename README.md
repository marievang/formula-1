# Formula 1 Race Podium Prediction

> A comprehensive machine learning pipeline to predict F1 Race Winners and Podium finishes using historical telemetry, driver error rates, and qualifying data.

This project aggregates historical F1 data (from 1950 to present) to build predictive models. It features custom features as calculating "Driver Error Rates" and handling historic team name changes (e.g., Toro Rosso → AlphaTauri → RB)—and compares **Random Forest** vs. **Logistic Regression** performance.

## Project Overview

The goal is to predict race outcomes based on pre-race factors like Qualifying position (`q3`), historical performance (`standings`), and driver reliability (`error_rates`).

### Features
* **Custom Feature:**
    * **Driver/Constructor Error Rates:** Calculates the probability of a driver crashing or having a mechanical failure based on historical `statusId` data.
    * **Home Advantage:** Flags if a driver is racing in their home country.
    * **Legacy Team Mapping:** Merges historical team IDs (e.g., Force India $\rightarrow$ Aston Martin) to maintain data continuity.
* **Dimensionality Reduction:** Uses **PCA (Principal Component Analysis)** to analyze feature variance.
* **Model Comparison:** Benchmarks **Random Forest** (with Hyperparameter Tuning) against **Logistic Regression** (CV).


The project is divided into Data Preparation, Feature Engineering, and Analysis.

### 1. Data Preparation & Cleaning
* **`create_training_dataset.py`:**
    * Merges raw CSVs (`results`, `drivers`, `constructors`, `qualifying`, etc.).
    * Standardizes Qualifying times (converts `1:20.142` to seconds).
    * Creates the master training file: `current_driver_dataset.csv`.
* **`fix_dataset.py`:**
    * Cleans specific datasets (`maria.tsv`) and merges them with the latest race results.
    * Generates the target variables: `podium.csv` (Top 3) and `winner.csv` (P1).

### 2. Feature Engineering
* **`calc_driver_error.py`:**
    * Analyzes `status.csv` to identify DNF causes (Accidents, Collisions, etc.).
    * Computes and plots error rates for current drivers (e.g., VER, HAM, LEC).
    * **Output:** `driver_error_rates.csv`, `constructor_error_rates.csv`.

### 3. Analysis & Modeling
* **`pca_new.py`:**
    * Standardizes data and runs **PCA** to visualize explained variance.
    * Generates Heatmaps and Feature Importance bar charts to understand which variables drive performance.
* **`Analysis.py`:**
    * **The Core Logic:** Trains models to predict **Winners** and **Podiums**.
    * **Random Forest:** Uses `RandomizedSearchCV` for optimization and Permutation Importance for feature selection.
    * **Logistic Regression:** Uses `LogisticRegressionCV` and calculates Odds Ratios.

##  Prerequisites

You will need Python 3 and the standard data science stack:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
