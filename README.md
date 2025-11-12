# Distance to Export — Replication Code

This repository contains the code for **“Distance to Export: A Machine Learning Approach with Portuguese Firms”** (Banco de Portugal Working Paper W202420) by  **Paulo Barbosa**, João Amador, and João Cortes. It includes scripts and notebooks to estimate firms’ “distance to export” and reproduce the paper’s main tables and figures.

## Repository structure
- `Logit_Logit-lasso_Bart_Bart-mia_Random-forest.R` — Fits Logit, Logit-LASSO, Random Forest, BART, and BART-MIA; exports metrics and plots.  
- `Neural Network/` — Notebook(s) for the neural-network model; writes predictions and diagnostics.  
- `predictions_and_plots/` — Saved predictions, figures, and tables used in the paper.

## Method (one-paragraph overview)
We predict the probability that a firm becomes a **successful exporter** using rich firm-level data and a suite of ML models (Logit/penalized Logit, Random Forest, BART/BART-MIA, and a Neural Network). The paper compares out-of-sample performance across models and documents the importance of variables such as labour productivity, imports, capital intensity, wages, size, and age.

## Data
The original analysis relies on confidential Portuguese firm-level microdata and **cannot be redistributed**.  
To run the code, provide an equivalent firm-year panel with:
- **Identifiers:** firm ID, year  
- **Outcome:** exporter status (as defined in the paper)  
- **Features:** the variable set described in the manuscript (e.g., productivity, imports, capital intensity, wages, size, age, etc.)

Point the scripts/notebooks to your dataset using the input paths at the top of each file. You can test the pipeline with mock data in the same structure.

## Requirements
- **R (≥ 4.x)** with packages for wrangling/plotting and modeling (e.g., `tidyverse`, `glmnet`, `randomForest` or `ranger`, BART libraries).  
- **Python (≥ 3.x)** for the neural-network notebook (TensorFlow/Keras or PyTorch; plus `pandas`, `numpy`, `scikit-learn`).  

Set seeds are used where applicable; minor deviations may occur due to algorithmic randomness and library versions.

## Quick start
1. Clone the repository:
   ```bash
   git clone https://github.com/paulofbarbosa5/distance-to-export-estimations.git
   cd distance-to-export-estimations
