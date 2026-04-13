# mlops_lab_exam
Lab exam final repo

This repository contains a machine learning pipeline for training a regression model on the wine quality dataset.

## Features
- Trains a Random Forest regression model on wine quality data
- Calculates MSE and R² metrics
- Saves trained model as `model.pkl`
- Exports metrics to `metrics.json`
- Automated CI/CD pipeline using GitHub Actions

## Local Training

To train the model locally:

```bash
pip install pandas numpy scikit-learn
python train.py
```

## GitHub Actions Workflow

The ML pipeline is automatically triggered on push to the main branch:

1. **Train Job**: Installs dependencies and trains the model
2. **Report Job**: Downloads artifacts and prints completion status

Check the Actions tab in GitHub to see workflow runs.

## Author
Surya Sanjan - 2022BCS0194
