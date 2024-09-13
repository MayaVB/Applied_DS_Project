# Unveiling startups Success predictors: Harnessing ML for predicting startup acquisitions

## Research Paper

Our comprehensive research paper detailing the methods, findings, and insights of this project is available [here](https://medium.com/@natalyasegal/unveiling-startups-success-predictors-harnessing-ml-for-predicting-startup-acquisitions-f879ed372deb).

## Introduction

There are 150 million startups in the world today, and 50 million new startups are launching every year. As of 2024, early-stage venture funding totaled around $29.5 billion, up 6% year over year (based on Crunchbase). Still, less than 10% of venture-backed startups make it to either an IPO or an acquisition. This positions startup investment as a high-risk, high-yield category, forcing high pressure on the founders as they are expected to have at least 3x returns to make up for those that fail. The industry will greatly benefit from improving an early prediction of the startup’s fate and, in this way, reducing the investment risk.

## Data Description

The [dataset](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction/data) contains industry trends, investment insights, and individual company information. We have added 30+ more parameters from macroeconomic factors that significantly improve the accuracy of our model.

## Key Objectives

1. Forecast the probability of a startup transforming into a successful company: We do so by preprocessing the data, exploring visualizations, comparing several classification models and implementing several regression models. 
2. Transparent and Reproducible Research: We provide a clear and detailed explanation of our methodology to ensure transparency and reproducibility in the research community.
3. Future Directions: We discuss potential options for further development of this work.

## Repository Structure

The repository is organized as follows:

```
📁 Applied_DS_Project
   |
   ├── 📄 README.md                         # This file, providing an overview of the project
   |
   ├── 📄 requirements.txt                  # This file contains a list of packages or libraries needed to work on the project
   |
   📁 data
   |
   ├── 📄 startup_data.csv                  # This file contains the dataset in csv format.
   |
   📁 notebooks
   |
   ├── 📄 main_final.ipynb                        # Main notebook containing the research findings and results
   |
   └── 📄 EDA_final.ipynb                         # Exploratory data analysis notebook containing visualizations of the dataset.
   |
   📁 src
   |
   ├── 📄 eval.py                           # This file contains predictions evaluations (some of the metrics: AUC, precision, recall)
   |
   ├── 📄 getdata.py                        # Get macroeconomic factors (GDP, UEM, Nasdaq annual changes) 
   |
   ├── 📄 main.py                           # Main file containing the data loading, processing and analyzing.
   |
   ├── 📄 models.py                         # Contains the classification models we used.
   |
   ├── 📄 preprocess.py                     # Contains the data preprocessing (like outliers handling, features normalization, PCA, etc)
   |
   ├── 📄 printstatistics.py                # This file prints correlations.
   |
   ├── 📄 regression_models.py              # In this file we train and evaluate regression models.  
   |
   ├── 📄 utils.py                          # utility file
   |
   └──
```

## Getting Started

To replicate our research or explore the project, follow these steps:

1. Start by cloning this repository, from a colab you can do it by adding a cell with the following command:
!git clone https://github.com/MayaVB/Applied_DS_Project.git

2. Make sure you install all packages and libraries as mentioned in requirements.txt

3. Open main file to access the research results.

4. Follow the instructions in the notebook to reproduce our findings and explore the research in detail.

## Conclusion

In this project we saw that adding macroeconomic factors to the original dataset affected the classification performance. We have also seen that there are startup industries in our classifier that perform better and those that perform much worse. We suggest proceeding with additional data augmentation, this time using industry-related economic indices.


