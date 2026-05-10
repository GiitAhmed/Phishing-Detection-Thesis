# Phishing Email Detection System 

**Author:** Ahmed Ibrahim  
**Project Type:** Bachelor's Thesis (Information Technology Engineering)  
**Institution:** South-Eastern Finland University of Applied Sciences (Xamk)  
**Year:** 2026  

## 📌 Project Overview
This repository contains the machine learning pipeline developed for the thesis *"Design and Implementation of a Machine Learning-Based Phishing Email Detection System."* The primary objective of this project is to provide a lightweight, open-source cybersecurity solution tailored for Small and Medium-sized Enterprises (SMEs). The system utilizes streamlined text preprocessing and TF-IDF vectorization to detect phishing emails without requiring the massive computational resources or expensive hardware typically associated with deep learning models.

The project evaluates three standard classification algorithms:
* **Random Forest** (Ensemble Method)
* **Naive Bayes** (Probabilistic Method)
* **Logistic Regression** (Linear Method)

## 📊 Dataset
This project utilizes the publicly available **Phishing Email Dataset** compiled by Naser Abdullah Alam. The raw dataset contains over 82,000 emails, which are reduced to 39,154 highly labeled samples during the script's cleaning phase.

* **Dataset Link:** [Kaggle: Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
* **Preparation:** Download the dataset from Kaggle, extract the `.csv` file, and ensure it is named `Phishing_Email.csv` and placed in the root directory of this project before running the scripts.

## 🛠️ Installation & Prerequisites
This pipeline was built and tested using **Python 3.10**. 

To replicate the environment and run the scripts, install the required mathematical and data processing libraries via your terminal:

```bash
pip install pandas scikit-learn matplotlib
```

🚀 Two-Phase Methodology (How to Run)

To evaluate the models' viability for SMEs, the experiment is divided into two distinct phases. You can run each phase independently using the provided scripts.
Phase 1: The Baseline Test (main_baseline.py)

This script trains the models on 80% of the dataset and tests them on the remaining 20%. It establishes the maximum theoretical accuracy of the algorithms under ideal, data-rich conditions.

    To execute: python main_baseline.py

    Key Finding: Random Forest achieves the optimal balance with an overall accuracy of 99.17% and a precision of 99.24%.

Phase 2: The SME Stress Test (main_stresstest.py)

Because most SMEs do not possess tens of thousands of historical emails, this script stress-tests the algorithms by restricting the training data to just 2% (~783 emails).

    To execute: python main_stresstest.py

    Key Finding: The linear models demonstrate remarkable resilience. Even starved of data, Logistic Regression maintains the highest overall accuracy (97.17%), demonstrating that baseline phishing security can be achieved with minimal historical data.

📈 Evaluation Metrics

The scripts output a detailed terminal table measuring:

    Accuracy: Overall correct predictions.

    Precision: The percentage of flagged emails that were truly malicious (critical for avoiding False Positives).

    Recall (Sensitivity): The percentage of all actual threats successfully intercepted.

Upon completion, both scripts will automatically generate and display a Matplotlib bar graph illustrating the comparative accuracy percentages of the three models.

📄 License and Usage

This code is provided as an open-source academic supplement. Researchers and IT professionals are welcome to review, test, and adapt the methodology for their own lightweight cybersecurity implementations.



