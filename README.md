Data Collection, Cleaning, and Preprocessing for Logistics Analysis

Overview

This beginner-friendly project demonstrates a Python preprocessing pipeline for logistics analysis using the public Brazilian E-Commerce Public Dataset by Olist.

Objectives

- Inspect dataset structure and missingness.
- Parse delivery timestamps and identify duplicates.
- Calculate delivery lead time and late-delivery status.
- Flag negative durations and IQR-based outliers for investigation.
- Aggregate order-item data before merging to order-level records.
- Demonstrate Min-Max normalization and standardization.
- Export a cleaned order-level CSV.

Dataset

Download the dataset from Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Place the original files in a folder named data . 
At minimum, this script uses:
`olist_orders_dataset.csv`
`olist_order_items_dataset.csv`

Do not commit the raw dataset to GitHub unless its terms and repository size make that appropriate. Link to the public source instead.

Requirements

pip install pandas numpy scikit-learn

Run

From the repository root:

python preprocess_logistics.py

The script prints basic quality checks and creates `cleaned_logistics_orders.csv`.

Important notes

- The script is a template; check filenames and outputs against the version you download.
- Missing delivery timestamps are not replaced with averages because that would invent event times.
- Outliers are flagged rather than automatically removed; extreme delivery times may be genuine operational issues.
- For predictive modelling, fit scalers on training data only to avoid data leakage.
- Add actual run outputs and any changes you make before presenting this as a completed analysis.

Project report

See `Logistics_Data_Preprocessing_Report.docx` for the methodology, code explanations, limitations, and conclusions.

References

- Olist dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Pandas: https://pandas.pydata.org/docs/
- scikit-learn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
