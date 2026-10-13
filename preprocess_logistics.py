"""
Data Collection, Cleaning, and Preprocessing for Logistics Analysis
Dataset: Brazilian E-Commerce Public Dataset by Olist
Run from project root after placing the original CSVs in data/.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA = "data/"
orders = pd.read_csv(DATA + "olist_orders_dataset.csv")
items = pd.read_csv(DATA + "olist_order_items_dataset.csv")

date_cols = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

print("Initial orders shape:", orders.shape)
print("Missing values:\n", orders.isna().sum().sort_values(ascending=False))
print("Duplicate order IDs:", orders.duplicated(subset=["order_id"]).sum())

# Remove only exact repeated order records after checking the source.
orders = orders.drop_duplicates(subset=["order_id"], keep="first")

delivered = orders.loc[orders["order_status"].eq("delivered")].copy()
delivered = delivered.dropna(
    subset=["order_purchase_timestamp", "order_delivered_customer_date"]
)
delivered["delivery_lead_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400

negative = delivered["delivery_lead_days"] < 0
print("Negative delivery durations flagged:", int(negative.sum()))
delivered = delivered.loc[~negative].copy()

delivered["late_delivery_flag"] = (
    delivered["order_delivered_customer_date"]
    > delivered["order_estimated_delivery_date"]
)

q1 = delivered["delivery_lead_days"].quantile(0.25)
q3 = delivered["delivery_lead_days"].quantile(0.75)
iqr = q3 - q1
lower = max(0, q1 - 1.5 * iqr)
upper = q3 + 1.5 * iqr
delivered["delivery_time_outlier"] = (
    (delivered["delivery_lead_days"] < lower)
    | (delivered["delivery_lead_days"] > upper)
)

items["price"] = pd.to_numeric(items["price"], errors="coerce")
items["freight_value"] = pd.to_numeric(items["freight_value"], errors="coerce")
item_summary = items.groupby("order_id", as_index=False).agg(
    item_count=("order_item_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
)

analysis_df = delivered.merge(
    item_summary, on="order_id", how="left", validate="one_to_one"
)

# Demonstration of scaling. For modelling, fit scalers on training data only.
scale_rows = analysis_df["delivery_lead_days"].notna()
minmax = MinMaxScaler()
analysis_df.loc[scale_rows, "delivery_days_minmax"] = minmax.fit_transform(
    analysis_df.loc[scale_rows, ["delivery_lead_days"]]
).ravel()
standard = StandardScaler()
analysis_df.loc[scale_rows, "delivery_days_standard"] = standard.fit_transform(
    analysis_df.loc[scale_rows, ["delivery_lead_days"]]
).ravel()

print("Cleaned analysis shape:", analysis_df.shape)
print("Unique orders:", analysis_df["order_id"].nunique())
print("Late delivery rate (%):", round(analysis_df["late_delivery_flag"].mean() * 100, 2))
analysis_df.to_csv("cleaned_logistics_orders.csv", index=False)
print("Saved cleaned_logistics_orders.csv")
