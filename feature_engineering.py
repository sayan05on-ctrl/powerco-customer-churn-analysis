import pandas as pd
import numpy as np

# ==========================================
# Load Data
# ==========================================

df = pd.read_csv("clean_data_after_eda.csv")

print("Original Shape:", df.shape)

# ==========================================
# Convert Date Columns
# ==========================================

date_cols = [
    "date_activ",
    "date_end",
    "date_modif_prod",
    "date_renewal"
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# ==========================================
# Feature Engineering
# ==========================================

df["avg_yearly_price_change"] = (
    df["var_year_price_off_peak"] +
    df["var_year_price_peak"] +
    df["var_year_price_mid_peak"]
) / 3

df["avg_6m_price_change"] = (
    df["var_6m_price_off_peak"] +
    df["var_6m_price_peak"] +
    df["var_6m_price_mid_peak"]
) / 3

df["price_sensitivity"] = (
    df["avg_yearly_price_change"] /
    (df["forecast_cons_12m"] + 1)
)

if all(col in df.columns for col in ["date_end", "date_activ"]):
    df["customer_tenure_days"] = (
        df["date_end"] - df["date_activ"]
    ).dt.days

if all(col in df.columns for col in ["date_end", "date_modif_prod"]):
    df["days_since_modification"] = (
        df["date_end"] - df["date_modif_prod"]
    ).dt.days

if all(col in df.columns for col in ["date_renewal", "date_activ"]):
    df["days_until_renewal"] = (
        df["date_renewal"] - df["date_activ"]
    ).dt.days

if all(col in df.columns for col in ["cons_last_month", "cons_12m"]):
    df["consumption_ratio"] = (
        df["cons_last_month"] /
        (df["cons_12m"] + 1)
    )

df["profitability_ratio"] = (
    df["net_margin"] /
    (df["forecast_cons_12m"] + 1)
)

print("Feature Engineering Completed")
print(df.head())

df.to_csv("clean_data_feature_engineered.csv", index=False)
print("Saved: clean_data_feature_engineered.csv")
