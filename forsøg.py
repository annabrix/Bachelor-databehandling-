import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.display import Markdown as md
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd

file_PE = os.path.join(os.getcwd(), 'Alle RA 2025 pr. 2026-03-11 Q1-Q2.csv')


# Read CSV with a forgiving encoding
df_data = pd.read_csv(file_PE, sep=';', encoding='latin1')
df_data.columns = df_data.columns.str.strip()

raw_dt = df_data["ind.tid"].astype(str).str.strip()

#Marker rækker med 24:00"
mask_24 = raw_dt.str.endswith("24:00", na=False)

#Erstat 24:00 med 00:00 midlertidigt
dt_clean = raw_dt.str.replace("24:00", "00:00", regex=False)

# Parse robust, men uden at crashe
parsed_dt = pd.to_datetime(
    dt_clean,
    format="%d-%m-%Y %H:%M",
    dayfirst=True,
    errors="coerce"
)

# Læg 1 dag til de rækker som oprindeligt havde 24:00
parsed_dt.loc[mask_24] = parsed_dt.loc[mask_24] + pd.Timedelta(days=1)

# Tjek hvor mange der fejler
bad_mask = parsed_dt.isna()
print("Antal dårlige timestamps:", bad_mask.sum())

# Vis de første 20 dårlige originale værdier
print("\nFørste 20 dårlige timestamps:")
print(raw_dt.loc[bad_mask].head(20).tolist())

# Vis hele de problematiske rækker
bad_rows = df_data.loc[bad_mask].copy()
bad_rows["original_ind.tid"] = raw_dt.loc[bad_mask]

print("\nProblematiske rækker:")
print(bad_rows.head(20))

# Først når du er tilfreds, gem den parsed version
df_data["ind.tid"] = parsed_dt

# Sæt kun index hvis du vil fortsætte med de gyldige rækker
# df_data = df_data.dropna(subset=["ind.tid"])
# df_data.set_index("ind.tid", inplace=True)











# dt = df_data["ind.tid"].astype(str).str.strip()
# mask = dt.str.endswith("24:00")

# dt = dt.str.replace("24:00", "00:00", regex=False)
# dt = pd.to_datetime(dt, format="%d-%m-%Y %H:%M", dayfirst=True)
# dt.loc[mask] = dt.loc[mask] + pd.Timedelta(days=1)

# df_data["ind.tid"] = dt
# df_data.set_index("ind.tid", inplace=True)

# print(df_data.index.astype(str)[:20].tolist())






















df_data["km"] = pd.to_numeric(df_data["km"], errors="coerce").fillna(0)
df_data["extrakm"] = pd.to_numeric(df_data["extrakm"], errors="coerce").fillna(0)

df_data["sum_km"] = df_data["km"].values + df_data["extrakm"].values

df_data = df_data.drop(columns=["km", "extrakm", "kon.nr", "bilgrp", "spcnr", "st.u", "oprettelse", "udl.land", "lejer", "firmabss", "land", "firma", "model", "km.incl", "extrakm-dkk", "forsikring", "dekort", "exp-check-in", "check-in", "check-out", "moms", "total"])

#print(df_data["sum_km"])
print("hej!")
bad_times = df_data.index[df_data.index.isna()]
print(bad_times)

cols_932 = []
cols_932 = [col for col in df_data.columns
            if df_data[col].astype(str).str.contains("932-EV CHARGE CARS", na=False).any()]

mask_932 = df_data[cols_932].astype(str).apply(
    lambda col: col.str.contains("932-EV CHARGE CARS", na=False)
).any(axis=1)

df_932_full = df_data.loc[mask_932]


#print("hej")
#print(df_932_full.head())
#print("hej")
