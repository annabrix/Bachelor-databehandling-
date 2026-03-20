#%%
from itertools import count
from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
from requests import patch
import statsmodels.api as sm
from IPython.display import Markdown as md
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd

#indlæser csv filerne og sætter ind.tid som index, da det er det der skal analyseres på
file_PE = os.path.join(os.getcwd(), 'Alle RA 2025 pr. 2026-03-11 Q1-Q2.csv')
file_PE2 = os.path.join(os.getcwd(), 'Alle RA 2025 pr. 2026-03-11 Q3-Q4.csv')
file_fuel = os.path.join(os.getcwd(), 'fuel gmk.csv')

df_data1 = pd.read_csv(file_PE, sep=';', encoding='latin1')
#print(len(df_data1))

df_data2 = pd.read_csv(file_PE2, sep=';', encoding='latin1')
df_fuel = pd.read_csv(file_fuel, sep=',')

# samler alle tre datasæt, så vi kigger på et helt år samt kigger 
df_data = pd.concat([df_data1, df_data2], ignore_index=True)
#print(df_data)

# fjerner eventuelle mellemrum i kolonnenavnene, da det kan give problemer senere
df_data.columns = df_data.columns.str.strip()
df_fuel.columns = df_fuel.columns.str.strip()

# ind.tid har nogle gange 24:00, hvilket ikke er en gyldig tid, så det erstattes med 00:00 og tilføjer en dag
dt = df_data["ind.tid"].astype(str).str.strip()
dt_ud = df_data["ud.tid"].astype(str).str.strip()
#print(dt)
mask_ind = dt.str.endswith("24:00")
mask_ud = dt_ud.str.endswith("24:00")

dt_fuel = df_fuel["Transaction Date/Time"].astype(str).str.strip()

dt = dt.str.replace("24:00", "00:00", regex=False)
dt_ud = dt_ud.str.replace("24:00", "00:00", regex=False)

dt = pd.to_datetime(dt, format="%d-%m-%Y %H:%M", dayfirst=True, errors="coerce")
dt_ud = pd.to_datetime(dt_ud, format="%d-%m-%Y %H:%M", dayfirst=True, errors="coerce")


dt.loc[mask_ind] = dt.loc[mask_ind] + pd.Timedelta(days=1)
dt_ud.loc[mask_ud] = dt_ud.loc[mask_ud] + pd.Timedelta(days=1)

print("NaT i ind.tid:", dt.isna().sum())
print("NaT i ud.tid:", dt_ud.isna().sum())


#indsæter nye ind.tid og ud.tid og fjerner alle rækker hvor de er NaN
df_data["ind.tid"] = dt
df_data["ud.tid"] = dt_ud

df_data = df_data.dropna(subset=["ind.tid"])
df_data = df_data.dropna(subset=["ud.tid"])

print("Dropna", len(df_data))

#Sætter ind.tid som index, da det er det der skal analyseres på først
#df_data.set_index("ind.tid", inplace=True)



#fjerner alle rækker som ikke er status 4 da status 4 er afsluttede udlejninger
mask_df_data = df_data["stat"].astype(str).str.contains("4", na=False)
df_data = df_data.loc[mask_df_data]
print("only status 4:", len(df_data))


#Finder de biler der kommer ind på gammel kongevej
#print(df_data["st.i"].unique())
df_data["st.i"] = pd.to_numeric(df_data["st.i"], errors="coerce")
df_gmk = df_data[df_data["st.i"] == 5]

print("Gmk specific", len(df_gmk))

df_gmk = df_gmk.drop(columns=[
    "kon.nr", "spcgrp", "spcnr", "k/f", "st.u", "st.i", "stat", "leje.dg", "oprettelse",
    "udl.land", "lejer", "firmabss", "firma", "land", "mærke", "model", "km.incl", "styr.rate", "styr.ratekode",
    "rate2", "rate2-dkk", "rate3", "rate3-dkk", "rate4", "rate4-dkk", "rate5", "rate5-dkk", "rate6", "rate6-dkk",
    "rate7", "rate7-dkk", "rate8", "rate8-dkk", "rate9", "rate9-dkk", "rate10", "rate10-dkk", "extrakm-dkk", "moms",
    "forsikring", "total", "dekort", "check-out", "exp-check-in", "check-in"
])

#Nu inkorperes fueldataen:

df_fuel["Transaction Date/Time"] = pd.to_datetime(
    df_fuel["Transaction Date/Time"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)


#laver ny kolonne som har samme datetime format som de andre dataframes
df_fuel["Transaction Date/Time_str"] = df_fuel["Transaction Date/Time"].dt.strftime("%d-%m-%Y %H:%M:%S")

#Investigating fuel dataframe
print(df_fuel.head())
print(df_fuel.columns)
for col in df_fuel.columns:
    print(f"'{col}'")
print(df_fuel["Transaction Date/Time_str"])

#%%
# Merging a new dataframe where fueldata and incoming cars are matched 

# Clearing Vehicle number columns
df_gmk["reg.nr"] = df_gmk["reg.nr"].astype(str).str.strip().str.upper()
df_fuel["Vehicle Number"] = df_fuel["Vehicle Number"].astype(str).str.strip().str.upper()

# lav datokolonner
df_gmk["Date1"] = df_gmk["ind.tid"].dt.date
df_fuel["Date2"] = df_fuel["Transaction Date/Time"].dt.date

df_gmk_small = df_gmk[["Date1", "reg.nr", "ind.tid", "bilgrp"]]

df_fuel_small = df_fuel[["Date2", "Transaction Date/Time_str", "Vehicle Number", "Product", "Volume", "Customer Price", "Total Price"]]

df_new = df_gmk_small.merge(
    df_fuel_small,
    left_on=["Date1", "reg.nr"],
    right_on=["Date2", "Vehicle Number"],
    how="inner"
)
print(df_new.head())
print(len(df_new))
# %%
#Now we can analyze on how many from specific groups of cars are fueled pr day
print("Before merge:")
print(df_gmk_small["bilgrp"].value_counts(dropna=False))

print("\nAfter merge:")
print(df_new["bilgrp"].value_counts(dropna=False))

# %%
