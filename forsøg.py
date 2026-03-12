#%%
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
file_PE2 = os.path.join(os.getcwd(), 'Alle RA 2025 pr. 2026-03-11 Q3-Q4.csv')

# Read CSV with a forgiving encoding
df_data1 = pd.read_csv(file_PE, sep=';', encoding='latin1')
df_data2 = pd.read_csv(file_PE2, sep=';', encoding='latin1')

# Concatenate the two DataFrames
df_data = pd.concat([df_data1, df_data2], ignore_index=True)

df_data.columns = df_data.columns.str.strip()



dt = df_data["ind.tid"].astype(str).str.strip()
mask = dt.str.endswith("24:00")

dt = dt.str.replace("24:00", "00:00", regex=False)
dt = pd.to_datetime(dt, format="%d-%m-%Y %H:%M", dayfirst=True)
dt.loc[mask] = dt.loc[mask] + pd.Timedelta(days=1)

#fjerne alle rækker hvor ind.tid er NaN
df_data["ind.tid"] = dt
df_data.set_index("ind.tid", inplace=True)
df_data = df_data[~df_data.index.isna()]
#print(df_data.index)

#%%
df_data["km"] = pd.to_numeric(df_data["km"], errors="coerce").fillna(0)
df_data["extrakm"] = pd.to_numeric(df_data["extrakm"], errors="coerce").fillna(0)

df_data["sum_km"] = df_data["km"].values + df_data["extrakm"].values

df_data = df_data.drop(columns=["km", "extrakm", "kon.nr", "spcgrp", "spcnr", "st.u", "oprettelse", "udl.land", "lejer", "firmabss", "land", "firma", "model", "km.incl", "extrakm-dkk", "forsikring", "dekort", "exp-check-in", "check-in", "check-out", "moms", "total"])


#%%
#her finder jeg kun status 4
mask_s6 = df_data["stat"].astype(str).str.contains("4", na=False)
df_data = df_data[mask_s6]
#print("data_s4", df_data) #65003 rækker

# her finder jeg elbilerne samt dem der har betalt for ikke at være opladt
mask_EV = df_data["bilgrp"].astype(str).str.match(r"^.E")
df_EV = df_data[mask_EV]
#print("EV", df_EV) #her er 1402 rækker

cols_932 = []
cols_932 = [col for col in df_data.columns
            if df_data[col].astype(str).str.contains("932-EV CHARGE CARS", na=False).any()]

mask_932 = df_data[cols_932].astype(str).apply(
    lambda col: col.str.contains("932-EV CHARGE CARS", na=False)
).any(axis=1)

# finding the rows that contain "939-FULL TANK OPTION" in any column of df_EV
cols_939 = []
cols_939 = [col for col in df_EV.columns
            if df_EV[col].astype(str).str.contains("939-FULL TANK OPTION", na=False).any()]

mask_939 = df_EV[cols_939].astype(str).apply(
    lambda col: col.str.contains("939-FULL TANK OPTION", na=False)
).any(axis=1)

df_932_notfull = df_data.loc[mask_932]
df_939 = df_EV.loc[mask_939]

df_notfull = pd.concat([df_932_notfull, df_939])
df_notfull = df_notfull.sort_index()
print("rate 939+935", df_notfull) #her er 309 rækker


#Finder de biler der kommer ind på gammel kongevej
mask_gmk = df_data["st.i"].astype(str).str.fullmatch("5.0", na=False)
df_gmk = df_data[mask_gmk]

#print("gmk", df_gmk) 

# %%
#hvornår ankommer bilerne på gammel kongevej?
df_gmk["hour"] = df_gmk.index.hour
hour_counts = df_gmk["hour"].value_counts().sort_index()

print(hour_counts)

hour_distribution = hour_counts / hour_counts.sum()

print(hour_distribution)
hour_counts.plot(kind="bar")
plt.xlabel("Hour of day")
plt.ylabel("Number of arrivals")
plt.title("Distribution of car arrivals by hour")
plt.show()

# %%
# hvornår ankommer elbiler og hvornår ankommer dem som ikke er ladet op?
df_EV["hour"] = df_EV.index.hour
hour_counts_EV = df_EV["hour"].value_counts().sort_index()
df_notfull["hour"] = df_notfull.index.hour
hour_counts_NF = df_notfull["hour"].value_counts().sort_index()

# sørg for alle 24 timer findes
hour_counts_EV = hour_counts_EV.reindex(range(24), fill_value=0)
hour_counts_NF = hour_counts_NF.reindex(range(24), fill_value=0)

# lav én dataframe
hour_compare = pd.DataFrame({
    "EV": hour_counts_EV,
    "Not full": hour_counts_NF
})

# plot
hour_compare.plot(kind="bar")

plt.xlabel("Hour of day")
plt.ylabel("Number of arrivals")
plt.title("Distribution of car arrivals by hour")
plt.show()

# hour_distribution_EV = hour_counts_EV / hour_counts_EV.sum()
# hour_distribution_NF = hour_counts_NF / hour_counts_NF.sum()

# print( hour_distribution_EV, hour_distribution_NF)

# hour_counts_EV.plot(kind="bar")
# hour_counts_NF.plot(kind="bar")
# plt.xlabel("Hour of day")
# plt.ylabel("Number of arrivals")
# plt.title("Distribution of car arrivals by hour")
# plt.show()

