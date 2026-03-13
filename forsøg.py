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

#indlæser csv filerne og sætter ind.tid som index, da det er det der skal analyseres på
file_PE = os.path.join(os.getcwd(), 'Alle RA 2025 pr. 2026-03-11 Q1-Q2.csv')
file_PE2 = os.path.join(os.getcwd(), 'Alle RA 2025 pr. 2026-03-11 Q3-Q4.csv')
file_fuel = os.path.join(os.getcwd(), 'fuel gmk.csv')

df_data1 = pd.read_csv(file_PE, sep=';', encoding='latin1')
df_data2 = pd.read_csv(file_PE2, sep=';', encoding='latin1')
df_fuel = pd.read_csv(file_fuel, sep=',')

# samler de to datasæt, så vi kigger på et helt år
df_data = pd.concat([df_data1, df_data2], ignore_index=True)

# fjerner eventuelle mellemrum i kolonnenavnene, da det kan give problemer senere
df_data.columns = df_data.columns.str.strip()
df_fuel.columns = df_fuel.columns.str.strip()


# ind.tid har nogle gange 24:00, hvilket ikke er en gyldig tid, så det erstattes med 00:00 og tilføjer en dag
dt = df_data["ind.tid"].astype(str).str.strip()
mask = dt.str.endswith("24:00")

dt_fuel = df_fuel["Transaction Date/Time"].astype(str).str.strip()

dt = dt.str.replace("24:00", "00:00", regex=False)
dt = pd.to_datetime(dt, format="%d-%m-%Y %H:%M", dayfirst=True)
dt_fuel= pd.to_datetime(df_fuel["Transaction Date/Time"], format="%Y-%m-%d %H:%M:%S")
dt.loc[mask] = dt.loc[mask] + pd.Timedelta(days=1)

#fjerne alle rækker hvor ind.tid er NaN
df_data["ind.tid"] = dt
df_data.set_index("ind.tid", inplace=True)
df_data = df_data[~df_data.index.isna()]


#fjerner alle rækker som ikke er status 4
mask_df_data = df_data["stat"].astype(str).str.contains("4", na=False)
df_data = df_data.loc[mask_df_data]

#Finder de biler der kommer ind på gammel kongevej
mask_gmk = df_data["st.i"].astype(str).str.fullmatch("5.0", na=False)
df_gmk = df_data.loc[mask_gmk].copy()

df_gmk.columns = df_gmk.columns.str.strip()

df_fuel.set_index("Transaction Date/Time", inplace=True)
df_fuel = df_fuel[~df_fuel.index.isna()]
#print("Dataframe with fuels:", df_fuel)

# Create 'ind.tid' column from index
df_fuel['ind.tid'] = pd.to_datetime(df_fuel.index).strftime("%d-%m-%Y %H:%M")

# Create formatted string from df_data index
df_gmk['ind.tid_str'] = df_gmk.index.strftime("%d-%m-%Y %H:%M")

# Merge
df_merged = pd.merge(
    df_gmk,
    df_fuel[['ind.tid', 'Vehicle Number','Volume', 'Total Price']],
    left_on='ind.tid_str',
    right_on='ind.tid',
    how='left'
)
df_merged = df_merged.drop(columns=["mærke","k/f" ,"ud.tid", "leje.dg","km", "extrakm", "kon.nr", "spcgrp", "spcnr", "st.u", "oprettelse", "udl.land", "lejer", "firmabss", "land", "firma", "model", "km.incl", "extrakm-dkk", "forsikring", "dekort", "exp-check-in", "check-in", "check-out", "moms", "total"])

print(df_merged["Volume"])
print(df_merged["Volume"].notna().value_counts())




#%%
#samler km og extrakm i en kolonne, da det er det samlede antal km der er interessant for analysen
df_data["km"] = pd.to_numeric(df_data["km"], errors="coerce").fillna(0)
df_data["extrakm"] = pd.to_numeric(df_data["extrakm"], errors="coerce").fillna(0)
df_data["sum_km"] = df_data["km"].values + df_data["extrakm"].values

#fjerner alt det unødvendige for analysen, da det er en stor dataramme og det gør det nemmere at arbejde med
#df_data = df_data.drop(columns=["km", "extrakm", "kon.nr", "spcgrp", "spcnr", "st.u", "oprettelse", "udl.land", "lejer", "firmabss", "land", "firma", "model", "km.incl", "extrakm-dkk", "forsikring", "dekort", "exp-check-in", "check-in", "check-out", "moms", "total"])


#%% Status 4, EV, 932+939, gmk
#her finder jeg kun status 4

#print("data_s4", df_data) #130783  rækker

# her finder jeg elbilerne samt dem der har betalt for ikke at være opladt
mask_EV = df_data["bilgrp"].astype(str).str.match(r"^.E")
df_EV = df_data[mask_EV]
print("EV", df_EV) #her er 3161  rækker

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



#print("gmk", df_gmk)

# %% hvornår ankommer bilerne på gammel kongevej?

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

# %%  hvornår ankommer elbiler vs ikke er ladet op?


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


#%% #plot fordelingen baseret på dagene gmk
df_gmk["day"] = df_gmk.index.dayofweek
day_counts = df_gmk["day"].value_counts().sort_index()

print(day_counts)

day_distribution = day_counts / day_counts.sum()

print(day_distribution)
day_counts.plot(kind="bar")
plt.xlabel("Day of week (0=Monday, 6=Sunday)")
plt.ylabel("Number of arrivals")
plt.title("Distribution of car arrivals by day of week")
plt.show()

#%% Bilers ankomst fordelt på dagene for hver time
# lav kolonner for time og ugedag
df_gmk["hour"] = df_gmk.index.hour
df_gmk["day"] = df_gmk.index.dayofweek

# tæl ankomster per dag og time
hour_day_counts = df_gmk.groupby(["day", "hour"]).size().unstack(fill_value=0)

# sørg for alle timer er med
hour_day_counts = hour_day_counts.reindex(columns=range(24), fill_value=0)

# giv dagene navne
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
hour_day_counts.index = days

# transponér så timer bliver x-akse
plot_data = hour_day_counts.T

# plot
plt.figure(figsize=(14,6))
plot_data.plot(kind="bar", width=0.8)

plt.xlabel("Hour of day")
plt.ylabel("Number of arrivals")
plt.title("Car arrivals by hour and day of week")
plt.legend(title="Day of week")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

#%% Forskellen mellem 17-07 og 08-16 for hver dag
# lav kolonne med time
df_gmk["hour"] = df_gmk.index.hour

# funktion der grupperer timer
def hour_group(h):
    if (17 <= h <= 23) or (0 <= h <= 7):
        return "17-07"
    else:
        return "08-16"

df_gmk["hour_group"] = df_gmk["hour"].apply(hour_group)

df_gmk["day"] = df_gmk.index.dayofweek

group_counts = df_gmk.groupby(["day", "hour_group"]).size().unstack(fill_value=0)

days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
group_counts.index = days

group_counts.plot(kind="bar", figsize=(10,6))

plt.xlabel("Day of week")
plt.ylabel("Number of arrivals")
plt.title("Car arrivals grouped by time period")
plt.legend(title="Hour group")

plt.tight_layout()
plt.show()

#%% Andelen af ankomster uden for åbningstider
#finder andelen før 7 og efter 17
before_7 = df_gmk[df_gmk["hour"] <= 7].shape[0]
after_17 = df_gmk[df_gmk["hour"] >= 17].shape[0]
samlet = before_7 + after_17
total = df_gmk.shape[0]

print(before_7, after_17, total)
print("Percentage of arrivals outside opening hours:", samlet / total * 100, "%")


#%% timer og ugedage sammenlignet i hvert sit plot

# lav kolonner for time og ugedag
df_gmk["hour"] = df_gmk.index.hour
df_gmk["day"] = df_gmk.index.dayofweek

# tæl ankomster per dag og time
hour_day_counts = df_gmk.groupby(["day", "hour"]).size().unstack(fill_value=0)

# sørg for alle 24 timer er med
hour_day_counts = hour_day_counts.reindex(columns=range(24), fill_value=0)

# plot 7 plots
fig, axes = plt.subplots(7, 1, figsize=(10, 18), sharex=True)

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

for i, ax in enumerate(axes):
    hour_day_counts.loc[i].plot(kind="bar", ax=ax)
    ax.set_title(days[i])
    ax.set_ylabel("Arrivals")


plt.xlabel("Hour of day")
plt.tight_layout()
plt.show()

#%% Månedsfordelingen på et år gmk
df_gmk["month"] = df_gmk.index.month
month_counts = df_gmk["month"].value_counts().sort_index()

month_distribution = month_counts / month_counts.sum()

month_counts.plot(kind="bar")
plt.xlabel("Month")
plt.ylabel("Number of arrivals")
plt.title("Distribution of car arrivals by month in gmk")
plt.show()
#%% Månedsfordelingen på et år gmk
df_data["month"] = df_data.index.month
month_counts_all = df_data["month"].value_counts().sort_index()

month_distribution_all = month_counts_all / month_counts_all.sum()
month_counts_all.plot(kind="bar")
plt.xlabel("Month")
plt.ylabel("Number of arrivals")
plt.title("Distribution of car arrivals by month for all locations")
plt.show()


# %% heatmap for timer og ugedage (ikke så relevant)

# lav kolonner for time og ugedag
df_gmk["hour"] = df_gmk.index.hour
df_gmk["day"] = df_gmk.index.dayofweek

# tæl antal ankomster per dag og time
heatmap_data = df_gmk.groupby(["day", "hour"]).size().unstack(fill_value=0)

# sørg for alle timer 0-23 er med
heatmap_data = heatmap_data.reindex(columns=range(24), fill_value=0)

# sørg for alle dage 0-6 er med
heatmap_data = heatmap_data.reindex(index=range(7), fill_value=0)

# navne på dage
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heatmap_data.index = day_names

# plot heatmap
plt.figure(figsize=(14, 6))
plt.imshow(heatmap_data, aspect="auto")

plt.colorbar(label="Number of arrivals")
plt.xticks(ticks=range(24), labels=range(24))
plt.yticks(ticks=range(7), labels=day_names)

plt.xlabel("Hour of day")
plt.ylabel("Day of week")
plt.title("Car arrivals by day and hour")
plt.show()

#%% heatmap for andelen (også ikke så relevant)

heatmap_distribution = heatmap_data.div(heatmap_data.sum(axis=1), axis=0).fillna(0)

plt.figure(figsize=(14, 6))
plt.imshow(heatmap_distribution, aspect="auto")

plt.colorbar(label="Share of daily arrivals")
plt.xticks(ticks=range(24), labels=range(24))
plt.yticks(ticks=range(7), labels=day_names)

plt.xlabel("Hour of day")
plt.ylabel("Day of week")
plt.title("Distribution of car arrivals by day and hour")
plt.show()

 #%% Leger lidt
cols = ["sum_km","ud.tid","bilgrp","k/f","st.i","forsikring","mærke","leje.dg"]
#the country of which the renter is from 

import seaborn as sns

plt.figure(figsize=(10,6))
sns.heatmap(df_data[cols], cmap="viridis")

plt.title("Heatmap of 10 columns")
plt.show()

df_data["hour"] = df_data.index.hour
#%%
plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, cmap="viridis")

plt.xlabel("Columns")
plt.ylabel("Hour of day")
plt.title("Average values by hour")

plt.show()