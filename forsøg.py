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
file_fuel = os.path.join(os.getcwd(), 'fuel_nu.csv')

df_data1 = pd.read_csv(file_PE, sep=';', encoding='latin1')
#print(len(df_data1))

df_data2 = pd.read_csv(file_PE2, sep=';', encoding='latin1')
df_fuel = pd.read_csv(file_fuel, sep=',')

# samler de to datasæt, så vi kigger på et helt år
df_data = pd.concat([df_data1, df_data2], ignore_index=True)
#print(len(df_data))

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
df_fuel["Transaction Date/Time"] = pd.to_datetime(
    df_fuel["Transaction Date/Time"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)
#laver ny kolonne som har samme datetime format som de andre dataframes
df_fuel["Transaction Date/Time_str"] = df_fuel["Transaction Date/Time"].dt.strftime("%d-%m-%Y %H:%M:%S")

dt.loc[mask_ind] = dt.loc[mask_ind] + pd.Timedelta(days=1)
dt_ud.loc[mask_ud] = dt_ud.loc[mask_ud] + pd.Timedelta(days=1)

#print("NaT i ind.tid:", dt.isna().sum())
#print("NaT i ud.tid:", dt_ud.isna().sum())

#indsæter nye ind.tid og ud.tid og fjerner alle rækker hvor de er NaN
df_data["ind.tid"] = dt
df_data["ud.tid"] = dt_ud

df_data = df_data.dropna(subset=["ind.tid"])
df_data = df_data.dropna(subset=["ud.tid"])

#print(len(df_data))

#Sætter ind.tid som index, da det er det der skal analyseres på først
df_data.set_index("ind.tid", inplace=True)

#print(df_data)  

#fjerner alle rækker som ikke er status 4 da status 4 er afsluttede udlejninger
mask_df_data = df_data["stat"].astype(str).str.contains("4", na=False)
df_data = df_data.loc[mask_df_data]
#print(len(df_data))
#Finder de biler der kommer ind på gammel kongevej
mask_gmk = df_data["st.i"].astype(str).str.fullmatch("5.0", na=False)
df_gmk = df_data.loc[mask_gmk]
#print("hej", df_gmk)

#Here we create a variabel for all columns to drop and then drop them from the dataset
Columns_todrop =[
    "kon.nr", "spcgrp", "spcnr", "k/f", "st.u", "st.i", "stat", "leje.dg", "oprettelse",
    "udl.land", "lejer", "firmabss", "firma", "land", "mærke", "model", "km.incl", "styr.rate", "styr.ratekode",
    "rate2", "rate2-dkk", "rate3", "rate3-dkk", "rate4", "rate4-dkk", "rate5", "rate5-dkk", "rate6", "rate6-dkk",
    "rate7", "rate7-dkk", "rate8", "rate8-dkk", "rate9", "rate9-dkk", "rate10", "rate10-dkk", "extrakm-dkk", "moms",
    "forsikring", "total", "dekort", "check-out", "exp-check-in", "check-in", "km"  , 
]
df_gmk = df_gmk.drop(columns=Columns_todrop)

# We only keep the rows where these specific cargroups ['SFAR', 'SWAR', 'GWAR','CCAR', 'CDMR'] are NOT in the list
df_gmk = df_gmk[~df_gmk['bilgrp'].isin(['SFAR', 'SWAR', 'GWAR','CCAR', 'CDMR'])]


# Adding date-columns for merge (without changing the index)
df_gmk["dato"] = pd.to_datetime(df_gmk.index).normalize()
df_fuel["dato"] = df_fuel["Transaction Date/Time"].dt.normalize()

# Gør index til kolonne
df_gmk = df_gmk.reset_index()


# Sørg for datetime
df_gmk["ind.tid"] = pd.to_datetime(df_gmk["ind.tid"], errors="coerce")
df_fuel["Transaction Date/Time"] = pd.to_datetime(df_fuel["Transaction Date/Time"], errors="coerce")

# -----------------------------
# Rens nummerplader
# -----------------------------
def clean_plate(s):
    if pd.isna(s):
        return pd.NA
    s = "".join(ch for ch in str(s).upper().strip().replace(" ", "").replace("-", "") if ch.isalnum())
    return s if s else pd.NA

def looks_like_plate(s):
    return pd.notna(s) and len(s) >= 5 and any(c.isalpha() for c in str(s)) and any(c.isdigit() for c in str(s))

df_gmk = df_gmk.copy()
df_fuel = df_fuel.copy()

df_gmk["nummerplade"] = df_gmk["reg.nr"].apply(clean_plate)
df_fuel["nummerplade"] = df_fuel["Vehicle Number"].apply(clean_plate)

# Behold kun realistiske nummerplader
df_gmk = df_gmk[df_gmk["nummerplade"].apply(looks_like_plate)].copy()
df_fuel = df_fuel[df_fuel["nummerplade"].apply(looks_like_plate)].copy()

# -----------------------------
# Klargør data til matching
# -----------------------------
df_gmk_match = (
    df_gmk.reset_index()
    .dropna(subset=["ind.tid", "nummerplade"])
    .sort_values("ind.tid")
    .copy()
)

df_fuel_match = (
    df_fuel.dropna(subset=["Transaction Date/Time", "nummerplade", "Volume"])
    .sort_values("Transaction Date/Time")
    .copy()
)

# -----------------------------
# Match nærmeste tankning til ind.tid inden for 12 timer
# -----------------------------
df_merged = pd.merge_asof(
    df_gmk_match,
    df_fuel_match[["Transaction Date/Time", "nummerplade", "Volume"]].sort_values("Transaction Date/Time"),
    left_on="ind.tid",
    right_on="Transaction Date/Time",
    by="nummerplade",
    direction="nearest",   # skift til "forward" hvis tankning kun må ske efter ind.tid
    tolerance=pd.Timedelta("12h")
)

df_merged["timediff_timer"] = (
    (df_merged["Transaction Date/Time"] - df_merged["ind.tid"])
    .abs()
    .dt.total_seconds()
    / 3600
)

df_gmk = df_merged.set_index("ind.tid").copy()

# -----------------------------
# Debug / kontrol
# -----------------------------
print("Antal rækker med matched Volume:", df_gmk["Volume"].notna().sum())

print(
    df_gmk.loc[
        df_gmk["Volume"].notna(),
        ["nummerplade", "Transaction Date/Time", "Volume", "timediff_timer"]
    ].head(20)
)

# -----------------------------
# Overlap i nummerplader
# -----------------------------
plates_gmk = set(df_gmk["nummerplade"].dropna().unique())
plates_fuel = set(df_fuel["nummerplade"].dropna().unique())

print("Nummerplader i GMK:", len(plates_gmk))
print("Nummerplader i fuel:", len(plates_fuel))
print("Fælles nummerplader:", len(plates_gmk & plates_fuel))
print("Kun i fuel:", len(plates_fuel - plates_gmk))
print("Kun i GMK:", len(plates_gmk - plates_fuel))

print("Eksempler kun i fuel:", sorted(plates_fuel - plates_gmk)[:50])
print("Eksempler kun i GMK:", sorted(plates_gmk - plates_fuel)[:50])

# -----------------------------
# Hvor mange fuel-rækker finder et match? (12 timer)
# -----------------------------
gmk_check = (
    df_gmk.reset_index()[["ind.tid", "nummerplade"]]
    .dropna()
    .sort_values("ind.tid")
    .copy()
)

fuel_check = (
    df_fuel[["Transaction Date/Time", "nummerplade", "Volume"]]
    .dropna()
    .sort_values("Transaction Date/Time")
    .copy()
)

fuel_to_gmk_12h = pd.merge_asof(
    fuel_check,
    gmk_check,
    left_on="Transaction Date/Time",
    right_on="ind.tid",
    by="nummerplade",
    direction="nearest",
    tolerance=pd.Timedelta("12h")
)

print("antal rækker i gmk:", len(df_gmk))
print("Fuel-rækker i alt:", len(fuel_to_gmk_12h))
print("Fuel-rækker med match (12t):", fuel_to_gmk_12h["ind.tid"].notna().sum())
print("Fuel-rækker uden match (12t):", fuel_to_gmk_12h["ind.tid"].isna().sum())

# -----------------------------
# Samme test med 24 timer
# -----------------------------
fuel_to_gmk_24h = pd.merge_asof(
    fuel_check,
    gmk_check,
    left_on="Transaction Date/Time",
    right_on="ind.tid",
    by="nummerplade",
    direction="nearest",
    tolerance=pd.Timedelta("24h")
)

print("Fuel-rækker med match (24t):", fuel_to_gmk_24h["ind.tid"].notna().sum())
print("Fuel-rækker uden match (24t):", fuel_to_gmk_24h["ind.tid"].isna().sum())

# -----------------------------
# Test uden tidsgrænse
# -----------------------------
fuel_to_gmk_no_limit = pd.merge_asof(
    fuel_check,
    gmk_check,
    left_on="Transaction Date/Time",
    right_on="ind.tid",
    by="nummerplade",
    direction="nearest"
)

fuel_to_gmk_no_limit["timediff_timer"] = (
    (fuel_to_gmk_no_limit["Transaction Date/Time"] - fuel_to_gmk_no_limit["ind.tid"])
    .abs()
    .dt.total_seconds()
    / 3600
)

print(fuel_to_gmk_no_limit["timediff_timer"].describe())

unmatched = fuel_to_gmk_no_limit[fuel_to_gmk_no_limit["ind.tid"].isna()]
print("Uden match uden tidsgrænse:", len(unmatched))
#%% Kapacitet af grupperne


df_gmk["en_consumption"] = np.nan
# mask_df_data_km_L = df_data["bilgrp"].astype(str).str.contains("4", na=False)
# df_data = df_data.loc[mask_df_data]


bilgrp = df_gmk["bilgrp"].fillna("").astype(str)
første = bilgrp.str[0]
andet = bilgrp.str[1:2]


km_map = {
    "O": 20.5,
    "A": 19.0,
    "B": 17.75,
    "C": 17.0,
    "D": 16.5,
    "E": 16.25,
    "F": 18.5,
    "X": 17.25,
    "G": 17.33,
    "H": 15.5,
    "L": 14.0,
    "M": 13.5,
    "I": 16.0,
    "J": 12.5,
    "V": 12.5,
    "1": 16.475,
    "3": 17.5,
    "4": 13.0,

}

km_kWh_map = {
    "OE": 6.5,
    "AE": 7.1,
    "BE": 6.7,
    "CE": 6.43,
    "DE": 6.15,
    "EE": 6.2,
    "HE": 6.4,
    "IE": 6.4,
    "LE": 6.167,
    "FE": 6.167,
    "GE": 6.167,
    "XE": 6.167,
    "ME": 5.9,
    "JE": 4.85,
    "VE": 3.1,
    "1E": 5.65,
    "3E": 3.75,
    "4E": 3.1,
}

#tag lige stilling til 2 og 7

mask = første.isin(km_map) & ~andet.eq("E")

df_gmk.loc[mask, "en_consumption"] = (
    df_gmk.loc[mask, "Volume"]
    * første.map(km_map)[mask]
    / (første + "E").map(km_kWh_map)[mask]
)

#print("Dataframe with en_consumption", df_gmk)
#print(df_gmk.index.min(), df_gmk.index.max())
print(df_gmk[["bilgrp","Volume", "en_consumption"]].dropna())
print(df_gmk["en_consumption"].notna().sum())
print(df_gmk["Volume"].notna().sum())

# df_problem = df_gmk[
#     df_gmk["Volume"].notna() & df_gmk["en_consumption"].isna()
# ]

# print(df_problem[["bilgrp", "Volume", "en_consumption"]])
# print(
#     df_problem[["bilgrp", "Volume"]]
#     .value_counts()
#     .head(20)
# )


#%%
df_gmk = df_gmk[df_gmk.index <= "2025-12-31"]
consumption_per_day = df_gmk["en_consumption"].groupby(df_gmk.index.normalize()).sum()

# Plot
plt.figure(figsize=(12,6))
plt.plot(consumption_per_day.index, consumption_per_day.values)
plt.xlabel("Dato")
plt.ylabel("Samlet energiforbrug i kWh")
plt.title("Samlet energiforbrug pr. dag")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%%
# Laver simpelt plot af volumen fuel pr dag:
# Summer samlet volume pr. dag
volume_per_day = df_gmk["Volume"].groupby(df_gmk.index.normalize()).sum()


# Plot
plt.figure(figsize=(12,6))
plt.plot(volume_per_day.index, volume_per_day.values)
plt.xlabel("Dato")
plt.ylabel("Samlet volumen i liter")
plt.title("Samlet fuel volume pr. dag")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
#Laver plot af volumen pr hour for alle dagene:
df_fuel["hour"] = df_fuel["Transaction Date/Time"].dt.hour
hour_counts_fuel = df_fuel["hour"].value_counts().sort_index()

print(hour_counts_fuel)

hour_counts_fuel.plot(kind="bar")
plt.xlabel("Hour of day")
plt.ylabel("Number of arrivals")
plt.title("Distribution of fuels pr hour on the day")
plt.show()

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
#print("EV", df_EV) #her er 3161  rækker
print(df_EV.columns)

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
print("rate 939+935", df_notfull, df_939) #her er 309 rækker


#print("gmk", df_gmk)

# %% hvornår bliver elbilerne udlejet ?
df_EV["hour"] = df_EV.index.hour
hour_counts_EV = df_EV["hour"].value_counts().sort_index()
print(df_EV.columns)

#df_EV = df_EV.drop(columns=['kon.nr', 'spcgrp', 'spcnr', 'km', 'k/f', 'st.u', 'st.i', 'stat', 'oprettelse', 'udl.land', 'lejer', 'firmabss', 'firma', 'land','extrakm', 'extrakm-dkk', 'moms', 'forsikring', 'dekort', 'check-out','exp-check-in')

# %% hvornår ankommer bilerne på gammel kongevej?

df_gmk["hour"] = df_gmk.index.hour
hour_counts = df_gmk["hour"].value_counts().sort_index()

print(hour_counts)

#hour_distribution = hour_counts / hour_counts.sum()

#print(hour_distribution)
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
plt.title("Distribution of EV arrivals by hour")
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

#%% Bilers ankomst fordelt på dagene for hver time i gennemsnit
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
#cols = ["sum_km","ud.tid","bilgrp","k/f","st.i","forsikring","mærke","leje.dg"]
#the country of which the renter is from 

import seaborn as sns

#plt.figure(figsize=(10,6))
#sns.heatmap(df_data[cols], cmap="viridis")

#plt.title("Heatmap of 10 columns")
#plt.show()

#df_data["hour"] = df_data.index.hour
#%%
#plt.figure(figsize=(12,6))
#sns.heatmap(heatmap_data, cmap="viridis")

#lt.xlabel("Columns")
#plt.ylabel("Hour of day")
#plt.title("Average values by hour")

#plt.show()

#%%
#%%
# Undersøger hvor mange udlejninger vs antallet af udlejningsdage
# For at se fordelingen i udlejningsperioderne:

# Kopiér kolonnen
rental_days = df_data["leje.dg"].dropna().copy()

# Saml alle >30 dage i kategorien 30
rental_days[rental_days > 30] = 30

# Plot histogram
plt.figure(figsize=(10,6))

counts, bins, patches = plt.hist(
    rental_days,
    bins=range(1,32),
    edgecolor="black"
)
#Beregner hvor mange procentdel af udlejningerne varer 1-7 dage?
percentage = rental_days[rental_days <= 7].count() / rental_days.count() * 100

plt.text(
    8,  # x-position i plottet
    plt.ylim()[1]*0.8,  # y-position
    "Percentage of rentals lasting 1–7 days: {:.2f}%".format(percentage),
    bbox=dict(facecolor="white", edgecolor="black")
)

# Farver bars i Europcars farver
for i, patch in enumerate(patches):

    day = bins[i]

    if 1 <= day <= 7:
        patch.set_facecolor("yellow")   # 1-7 dage
    else:
        patch.set_facecolor("green")    # resten

plt.xlabel("Rental duration (days)")
plt.ylabel("Number of rentals")
plt.title("Distribution of rental durations")

# x-akse labels
ticks = list(range(1,31))
labels = [str(t) for t in ticks]
labels[-1] = "30+"

plt.xticks(ticks, labels, rotation=90)

plt.show()

# %%
#Hvor mange procentdel af udlejningerne varer 1-7 dage?
short_rentals = rental_days[rental_days <= 7].count()
total_rentals = rental_days.count()
print("Percentage of rentals lasting 1-7 days:", short_rentals / total_rentals * 100, "%")

# %%
#Average af indkommende biler vs udlejninger pr dag
# Antal ankomster pr. dag (baseret på index = ind.tid)
daily_arrivals = df_gmk.groupby(df_gmk.index.normalize()).size()

#finder average af ankomster pr dag
daily_arrivals_avg = daily_arrivals.mean()

# Antal udlejninger pr. dag (baseret på ud.tid-kolonnen)
daily_rentals = df_gmk.groupby(df_gmk["ud.tid"].dt.normalize()).size()

#finder average af udlejninger pr dag
daily_rentals_avg = daily_rentals.mean()

#Farver bar i Europcars farver


# Plot
plt.figure(figsize=(12,6))
plt.plot(daily_arrivals.index, daily_arrivals.values, label="Daily arrivals", color="black")
plt.plot(daily_rentals.index, daily_rentals.values, label="Daily rentals", color="green")

plt.xlabel("Date")
plt.ylabel("Count")
plt.title("Daily arrivals vs daily rentals")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Dette plot gør det meget tydeligt at de næsten af præcis samme antal indkommende biler som udlejninger pr dag
#Det er interessant ift at kunne estimere det endelige energibheov hvis det var elbiler der var tale om
# %%
#Nu vil jeg have det pr time i stedet for pr dag:

# Extracter timerne hen ad dagen
hourly_arrivals = df_gmk.groupby(df_gmk.index.hour).size()

# Antal udlejninger pr. time
hourly_rentals = df_gmk.groupby(df_gmk["ud.tid"].dt.hour).size()

# Sørg for at alle 24 timer er med
hourly_arrivals = hourly_arrivals.reindex(range(24), fill_value=0)
hourly_rentals = hourly_rentals.reindex(range(24), fill_value=0)

# Plot
plt.figure(figsize=(12,6))
plt.plot(hourly_arrivals.index, hourly_arrivals.values, label="Hourly arrivals", color="black")
plt.plot(hourly_rentals.index, hourly_rentals.values, label="Hourly rentals", color="green")

plt.xlabel("Hour of day")
plt.ylabel("Count")
plt.title("Hourly arrivals vs hourly rentals")
plt.legend()
plt.xticks(range(24))
plt.tight_layout()
plt.show()

# %%
#Finder ud af de specifikke omvendningtider alle bilerne har 

Omvendningstider = df_gmk.index - df_gmk["ud.tid"]
print(len(df_gmk))
print(len(Omvendningstider))

# %%
# lav kolonner for time og ugedag
df_gmk["hour"] = df_gmk.index.hour
df_gmk["day"] = df_gmk.index.dayofweek

# Tæller antal ankomster
hour_day_counts = df_gmk.groupby(["day", "hour"]).size().unstack(fill_value=0)

# sørg for alle timer er med
hour_day_counts = hour_day_counts.reindex(columns=range(24), fill_value=0)

# Finder antal unikke dage per ugedag
days_count = df_gmk.groupby("day").apply(lambda x: x.index.normalize().nunique())

# Dividere for at finde → gennemsnit pr dag (pr time)
hour_day_avg = hour_day_counts.div(days_count, axis=0)

# giver dagene navne
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
hour_day_avg.index = days

# transponere så timer bliver x-akse
plot_data = hour_day_avg.T

# plot
plt.figure(figsize=(14,6))
plot_data.plot(kind="bar", width=0.8)

plt.xlabel("Hour of day")
plt.ylabel("Average number of arrivals")
plt.title("Average car arrivals by hour and day of week")
plt.legend(title="Day of week")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
#%%
#Grouping the cars in sizes as Europcar alsp does on their websites for the customers

# %%
# lav kolonne for time
df_gmk["hour"] = df_gmk.index.hour

# Tæller ankomster per bilgruppe og time
hour_group_counts = df_gmk.groupby(["bilgrp", "hour"]).size().unstack(fill_value=0)

# sørg for alle timer er med
hour_group_counts = hour_group_counts.reindex(columns=range(24), fill_value=0)

# Antal unikke dage i hele datasættet
n_days = df_gmk.index.normalize().nunique()

# Gennemsnit pr dag
hour_group_avg = hour_group_counts / n_days

# Transponér → timer på x-akse
plot_data = hour_group_avg.T

# plot
plt.figure(figsize=(14,6))
plot_data.plot(kind="bar", width=0.8)

plt.xlabel("Hour of day")
plt.ylabel("Average number of arrivals per day")
plt.title("Average car arrivals by hour and vehicle group")
plt.legend(title="Vehicle group")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

#%%

print(df_gmk["bilgrp"].dropna().astype(str).head(50).tolist())
#%%
#Samler grupperne
df_gmk["bilgrp_original"] = df_gmk["bilgrp"]

bevar_prefix = {"AE", "BE", "CE", "DE", "EE", "HE", "IE", "LE", "FE", "GE", "XE", "ME", "JE", "VE", "1E", "3E", "4E"}

def saml_gruppe(x):
    if pd.isna(x):
        return x
    
    x = str(x).strip().upper()
    if not x:
        return x
    
    if len(x) >= 2 and x[:2] in bevar_prefix:
        return x
    
    return x[0]

# ny kolonne, ikke overskriv originalen
df_gmk["bilgrp_samlet"] = df_gmk["bilgrp_original"].apply(saml_gruppe)

print(df_gmk[["bilgrp_original", "bilgrp_samlet"]])

counts = (
    df_gmk["bilgrp_original"]
    .astype(str)
    .str.strip()
    .str.upper()
    .str[:2]
    .value_counts()
)

print(counts[counts.index.isin(bevar_prefix)])

#%%
# Average fuel volume pr. time pr. dag opdelt på bilgruppe


# lav kolonne for time
df_gmk["hour"] = df_gmk.index.hour

# sørg for at Volume er numerisk
df_gmk["Volume"] = pd.to_numeric(df_gmk["Volume"], errors="coerce")

# summer volume per bilgruppe og time
hour_group_volume = df_gmk.groupby(["bilgrp_samlet", "hour"])["Volume"].sum().unstack(fill_value=0)

# sørg for alle timer er med
hour_group_volume = hour_group_volume.reindex(columns=range(24), fill_value=0)

# antal unikke dage i hele datasættet
n_days = df_gmk.index.normalize().nunique()

# gennemsnitlig volume pr dag
hour_group_volume_avg = hour_group_volume / n_days

# transponér så timer bliver x-akse
plot_data = hour_group_volume_avg.T

# plot
plt.figure(figsize=(14,6))
plot_data.plot(kind="bar", width=0.8)

plt.xlabel("Hour of day")
plt.ylabel("Average fuel volume per day")
plt.title("Average fuel volume by hour and vehicle group")
plt.legend(title="Vehicle group")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# %%
