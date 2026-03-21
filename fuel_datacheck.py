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
df_gmk_match["_key"] = df_gmk_match["nummerplade"].str[:6]
df_fuel_match["_key"] = df_fuel_match["nummerplade"].str[:6]

df_merged = pd.merge_asof(
    df_gmk_match,
    df_fuel_match[["Transaction Date/Time", "_key", "nummerplade", "Volume"]].sort_values("Transaction Date/Time"),
    left_on="ind.tid",
    right_on="Transaction Date/Time",
    by="_key",
    direction="nearest",
    tolerance=pd.Timedelta("12h")
)

# Ryd op — fjern hjælpekolonnen og behold kun GMK's nummerplade
df_merged = df_merged.drop(columns=["_key"]).rename(columns={"nummerplade_x": "nummerplade", "nummerplade_y": "nummerplade_fuel"})
df_merged["timediff_timer"] = (
    (df_merged["Transaction Date/Time"] - df_merged["ind.tid"])
    .abs()
    .dt.total_seconds()
    / 3600
)

df_gmk = df_merged.set_index("ind.tid").copy()

df_gmk_match["_key"] = df_gmk_match["nummerplade"].str[:6]
df_fuel_match["_key"] = df_fuel_match["nummerplade"].str[:6]

# print(df_gmk["nummerplade"].str.len().value_counts())
# print(df_fuel["nummerplade"].str.len().value_counts())

df_gmk_match = df_gmk_match.sort_values(["_key", "ind.tid"])
df_fuel_match = df_fuel_match.sort_values(["_key", "Transaction Date/Time"])

# print(df_gmk["timediff_timer"].describe())
# print(df_gmk["timediff_timer"].value_counts(bins=10).sort_index())

# print(df_fuel["nummerplade"].sample(20).tolist())
# print(df_gmk["nummerplade"].sample(20).tolist())

# # Hvor mange unikke plader er der i hver?
# print("GMK unikke plader:", df_gmk["nummerplade"].nunique())
# print("Fuel unikke plader:", df_fuel["nummerplade"].nunique())

# Overlap
# gmk_plader = set(df_gmk["nummerplade"].dropna())
# fuel_plader = set(df_fuel["nummerplade"].dropna())
# print("Fælles plader:", len(gmk_plader & fuel_plader))
# print("Kun i GMK:", len(gmk_plader - fuel_plader))
# print("Kun i Fuel:", len(fuel_plader - gmk_plader))


# # -----------------------------
# # Debug / kontrol
# # -----------------------------
# print("Antal rækker med matched Volume:", df_gmk["Volume"].notna().sum())

# print(
#     df_gmk.loc[
#         df_gmk["Volume"].notna(),
#         ["nummerplade", "Transaction Date/Time", "Volume", "timediff_timer"]
#     ].head(20)
# )


#from tqdm import tqdm  # valgfri progress bar

# Lav en kopi af fuel så vi kan fjerne brugte tankninger
# Lav alle mulige kombinationer inden for 12 timer
records = []

for _, gmk_row in df_gmk_match.iterrows():
    plate = gmk_row["nummerplade"]
    tid = gmk_row["ind.tid"]
    
    candidates = df_fuel_match[df_fuel_match["nummerplade"] == plate].copy()
    if candidates.empty:
        continue
    
    candidates["timediff_signed"] = (candidates["Transaction Date/Time"] - tid).dt.total_seconds() / 3600
    candidates = candidates[
        (candidates["timediff_signed"] >= -3) &
        (candidates["timediff_signed"] <= 24)
    ]
    candidates["timediff_timer"] = candidates["timediff_signed"].abs()
    
    for _, fuel_row in candidates.iterrows():
        records.append({
            "ind.tid": tid,
            "gmk_idx": gmk_row.name,
            "fuel_idx": fuel_row.name,
            "nummerplade": plate,
            "Transaction Date/Time": fuel_row["Transaction Date/Time"],
            "Volume": fuel_row["Volume"],
            "timediff_timer": fuel_row["timediff_timer"]
        })

df_candidates = pd.DataFrame(records).sort_values("timediff_timer")

used_gmk = set()
used_fuel = set()
matched = []

for _, row in df_candidates.iterrows():
    if row["gmk_idx"] in used_gmk or row["fuel_idx"] in used_fuel:
        continue
    matched.append(row)
    used_gmk.add(row["gmk_idx"])
    used_fuel.add(row["fuel_idx"])

df_matched = pd.DataFrame(matched).set_index("ind.tid")
# print("Matchede volumener:", len(df_matched))
# print(df_matched[["nummerplade", "Transaction Date/Time", "Volume", "timediff_timer"]].head(20))
# print("Antal rækker med matched Volume:", df_matched["Volume"].notna().sum())

# Merge df_matched tilbage på den fulde GMK så umatchede rækker beholder NaN
df_gmk_final = df_gmk_match.copy()

df_gmk_final = df_gmk_final.merge(
    df_matched[["gmk_idx", "Transaction Date/Time", "Volume", "timediff_timer"]],
    left_index=True,
    right_on="gmk_idx",
    how="left"
).set_index("ind.tid")

# print("Matchede volumener:", len(df_matched))
# print("Antal rækker med matched Volume:", df_gmk_final["Volume"].notna().sum())
# print(df_gmk_final[["nummerplade", "Transaction Date/Time", "Volume", "timediff_timer"]].head(20))


# Hvor mange unikke plader i GMK har MINDST én tankning i fuel overhovedet?
plates_gmk = set(df_gmk_match["nummerplade"].dropna().unique())
plates_fuel = set(df_fuel_match["nummerplade"].dropna().unique())
fælles = plates_gmk & plates_fuel

print("Unikke plader i GMK:", len(plates_gmk))
print("Unikke plader i fuel:", len(plates_fuel))
print("Fælles plader:", len(fælles))

# Hvor mange GMK-rækker har en plade der overhovedet findes i fuel?
gmk_med_fuel_plade = df_gmk_match[df_gmk_match["nummerplade"].isin(plates_fuel)]
print("\nGMK-rækker hvor pladen findes i fuel:", len(gmk_med_fuel_plade))

# Af disse — hvor mange ligger inden for [-3, +24] timer af en tankning?
matches_mulige = 0
for _, gmk_row in gmk_med_fuel_plade.iterrows():
    plate = gmk_row["nummerplade"]
    tid = gmk_row["ind.tid"]
    candidates = df_fuel_match[df_fuel_match["nummerplade"] == plate].copy()
    candidates["timediff_signed"] = (candidates["Transaction Date/Time"] - tid).dt.total_seconds() / 3600
    candidates = candidates[
        (candidates["timediff_signed"] >= -8) &
        (candidates["timediff_signed"] <= 24)
    ]
    if len(candidates) > 0:
        matches_mulige += 1

print("GMK-rækker med mindst én kandidat inden for [-8, +24] timer:", matches_mulige)
print("Heraf ikke matched pga. genbrug forbudt:", matches_mulige - 2244)

for timer_efter in [24, 48, 72, 168]:  # 1, 2, 3, 7 dage
    matches_mulige = 0
    for _, gmk_row in gmk_med_fuel_plade.iterrows():
        plate = gmk_row["nummerplade"]
        tid = gmk_row["ind.tid"]
        candidates = df_fuel_match[df_fuel_match["nummerplade"] == plate].copy()
        candidates["timediff_signed"] = (candidates["Transaction Date/Time"] - tid).dt.total_seconds() / 3600
        candidates = candidates[
            (candidates["timediff_signed"] >= -8) &
            (candidates["timediff_signed"] <= timer_efter)
        ]
        if len(candidates) > 0:
            matches_mulige += 1
    print(f"[-8, +{timer_efter}t]: {matches_mulige} mulige matches")


#%%

# Hvor mange fuel-rækker har en GMK-aflevering inden for vinduet?
fuel_med_gmk_plade = df_fuel_match[df_fuel_match["nummerplade"].isin(plates_gmk)].copy()
print("Fuel-rækker hvor pladen findes i GMK:", len(fuel_med_gmk_plade))

matches_per_vindue = {}
for timer_efter in [12, 24, 48, 72, 168]:
    count = 0
    for _, fuel_row in fuel_med_gmk_plade.iterrows():
        plate = fuel_row["nummerplade"]
        tid = fuel_row["Transaction Date/Time"]
        candidates = df_gmk_match[df_gmk_match["nummerplade"] == plate].copy()
        candidates["timediff_signed"] = (candidates["ind.tid"] - tid).dt.total_seconds() / 3600
        candidates = candidates[
            (candidates["timediff_signed"] >= -12) &
            (candidates["timediff_signed"] <= timer_efter)
        ]
        if len(candidates) > 0:
            count += 1
    matches_per_vindue[timer_efter] = count
    print(f"Fuel-rækker med GMK-match inden for [-12, +{timer_efter}t]: {count}")


#%%

# Fuel-rækker der aldrig blev matched
brugte_fuel_idx = set(df_matched["fuel_idx"])

df_fuel_umatched = df_fuel_match[~df_fuel_match.index.isin(brugte_fuel_idx)].copy()

print("Antal umatchede tankninger:", len(df_fuel_umatched))
print(df_fuel_umatched[["nummerplade", "Transaction Date/Time", "Volume"]].sort_values("Transaction Date/Time"))

#%%

df_fuel_umatched[["nummerplade", "Transaction Date/Time", "Volume"]].sort_values("Transaction Date/Time").to_excel("umatchede_tankninger.xlsx", index=False)

# -----------------------------
# Overlap i nummerplader
# -----------------------------
# plates_gmk = set(df_gmk["nummerplade"].dropna().unique())
# plates_fuel = set(df_fuel["nummerplade"].dropna().unique())

# print("Nummerplader i GMK:", len(plates_gmk))
# print("Nummerplader i fuel:", len(plates_fuel))
# print("Fælles nummerplader:", len(plates_gmk & plates_fuel))
# print("Kun i fuel:", len(plates_fuel - plates_gmk))
# print("Kun i GMK:", len(plates_gmk - plates_fuel))

# print("Eksempler kun i fuel:", sorted(plates_fuel - plates_gmk)[:50])
# print("Eksempler kun i GMK:", sorted(plates_gmk - plates_fuel)[:50])

# -----------------------------
# Hvor mange fuel-rækker finder et match? (12 timer)
# # -----------------------------
# gmk_check = (
#     df_gmk.reset_index()[["ind.tid", "nummerplade"]]
#     .dropna()
#     .sort_values("ind.tid")
#     .copy()
# )

# fuel_check = (
#     df_fuel[["Transaction Date/Time", "nummerplade", "Volume"]]
#     .dropna()
#     .sort_values("Transaction Date/Time")
#     .copy()
# )

# fuel_to_gmk_12h = pd.merge_asof(
#     fuel_check,
#     gmk_check,
#     left_on="Transaction Date/Time",
#     right_on="ind.tid",
#     by="nummerplade",
#     direction="nearest",
#     tolerance=pd.Timedelta("12h")
# )

# print("hello", fuel_check)
# print("hello", gmk_check)
# print("antal rækker i gmk:", len(df_gmk))
# print("Fuel-rækker i alt:", len(fuel_to_gmk_12h))
# print("Fuel-rækker med match (12t):", fuel_to_gmk_12h["ind.tid"].notna().sum())
# print("Fuel-rækker uden match (12t):", fuel_to_gmk_12h["ind.tid"].isna().sum())
# #Rækker med match: 2606
# #Fuel-rækker med match (12t): 2233
# #Fuel-rækker uden match (12t): 1908

# # -----------------------------
# # Samme test med 24 timer
# # -----------------------------
# fuel_to_gmk_24h = pd.merge_asof(
#     fuel_check,
#     gmk_check,
#     left_on="Transaction Date/Time",
#     right_on="ind.tid",
#     by="nummerplade",
#     direction="nearest",
#     tolerance=pd.Timedelta("24h")
# )

# print("Fuel-rækker med match (24t):", fuel_to_gmk_24h["ind.tid"].notna().sum())
# print("Fuel-rækker uden match (24t):", fuel_to_gmk_24h["ind.tid"].isna().sum())

# # -----------------------------
# # Test uden tidsgrænse
# # -----------------------------
# fuel_to_gmk_no_limit = pd.merge_asof(
#     fuel_check,
#     gmk_check,
#     left_on="Transaction Date/Time",
#     right_on="ind.tid",
#     by="nummerplade",
#     direction="nearest"
# )

# fuel_to_gmk_no_limit["timediff_timer"] = (
#     (fuel_to_gmk_no_limit["Transaction Date/Time"] - fuel_to_gmk_no_limit["ind.tid"])
#     .abs()
#     .dt.total_seconds()
#     / 3600
# )

# #print(fuel_to_gmk_no_limit["timediff_timer"].describe())

# unmatched = fuel_to_gmk_no_limit[fuel_to_gmk_no_limit["ind.tid"].isna()]
# print("Uden match uden tidsgrænse:", len(unmatched))

# # Kun de der matcher inden for 7 dage — hvad er fordelingen?
# within_7d = fuel_to_gmk_no_limit[fuel_to_gmk_no_limit["timediff_timer"] <= 168]
# print(within_7d["timediff_timer"].describe())

# # Histogram over tidsdifferencer
# within_7d["timediff_timer"].hist(bins=50)
# plt.xlabel("Timer fra ind.tid til tankning")
# plt.ylabel("Antal")
# plt.title("Fordeling af tidsdifferencer (inden for 7 dage)")
# #plt.show()

# unmatched_fuel = fuel_to_gmk_no_limit[fuel_to_gmk_no_limit["ind.tid"].isna()]

# # Hvilke nummerplader er det?
# print(unmatched_fuel["nummerplade"].value_counts().head(20))

# # Er det spredt over hele året eller koncentreret?
# print(unmatched_fuel["Transaction Date/Time"].dt.month.value_counts().sort_index())

# # Kig på de umatchede og find nærmeste GMK-aflevering selv uden tidsgrænse
# print(fuel_to_gmk_no_limit[fuel_to_gmk_no_limit["ind.tid"].isna()]["nummerplade"].head(20).tolist())

# # Er disse nummerplader overhovedet i GMK-data?
# unmatched_plates = set(unmatched_fuel["nummerplade"].unique())
# gmk_plates = set(df_gmk.reset_index()["nummerplade"].dropna().unique())

# print("Umatchede plader der IKKE findes i GMK overhovedet:", len(unmatched_plates - gmk_plates))
# print("Umatchede plader der FINDES i GMK men ikke inden for tidsgrænsen:", len(unmatched_plates & gmk_plates))

# unmatched_plates_list = sorted(unmatched_plates - gmk_plates)

# # Udskriv dem alle
# print(unmatched_plates_list)

# # Hvor mange ligner danske nummerplader (2 bogstaver + 5 cifre)?
# import re
# dansk_format = [p for p in unmatched_plates_list if re.fullmatch(r'[A-Z]{2}\d{5}', p)]
# ikke_dansk = [p for p in unmatched_plates_list if not re.fullmatch(r'[A-Z]{2}\d{5}', p)]

# print(f"\nDansk format (AB12345): {len(dansk_format)}")
# print(f"Ikke dansk format: {len(ikke_dansk)}")
# print("\nIkke-danske plader:", ikke_dansk)

# # Findes disse plader overhovedet i det originale df_data (alle stationer)?
# df_data_reset = df_data.reset_index()
# alle_plader = set(df_data_reset["reg.nr"].apply(clean_plate).dropna().unique())

# print("Umatchede plader der findes på ANDRE stationer:", len(unmatched_plates & alle_plader))
# print("Umatchede plader der slet ikke findes i udlejningsdata:", len(unmatched_plates - alle_plader))

# # Findes de 307 dansk-formaterede umatchede plader på andre stationer?
# dansk_umatchede = set(dansk_format)
# andre_stationer = set(df_data_reset["reg.nr"].apply(clean_plate).dropna().unique())

# print("Dansk-formaterede umatchede plader der findes på andre stationer:", 
#       len(dansk_umatchede & andre_stationer))
# print("Dansk-formaterede umatchede plader der slet ikke findes i systemet:", 
#       len(dansk_umatchede - andre_stationer))