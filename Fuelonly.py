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
# %%
# Indlæser data
file_fuel = os.path.join(os.getcwd(), 'fuel gmk.csv')
df_fuel = pd.read_csv(file_fuel, sep=',')

#Fjerner evt mellemrum i kollonne navnene 
df_fuel.columns = df_fuel.columns.str.strip()

# Konverterer dato/tid til datetime
df_fuel["Transaction Date/Time"] = pd.to_datetime(df_fuel["Transaction Date/Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce" )


# Sikrer at Volume er numerisk
df_fuel["Volume"] = pd.to_numeric(df_fuel["Volume"], errors="coerce")

# Sorterer efter tid
df_fuel = df_fuel.sort_values("Transaction Date/Time")

# %%
# Summerer volumen pr. dag og fuel-type
fuel_daily = (
    df_fuel.groupby([df_fuel["Transaction Date/Time"].dt.date, "Product"])["Volume"]
    .sum()
    .unstack(fill_value=0)
)

# Konverterer index tilbage til datetime for pænere plot
fuel_daily.index = pd.to_datetime(fuel_daily.index)

# %%
# Plotter alle fuel-typer i samme figur
plt.figure(figsize=(12, 6))

for fuel_type in fuel_daily.columns:
    plt.plot(fuel_daily.index, fuel_daily[fuel_type], label=fuel_type)

plt.xlabel("Dato")
plt.ylabel("Samlet volumen")
plt.title("Fuel volume over tid opdelt på fuel-type")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Making plot of the average fuel consumption pr hour at gammelkongevej

# extracting hour from the datetime column
df_fuel["hour"] = df_fuel["Transaction Date/Time"].dt.hour

# Calculating average volume pr hour
hourly_avg = df_fuel.groupby("hour")["Volume"].mean()

# sørger for at alle timer 0–23 findes
hourly_avg = hourly_avg.reindex(range(24), fill_value=0)

print(hourly_avg)

plt.figure(figsize=(10,5))
hourly_avg.plot(kind="bar")

plt.xlabel("Time på dagen")
plt.ylabel("Average volume")
plt.title("Average fuel volume pr time på dagen")
plt.xticks(range(24))
plt.tight_layout()
plt.show()

# %%
# Making plot of the average fuel consumption seperated into fuel types pr hour at gammelkongevej

# extracting hour from the datetime column
df_fuel["hour"] = df_fuel["Transaction Date/Time"].dt.hour

# Calculating average volume pr hour and fuel type
hourly_avg = (
    df_fuel.groupby(["hour", "Product"])["Volume"]
    .mean()
    .unstack()
)

# sørger for at alle timer 0–23 findes
hourly_avg = hourly_avg.reindex(range(24), fill_value=0)

print(hourly_avg)

# Plot
plt.figure(figsize=(10,5))
hourly_avg.plot(kind="bar")

plt.xlabel("Time på dagen")
plt.ylabel("Average volume")
plt.title("Average fuel volume pr time på dagen opdelt på fuel type")
plt.xticks(range(24))
plt.tight_layout()
plt.show()

# %%
#calculation from the fuel demand to energy demand 
# Energy content per liter for different fuel types (in MJ/liter)
energy_content = {
    "Diesel": 35.8,  # MJ/liter
    "Benzin": 34.2,  # MJ/liter
    "El": 0.0,       # MJ/liter (for electric vehicles, we will handle this separately)
}   
