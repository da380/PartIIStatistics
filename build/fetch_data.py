"""Reproduce the files in data/ from their original online sources.

Run from anywhere:  python build/fetch_data.py
The notebooks read the committed CSVs from GitHub, so this only needs re-running
to refresh the data (the CO2 and temperature series are extended monthly, and the
earthquake catalogue is revised for a while after each event).
"""
import os
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
os.makedirs(DATA, exist_ok=True)

# Old Faithful (Azzalini & Bowman 1990), as distributed with R
f = pd.read_csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/faithful.csv")
f[["eruptions", "waiting"]].to_csv(f"{DATA}/old_faithful.csv", index=False)
print("old_faithful.csv", f.shape)

# Mauna Loa monthly CO2, NOAA GML
c = pd.read_csv("https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv", comment="#")
c = c[["year", "month", "decimal date", "average", "deseasonalized"]]
c = c.rename(columns={"decimal date": "decimal_date", "average": "co2", "deseasonalized": "co2_deseasonalized"})
c.to_csv(f"{DATA}/mauna_loa_co2_monthly.csv", index=False)
print("mauna_loa_co2_monthly.csv", c.shape, c.year.min(), c.year.max())

# GISTEMP v4 annual global mean anomaly (J-D column)
g = pd.read_csv("https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv", skiprows=1)
g = g[["Year", "J-D"]].rename(columns={"Year": "year", "J-D": "anomaly"})
g["anomaly"] = pd.to_numeric(g["anomaly"], errors="coerce")
g.dropna().to_csv(f"{DATA}/gistemp_global_annual.csv", index=False)
print("gistemp_global_annual.csv", g.dropna().shape)

# USGS catalogue, M >= 5, 2014-2023
u = pd.read_csv("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
                "&starttime=2014-01-01&endtime=2024-01-01&minmagnitude=5&orderby=time-asc")
u = u[u["type"] == "earthquake"][["time", "latitude", "longitude", "depth", "mag"]]
u["time"] = u["time"].str.slice(0, 19)
u = u.round({"latitude": 3, "longitude": 3, "depth": 1, "mag": 1})
u.to_csv(f"{DATA}/usgs_earthquakes_M5_2014_2023.csv", index=False)
print("usgs_earthquakes_M5_2014_2023.csv", u.shape)

# Pearson (1901) data with York (1966) weights, as tabulated in York et al. (2004)
x = [0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4]
y = [5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5]
wx = [1000, 1000, 500, 800, 200, 80, 60, 20, 1.8, 1.0]
wy = [1, 1.8, 4, 8, 20, 20, 70, 70, 100, 500]
p = pd.DataFrame({"x": x, "y": y, "sigma_x": 1 / np.sqrt(wx), "sigma_y": 1 / np.sqrt(wy)}).round(4)
p.to_csv(f"{DATA}/pearson_york.csv", index=False)
print("pearson_york.csv", p.shape)
