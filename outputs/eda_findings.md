# EDA Findings — Lahore PM2.5 (Training Data)

> All numbers are computed from the training split
> (2019-05-09 to 2023-06-10, 1317 rows, pre-cleaning).
> Test period is not inspected.

---

## Finding 1 — Peak smog month is Dec
Monthly mean PM2.5 peaks in **Dec** at
**263.7 ug/m3**, more than double
the annual mean.  This aligns with the confluence of post-harvest
crop-residue burning (Punjab paddy harvest) and the onset of winter
thermal inversions that trap pollutants near the surface.

## Finding 2 — Smog season mean is 2.77x the off-season mean
October–January (smog season) mean PM2.5:
**199.7 ug/m3**.
February–September (off-season) mean:
**72.2 ug/m3**.
The smog season mean is **2.77x** higher.
During the smog season
**22.2%** of days fall
in the Hazardous (AQI 301-400) category, versus
**1.5%** in the off-season.

## Finding 3 — Weekday effect is minor (16.0 ug/m3 spread)
Mean PM2.5 is highest on **Wed**
(122.2 ug/m3) and lowest on
**Sun** (106.2 ug/m3),
a range of only **16.0 ug/m3**.  The small weekday
signal suggests traffic and industrial activity contribute marginally
compared with the dominant seasonal and meteorological drivers.

## Finding 4 — Strong lag-1 autocorrelation (0.852)
Pearson correlation between PM2.5[t] and PM2.5[t-1] is
**0.852** (calendar-day lag, computed
only on contiguous pairs).  By lag 7 it falls to
**0.644**, indicating
that air quality has a strong day-to-day persistence — a persistent
high-pollution episode typically lasts several days before dispersing.

## Finding 5 — Strongest weather correlate is temperature_2m_mean
Among temperature, humidity, wind speed and precipitation, the variable
most correlated with PM2.5 is
**temperature_2m_mean**
(r = **-0.606**).
Full weather correlations:
  - temperature_2m_mean: r = -0.606
  - relative_humidity_2m_mean: r = +0.229
  - wind_speed_10m_max: r = -0.379
  - precipitation_sum: r = -0.207

## Finding 6 — 2021 and 2022 show the most data gaps
21 gaps longer than 3 days span the full dataset, with the largest being
the 47-day outage (2021-10-22 to 2021-12-07) during peak smog season.
This concentrates missingness exactly when extreme values would be expected,
so models should be evaluated with awareness that some of the worst smog
events are absent from training data.

## Finding 7 — Low PM2.5 in monsoon months (Jul-Aug)
The month-year heatmap shows consistently low PM2.5 in July and August
across all years.  Monsoon rainfall provides wet deposition of particles
and increased mixing height, temporarily clearing the air despite continued
vehicle and industrial emissions.
