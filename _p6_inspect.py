import pandas as pd, json, sys, os
sys.path.insert(0, r"C:/Users/abdul/OneDrive/Documents/Sem5/AI Project/lahore-smog")

df = pd.read_csv(
    r"C:/Users/abdul/OneDrive/Documents/Sem5/AI Project/lahore-smog/data/processed/lahore_merged.csv",
    parse_dates=["date"], index_col="date"
).sort_index()

# lag4/5/6 are built on-the-fly; in CSV we have lag1/2/3 and roll7
first_full = df.dropna(subset=["pm25_lag1","pm25_lag2","pm25_lag3","pm25_roll7"]).index.min()
print("First date with lag1-3 + roll7:", first_full.date())

n = len(df); n_train=int(n*0.70); n_val=int(n*0.15)
test = df.iloc[n_train+n_val:]
print("Test period:", test.index.min().date(), "to", test.index.max().date())
print("Test rows:", len(test))

# demo dates
for d in ["2024-11-01","2024-11-15","2024-12-01"]:
    ts = pd.Timestamp(d)
    if ts in df.index:
        r = df.loc[ts]
        print(f"{d}: pm25={r.pm25_ugm3:.1f} ug/m3, lag1={r.pm25_lag1:.1f}, roll7={r.pm25_roll7:.1f}, temp={r.temperature_2m_mean:.1f}C")

print("\nModels:")
for f in sorted(os.listdir(r"C:/Users/abdul/OneDrive/Documents/Sem5/AI Project/lahore-smog/outputs/models")):
    print(" ", f)

# Check EDA and model_comparison JSONs exist
for jf in ["eda_summary.json","model_comparison.json","kpis.json"]:
    p = r"C:/Users/abdul/OneDrive/Documents/Sem5/AI Project/lahore-smog/outputs/results/" + jf
    exists = os.path.exists(p)
    print(f"  {jf}: {'EXISTS' if exists else 'MISSING'}")