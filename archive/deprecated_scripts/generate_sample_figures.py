import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
fig_dir = os.path.join(root, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# File paths
business_path = os.path.join(root, 'data', 'raw', 'business_registry.csv')
storefronts_path = os.path.join(root, 'data', 'raw', 'Storefronts_Reported_Vacant_or_Not_20250915.csv')
dob_path = os.path.join(root, 'data', 'raw', 'DOB_Permit_Issuance_20250915.csv')

outputs = []

# 1) Business Registry: borough distribution and top business types
try:
    df_bus = pd.read_csv(business_path, low_memory=False)
    borough_col = None
    for c in df_bus.columns:
        if 'boro' in c.lower() or 'borough' in c.lower():
            borough_col = c
            break
    if borough_col:
        counts = df_bus[borough_col].fillna('UNKNOWN').value_counts()
        plt.figure(figsize=(6,4))
        counts.plot(kind='bar', color='teal')
        plt.title('Business Registry: Counts by Borough')
        plt.xlabel('Borough')
        plt.ylabel('Number of Licenses')
        out = os.path.join(fig_dir, 'business_borough_counts.png')
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        outputs.append(out)
    type_col = None
    for c in df_bus.columns:
        if any(k in c.lower() for k in ['type', 'industry', 'licens']):
            type_col = c
            break
    if type_col:
        top = df_bus[type_col].fillna('UNKNOWN').value_counts().head(10)
        plt.figure(figsize=(6,4))
        top.plot(kind='barh', color='coral')
        plt.title('Top 10 Business Types / Licenses')
        plt.xlabel('Count')
        plt.tight_layout()
        out2 = os.path.join(fig_dir, 'business_top_types.png')
        plt.savefig(out2)
        plt.close()
        outputs.append(out2)
except Exception as e:
    print('Business Registry plotting failed:', e)

# 2) Storefronts: vacancy over time (monthly)
try:
    df_sf = pd.read_csv(storefronts_path, low_memory=False)
    date_col = None
    for c in df_sf.columns:
        lc = c.lower()
        if 'date' in lc or 'reported' in lc or 'vacant on' in lc:
            date_col = c
            break
    if date_col is None:
        for c in df_sf.columns:
            if 'vacant' in c.lower() and 'on' in c.lower():
                date_col = c
                break
    if date_col:
        df_sf['vac_date'] = pd.to_datetime(df_sf[date_col], errors='coerce')
        df_sf['month'] = df_sf['vac_date'].dt.to_period('M')
        monthly = df_sf.groupby('month').size()
        if len(monthly)>0:
            monthly.index = monthly.index.to_timestamp()
            plt.figure(figsize=(8,3))
            monthly.plot(marker='o')
            plt.title('Vacant Storefronts Reports: Monthly Count')
            plt.xlabel('Month')
            plt.ylabel('Reports')
            plt.tight_layout()
            out = os.path.join(fig_dir, 'storefronts_monthly_reports.png')
            plt.savefig(out)
            plt.close()
            outputs.append(out)
    else:
        bcol = None
        for c in df_sf.columns:
            if 'boro' in c.lower():
                bcol = c
                break
        if bcol:
            counts = df_sf[bcol].fillna('UNKNOWN').value_counts()
            plt.figure(figsize=(6,4))
            counts.plot(kind='bar', color='purple')
            plt.title('Storefront Vacancy Reports by Borough')
            plt.tight_layout()
            out = os.path.join(fig_dir, 'storefronts_borough_counts.png')
            plt.savefig(out)
            plt.close()
            outputs.append(out)
except Exception as e:
    print('Storefronts plotting failed:', e)

# 3) DOB Permits: aggregated permits per year (chunked)
try:
    sample = pd.read_csv(dob_path, nrows=5000)
    date_col = None
    for c in sample.columns:
        if any(k in c.lower() for k in ['file', 'date', 'issued', 'filing']):
            date_col = c
            break
    if date_col is None:
        for c in sample.columns:
            if 'filing' in c.lower():
                date_col = c
                break
    year_counts = {}
    if date_col:
        for chunk in pd.read_csv(dob_path, chunksize=200000, usecols=[date_col]):
            try:
                chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
                yrs = chunk[date_col].dt.year
                vc = yrs.value_counts()
                for y,v in vc.items():
                    year_counts[y] = year_counts.get(y,0) + int(v)
            except Exception:
                pass
        if year_counts:
            yf = pd.Series(year_counts).sort_index()
            plt.figure(figsize=(6,3))
            yf.plot(marker='o', color='green')
            plt.title('DOB Permits: Permits per Year (aggregate)')
            plt.xlabel('Year')
            plt.ylabel('Permit Count')
            plt.tight_layout()
            out = os.path.join(fig_dir, 'dob_permits_by_year.png')
            plt.savefig(out)
            plt.close()
            outputs.append(out)
    else:
        print('DOB date column not found; skipping DOB')
except Exception as e:
    print('DOB plotting failed:', e)

print('Saved outputs:', outputs)
