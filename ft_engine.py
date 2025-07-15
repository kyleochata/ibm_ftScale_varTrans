# # Suppress warnings
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
import utils
# set the theme of seaborn to the default
sns.set_theme()


url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data.tsv'
path = 'data/Ames_Housing_Data.tsv'
os.makedirs("data", exist_ok=True)
utils.download(url, path)

df = pd.read_csv(path, sep="\t")
print(df.info())

# Removal of outliers
# filter rows that "Gr Liv Area" are <= 4000; : - select every column (can change it to only return certain cols)
df = df.loc[df['Gr Liv Area'] <= 4000,:]
#df = df.loc[df['Gr Liv Area'] <= 4000,["Lot Area"]]
print("Number of rows in data:", df.shape[0])
print(f"Number of columns in the data: {df.shape[1]}")
# Store copy of original data after outlier removal
data = df.copy()
print(df.head())

# One-hot encoding for dummy variables
# Get pd.Series consisting of all the string categoricals
one_hot_encode_cols = df.dtypes[df.dtypes == object]
one_hot_encode_cols = one_hot_encode_cols.index.tolist()
# T - transpose method. Swap row and cols
print(df[one_hot_encode_cols].head().T)

# Convert categorical variables to dummy
df = pd.get_dummies(df, columns=one_hot_encode_cols, drop_first=True)
print(df.describe().T)

# Log transforming skew variables
#Create list of float columns to check for skewing
mask = data.dtypes == float
float_cols = data.columns[mask]
skew_limit = 0.75 
skew_vals = data[float_cols].skew()
# Show skewed cols
skew_cols = (skew_vals.sort_values(ascending=False)
             .to_frame() # change the Series to DataFrame
             .rename(columns={0: "Skew"})
             .query(f'abs(Skew) > {skew_limit}'))
print(skew_cols)

field = "BsmtFin SF 1"
# Create two "subplots" and a "figure" using matplotlib
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))

# Histogram on ax_before subplot
df[field].hist(ax=ax_before)



# Apply log transformation (numpy syntax) to this column
df[field].apply(np.log1p).hist(ax=ax_after)
ax_before.set(title="before np.log1p", ylabel="frequency", xlabel="value")
ax_after.set(title="after np.log1p", ylabel="frequency", xlabel="value")
fig.suptitle(f"Field {field}")

os.makedirs("plots", exist_ok=True)
utils.savePlot("plots/bsmtFinSF1_nplog1p.png")

# Perform skew transformation: fixes non-normal(skewed) distributions by compressing large values and expanding small values
# makes data more symmetric not necessarily normalized/standardized
# normalized = scale fts to a common range; standardized = zero mean/unit variance
# Skew (Log transforma) - Reduce the effects of outliers on data set; Changes the shape of data by bringing outliers closer to majority
# Normalization/Standardization - changes the numbers on axis while preserving the shape

for col in skew_cols.index.values:
    if col == "SalePrice": # ensure SalePrice col isn't affected
        continue
    # takes the raw val of the column and finds the ln(1+rawColVal)
    # log1p gracefully handles ln(0) (can lead to infinites)
    df[col] = df[col].apply(np.log1p)
# Now have a larger set of potentially useful features
print(df.shape)