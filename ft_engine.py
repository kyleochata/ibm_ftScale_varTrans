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
from sklearn import preprocessing
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
path = "plots/bsmtFinSF1_nplog1p.png"
if not utils.checkIfPlotExists(path):
    utils.savePlot(path)
    

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

df = data
print(f"og_data missing val: {data.isnull().sum().sort_values()}")

smaller_df= df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond', 
                      'Year Built', 'Year Remod/Add', 'Gr Liv Area', 
                      'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 
                      'Garage Cars','SalePrice']]

print(smaller_df.describe().T)
print(smaller_df.info())
# fill missing values with 0
smaller_df = smaller_df.fillna(0)
print(smaller_df.info())

# Pairplot to better visualize target and feature-target relationship
sns.pairplot(smaller_df, plot_kws=dict(alpha=.1, edgecolor="none"))
path = "plots/smaller_df.png"
if not utils.checkIfPlotExists(path):
    utils.savePlot(path)


# separate features from target
X = smaller_df.loc[:, ['Lot Area', 'Overall Qual', 'Overall Cond', 
                      'Year Built', 'Year Remod/Add', 'Gr Liv Area', 
                      'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 
                      'Garage Cars']]
Y = smaller_df['SalePrice']
print("xxxxxxxx")
print(X.info())

# Basic feature engineering: Adding polynomial and interaction terms
# In 'Overall Qual' and 'Gr Liv Area' theres an upward-curved relationship rather than simple linear correspondence -> polynomial terms or transformations
#  allow for expression of non-linear relationships while still using linear regression as the model

# Polynomial Features
X2 = X.copy()
X2['OQ2'] = X2['Overall Qual'] ** 2
X2['GLA2'] = X2['Gr Liv Area'] ** 2
print(X2)
plt.figure(figsize=(12, 6))
sns.regplot(x=X['Overall Qual'], y=Y, order=2, scatter_kws={'alpha':0.3})
plt.title('Overall Quality (with Quadratic Trendline) vs SalePrice')
plt.xlabel('Overall Quality (1-10)')
plt.ylabel('Sale Price ($)')
path = "plots/oq2_sp.png"
if not utils.checkIfPlotExists(path):
    utils.savePlot(path)

# Feature Interactions
# interaction effects: impact of one feature may depend on the current value of a different feature
X3 = X2.copy()
# multiplicative interaction
X3['OQ_x_YB'] = X3["Overall Qual"] * X3["Year Built"]
X3["OQ_/_LA"] = X3["Overall Qual"] / X3["Lot Area"]
print(X3)

# Incorporate categorical features into linear regression models
# Create a new feature column for each category value and fill with 0 | 1 == one-hot-encoding / dummy variable mthods
print(data["House Style"].value_counts)
print(pd.get_dummies(df["House Style"], drop_first=True).head())

nbh_counts = df["Neighborhood"].value_counts()
print(nbh_counts)

# map few least-represented neighborhoods to a "other" category before adding the feature to our feature set and running a new benchmark
other_nbhs = list(nbh_counts[nbh_counts <= 8].index)
print(other_nbhs)

X4 = X3.copy()
X4["Neighborhood"] = df["Neighborhood"].replace(other_nbhs, "Other")

# create fts that capture where a ft value lies relative to the members of a category it belongs to
# Deviance of a row's ft value from the mean value of the category that row belongs to (i.e how nice a house is relative to others in its neighborhood with the same style)
def add_deviation_feature(X, feature, category):
    
    # temp groupby object
    category_gb = X.groupby(category)[feature]
    
    # create category means and standard deviations for each observation
    category_mean = category_gb.transform(lambda x: x.mean())
    category_std = category_gb.transform(lambda x: x.std())
    
    # compute stds from category mean for each feature value,
    # add to X as new feature
    deviation_feature = (X[feature] - category_mean) / category_std 
    X[feature + '_Dev_' + category] = deviation_feature

X5 = X4.copy()
X5['House Style'] = df['House Style']
add_deviation_feature(X5, 'Year Built', 'House Style')
add_deviation_feature(X5, 'Overall Qual', 'Neighborhood')
print(X5.head())

# sklearn allows for building many higher-order terms at once with PolynomialFeatures

# Instantiate and provide desired degree;
pf = preprocessing.PolynomialFeatures(degree=2)
features = ["Lot Area", "Overall Qual"]
# fit and transform data
pf.fit(df[features])
feat_array = pf.transform(df[features])
# Get ft names and create DataFrame
ft_names = pf.get_feature_names_out(input_features=features)
poly_df = pd.DataFrame(feat_array, columns = ft_names)
print(poly_df.head())