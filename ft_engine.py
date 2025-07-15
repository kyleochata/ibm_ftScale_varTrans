# # Suppress warnings
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

import pandas as pd
import seaborn as sns
import skillsnetwork
import os
import requests

# set the theme of seaborn to the default
sns.set_theme()

def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        # with ensures automatic closing after write
        with open(filename, "wb") as f: # create/open file; "wb": w = write file, b = binary mode(req for non-text files to prevent corruption)
            f.write(response.content)

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data.tsv'
path = 'data/Ames_Housing_Data.tsv'
os.makedirs("data", exist_ok=True)
download(url, path)

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
