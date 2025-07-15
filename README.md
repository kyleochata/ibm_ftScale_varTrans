# ibm_ftScale_varTrans
IBM Feature Scaling &amp; Variable Transformation

Python project for a simple EDA to work out linear regression models to predict housing prices from Ames Housing Data. In particular to imporve on a baseline set of features via feature engineering: deriving new features from existing data. Feature engineering is often the difference between a weak model and a strong one. 

Basic Concepts to be explored:
- Simple EDA
- One-hot encoding variables
- Log transformation for skewed variables
- Pair plot for features
- Basic feature engineering:
    - Adding polynomial and interaction terms
- Feature Engineering:
    - Categories and features derived from category aggregates

Packages needed are:
- pandas for managing data
- seaborn for data visualization
- reqeusts for data fetching 
- skillsnetwork 

## Set up venv
Create the virtual environment to download needed packages
```bash
    python3 -m venv venv
```
Activate environment:

MAC/Linux
``` 
    source venv/bin/activate 
```
Windows(CMD)
```
    .\venv\Scripts\activate.bat
```
Windows(PowerShell)
```
    .\venv\Scripts\Activate
```

Install packages:
```
    pip isntall pandas seaborn skillsnetwork requests
```

To ensure that the packages are installed run `pip list` in the terminal with the venv active

**Known issue**:
    Be sure to change the interpreter to the venv corresponding interpreter to recognize the downloaded packages