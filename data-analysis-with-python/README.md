# IBM Data Science – Data Analysis with Python

This repository contains **notes, explanations, and a practical cheatsheet** created while completing the **IBM Data
Science Professional Certificate** and focuses on:

- Data Wrangling with Pandas
- Exploratory Data Analysis (EDA)
- Model Development
- Model Evaluation & Refinement

---

## Tech Stack

- **Python**
- **Pandas / NumPy** – data manipulation
- **Matplotlib / Seaborn** – visualization
- **SciPy** – statistics
- **Scikit-Learn** – modeling and evaluation

---

## 1. Data Wrangling (Pandas)

Data wrangling is the process of **loading, inspecting, cleaning, transforming, and preparing data** for analysis and
modeling.

### Import & Export Data

Load data from common formats into a Pandas DataFrame:

```python
pd.read_csv("file.csv")
pd.read_excel("file.xlsx", sheet_name="Sheet1")
pd.read_json("file.json")
pd.read_sql("SELECT * FROM table", conn)
```

Key parameters:

- `header=None` – CSV without header row
- `names=[...]` – define custom column names
- `sheet_name` – select Excel sheet

Save DataFrames:

```python
df.to_csv("out.csv", index=False)
df.to_excel("out.xlsx", index=False)
df.to_sql("table", conn, if_exists="replace", index=False)
```

- `index=False` prevents writing the index as a column
- `if_exists="replace"` overwrites existing SQL tables

---

### Inspecting Data

Understanding structure and quality of the data:

```python
df.head()
df.tail()
df.shape
df.columns
df.dtypes
df.info()
df.describe(include="all")  # include categorical columns
```

Common checks:

- `df.info()` → non-null counts and data types
- `df.describe()` → summary statistics for numerical columns
- `df.isna().sum()` → missing values per column

---

### Cleaning Missing & Invalid Data

Replace placeholders and handle missing values:

```python
df.replace("?", np.nan, inplace=True)
df["col"].fillna(df["col"].mean(), inplace=True)
df.dropna(subset=["col"], inplace=True)
```

Strategies:

- **Mean** → numerical data
- **Mode** → categorical data
- **Drop rows** if missing values are rare and unimportant

### Type Conversion & Renaming

Correct data types are critical for analysis and modeling:

```python
df["col"] = df["col"].astype(float)
df.rename(columns={"old": "new"}, inplace=True)
```

Panda data types:

- float64, int64, bool
- object, datetime64[ns], timedelta64[ns]

---

### Filtering & Selection

Select and filter rows and columns:

```python
df["col"]  # single column as Series
df[["col1", "col2"]]  # multiple columns as DataFrame
df[df["col"] > 0]  # boolean mask
df[(df["a"] > 0) & (df["b"] == "A")]  # complex conditions
```

- Use `&` and `|` instead of `and` / `or`
- Wrap conditions in parentheses

---

### Feature Engineering

Create new variables from existing data:

```python
df["new"] = df["a"] + df["b"]
```

**One-hot encoding** is a technique used to **convert categorical variables into numerical features** so they can be
used by machine learning models. Most ML algorithms cannot work directly with text or category labels and require
numeric input. Instead of assigning arbitrary numbers (which would imply an order), one-hot encoding creates **binary
indicator columns**. This returns a new DataFrame containing one column per category with 0/1 values.

```python
pd.get_dummies(df["category"])
```

### Binning

Binning is a feature engineering technique that **groups continuous numerical values into discrete intervals (bins)**. 
It is commonly used to **reduce noise, simplify analysis, and reveal patterns** by converting continuous data into categorical ranges (e.g. low, medium, high).

```python
bins = [0, 20, 50, 100]
groups = ["Low", "Medium", "High"]

pd.cut(df["col"], bins=bins, labels=groups) # returns Series with num values replaced by labels
```

---

### Aggregation & Reshaping

#### Group by

```python
df.groupby("col")["value"].mean() # group by col and calculate mean of value
df.groupby("col")["value"].agg(["mean", "sum", "count"]) # multiple aggregations of value

# group by multiple columns and aggregate using different functions
df.groupby(["col1", "col2"]).agg({"value1": "sum", "value2": "mean"})
```

#### Pivot Tables

```python
# simple pivot table with default aggregation
pd.pivot_table(df, values="val", index="row", columns="col")

# compute sum of sales and quantity by region and product for each year
pd.pivot_table(
    df,
    values=["sales", "quantity"],
    index=["region", "product"],
    columns="year",
    aggfunc="sum"
)
```
- Input data must be long / tidy (**Each row represents one observation, and each column represents one variable.**)
- Missing combinations → NaN
- Supports mean (default), sum, count, median, min, max, etc.

#### Merge & Concatenate
```python
pd.merge(df1, df2, on="key", how="inner") # Performs SQL-style joins between two DataFrames using a key column.
pd.concat([df1, df2])
```

---

## 2. Exploratory Data Analysis (EDA)

EDA is used to **understand distributions, relationships, and patterns** before modeling.

### Descriptive Statistics

```python
df.describe()
df["col"].value_counts()
```

Boxplots show distributions and outliers:

```python
df.boxplot(column="x", by="y")
```

---

### Correlation Analysis (Pearson)

Measures **linear dependency** between two variables:

```python
from scipy import stats

pearson_coef, p_value = stats.pearsonr(df["x"], df["y"])
```

Interpretation:

- Correlation close to **±1** → strong linear relationship
- `p_value < 0.05` → statistically significant

---

### Visualization (Seaborn)

Regression plot (relationship + trend):

```python
sns.regplot(x="x", y="y", data=df)
```

Residual plot (model assumptions):

```python
sns.residplot(x="x", y="y", data=df)
```

Random scatter around zero → linear model is appropriate.

---

## 3. Model Development

Model development involves **training a model to predict a target variable** using one or more predictors.

### Linear Regression (SLR & MLR)

```text
y = b0 + b1 * x
y = b0 + b1 * x1 + b2 * x2 + ...
```

Implementation:

```python
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X, y)
yhat = lm.predict(X)
```

Key attributes:

- `lm.intercept_` → intercept
- `lm.coef_` → feature coefficients

---

### Polynomial Regression

Used when the relationship is **non-linear**.

```python
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2, include_bias=False)
X_poly = pr.fit_transform(X)
```

Higher degree → more flexibility but higher risk of overfitting.

---

### Pipelines

Pipelines chain preprocessing and modeling steps:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("scale", StandardScaler()),
    ("model", LinearRegression())
])

pipeline.fit(X, y)
pipeline.predict(X)
```

Benefits:

- Cleaner code
- Prevents data leakage
- Easier evaluation

---

## 4. Model Evaluation

### Evaluation Metrics

Mean Squared Error (MSE):

```python
from sklearn.metrics import mean_squared_error

mean_squared_error(y, yhat)
```

R² score:

```python
lm.score(X, y)
```

---

### Train / Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)
```

---

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
scores.mean()
```

---

## 5. Regularization & Model Refinement

### Ridge Regression

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)
ridge.predict(X_test)
```

---

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

params = {"alpha": [0.001, 0.01, 1, 10, 100]}
grid = GridSearchCV(Ridge(), params, cv=5)
grid.fit(X_train, y_train)

grid.best_estimator_
```

---

## Purpose of This Repository

- Learning notes for the IBM Data Science course
- Practical Pandas & ML reference
- Fast recall during analysis or interviews  
