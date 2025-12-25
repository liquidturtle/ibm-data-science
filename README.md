# ibm-data-science

Code for IBM Data Science Professional on Coursera

## Module 2: Data Wrangling

### Import / Export

| Title                             | Description                            | Command                                                                       |
|-----------------------------------|----------------------------------------|-------------------------------------------------------------------------------|
| Read CSV w/o header               | Read CSV file into a DataFrame         | `df = pd.read_csv("file.csv")` or `df = pd.read_csv("file.csv", header=None)` |
| Read CSV with custom column names | Provide your own column names          | `df = pd.read_csv("file.csv", names=cols_list)`                               |
| Read Excel                        | Read Excel sheet into a DataFrame      | `df = pd.read_excel("file.xlsx", sheet_name="Sheet1")`                        |
| Read JSON                         | Read JSON file or URL into a DataFrame | `df = pd.read_json("file.json")`                                              |
| Read from SQL                     | Read from SQL database using a query   | `df = pd.read_sql("SELECT * FROM table", conn)`                               |
| Save to CSV                       | Save DataFrame to a CSV file           | `df.to_csv("output.csv", index=False)`                                        |
| Save to Excel                     | Save DataFrame to an Excel file        | `df.to_excel("output.xlsx", index=False)`                                     |
| Save to SQL                       | Write DataFrame to SQL table           | `df.to_sql("table_name", conn, if_exists="replace", index=False)`             |

### View / Inspect

| Title                | Description                             | Command                      |
|----------------------|-----------------------------------------|------------------------------|
| View first rows      | Show first *n* rows (default 5)         | `df.head(n)`                 |
| View last rows       | Show last *n* rows (default 5)          | `df.tail(n)`                 |
| Random sample        | Random sample of rows                   | `df.sample(n=5)`             |
| Shape                | Number of rows and columns              | `df.shape`                   |
| Column names         | List all column labels                  | `df.columns`                 |
| Data types           | Data types of each column               | `df.dtypes`                  |
| Basic info           | Summary: index, dtypes, non-null counts | `df.info()`                  |
| Describe numeric     | Summary stats for numeric columns       | `df.describe()`              |
| Describe all         | Summary stats for all columns           | `df.describe(include="all")` |
| Count missing values | Missing values per column               | `df.isna().sum()`            |
| Value counts         | Frequency of values in a column         | `df["col"].value_counts()`   |
| Unique values        | Unique values in a column               | `df["col"].unique()`         |

### Data Wrangling

| Package/Method                                 | Description                                                             | Code Example                                                                                                                                                                                                                                                      |
|------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Replace missing data with frequency (mode)** | Replace missing values with the most common (mode) entry in the column. | ```MostFrequentEntry = df['attribute_name'].value_counts().idxmax() df['attribute_name'].replace(np.nan,MostFrequentEntry,inplace=True)```                                                                                                                        |
| **Replace missing data with mean**             | Replace missing values with the mean of the column.                     | ```AverageValue = df['attribute_name'].astype(<data_type>).mean(axis=0) df['attribute_name'].replace(np.nan, AverageValue, inplace=True)```                                                                                                                       |
| **Fix data types**                             | Convert one or more columns to a specific data type.                    | ```df[['attribute1_name', 'attribute2_name', ...]] = df[['attribute1_name', 'attribute2_name', ...]].astype('data_type')\n# data_type can be int, float, str, etc.```                                                                                             |
| **Data Normalization**                         | Normalize a column so values are between 0 and 1.                       | ```df['attribute_name'] = df['attribute_name'] / df['attribute_name'].max()```                                                                                                                                                                                    |
| **Binning**                                    | Create bins for continuous data.                                        | ```bins = np.linspace(min(df['attribute_name']), max(df['attribute_name']), n) # n is number of bins GroupNames = ['Group1', 'Group2', 'Group3', ...] df['binned_attribute_name'] = pd.cut(df['attribute_name'], bins, labels=GroupNames, include_lowest=True)``` |
| **Change column name**                         | Rename a column.                                                        | ```df.rename(columns={'old_name': 'new_name'}, inplace=True)```                                                                                                                                                                                                   |
| **Indicator variables**                        | Convert categorical attributes into one-hot encoded variables.          | ```dummy_variable = pd.get_dummies(df['attribute_name']) df = pd.concat([df, dummy_variable], axis=1)```                                                                                                                                                          |

| Title                             | Description                                              | Command                                                                        |
|-----------------------------------|----------------------------------------------------------|--------------------------------------------------------------------------------|
| Select column                     | Select a single column as Series                         | `df["col"]`                                                                    |
| Select multiple columns           | Select several columns                                   | `df[["col1", "col2"]]`                                                         |
| Filter rows (condition)           | Filter rows matching condition                           | `df[df["col"] > 0]`                                                            |
| Filter with multiple conditions   | Combine conditions with `&` / `\|`                       | `df[(df["col1"] > 0) & (df["col2"] == "A")]`                                   |
| Reset index                       | Reset index to default                                   | `df.reset_index(drop=True, inplace=True)`                                      |
| Sort values                       | Sort by one or more columns                              | `df.sort_values(["col1", "col2"], ascending=[True, False])`                    |
| Rename columns                    | Rename one or more columns                               | `df.rename(columns={"old": "new"}, inplace=True)`                              |
| Drop columns                      | Remove columns                                           | `df.drop(columns=["col1", "col2"], inplace=True)`                              |
| Drop rows with NaN                | Drop rows with missing values                            | `df.dropna(subset=["col"], inplace=True)`                                      |
| Fill missing values               | Replace NaN with a statistic/value                       | `df["col"].fillna(df["col"].mean(), inplace=True)`                             |
| Replace values                    | Replace specific values                                  | `df["col"].replace({0: np.nan}, inplace=True)`                                 |
| Change type                       | Convert column to another dtype                          | `df["col"] = df["col"].astype(float)`                                          |
| Create new column                 | Compute new column from others                           | `df["new"] = df["a"] + df["b"]`                                                |
| Apply function                    | Apply custom function to column                          | `df["col"].apply(my_func)`                                                     |
| Group by                          | Group and aggregate                                      | `df.groupby("col")["value"].mean()`                                            |
| Multiple aggregations             | Apply several aggregations                               | `df.groupby("col")["value"].agg(["mean","sum","count"])`                       |
| Pivot table                       | Create pivot table                                       | `pd.pivot_table(df, values="val", index="row", columns="col", aggfunc="mean")` |
| Concatenate                       | Stack DataFrames vertically or horizontally              | `pd.concat([df1, df2], axis=0)`                                                |
| Merge / join                      | SQL-style join                                           | `pd.merge(df1, df2, on="key", how="inner")`                                    |
| Min-max scaling                   | Normalize features to range 0–1                          | `df_scaled = (df - df.min()) / (df.max() - df.min())`                          |
| Z-score scaling                   | Normalize using mean and std deviation                   | `df_scaled = (df - df.mean()) / df.std()`                                      |
| Simple feature scaling            | Scale features by dividing by std (same as z-score here) | `df_scaled = df / df.max()`                                                    |
| Replace '?' with NaN              | Convert missing value placeholders to actual NaN         | `df = df.replace("?", np.nan)`                                                 |
| Fill missing values with mean     | Replace NaN values using column mean                     | `df = df.fillna(df.mean())`                                                    |
| Fill missing col values with mean | Replace NaN values using column mean                     | `df["col"] = df["col"].fillna(df["col"].mean())`                               |
| Element-wise addition             | Increase all values in a column                          | `df['col'] = df['col'] + 1`                                                    |

## Module 3: Exloratory Data Analysis

Exploratory data analysis or in short, EDA, is an approach to analyze data in order to

- summarize main characteristics of the data
- gain better understanding of the data set
- uncover relationships between different variables
- extract important variables

### Descriptive statistics

Descriptive statistical analysis helps to describe basic features of a dataset and obtains a short summary about the
sample and measures of the data.

- `df.describe()` Using the describe function and applying it on your data frame, a describe function automatically
  computes basic statistics for all numerical variables.
  It shows the mean, the total number of data points, the standard deviation, the quartiles, and the extreme values. Any
  NaN values are automatically skipped in these statistics
- `df['col'].value_counts()` The value_counts function takes a Series as input and returns a Series containing counts of
  unique values.
- `df['col'].value_counts().to_frame()` saves it to a new DataFrame.
- `df.boxplot(column='col1', by='col2')` A boxplot is a method for graphically depicting groups of numerical data
  through their quartiles.

### GroupBy and Pivot Tables

- `df['drive-wheels'].unique()` returns an array of unique values in a column.
- `df.groupby(['col1', 'col2'], as_index=False).mean()` The groupby function groups rows of a DataFrame based on some
  criteria.
- `df.pivot(index='col1', columns='col2', values='col3')` The pivot_table function creates a spreadsheet-like table of a
  DataFrame.

### Seaborn functions

- `sns.regplot(x = 'header_1',y = 'header_2',data= df)` A regression plot draws a scatter plot of two variables, x and
  y, and then fits the regression model and plots the resulting regression line along with a 95% confidence interval for
  that regression. The x and y parameters can be shared as the dataframe headers to be used, and the data frame itself
  is passed to the function as well.
-

### Pearson Correlation Coefficient

The Pearson Correlation measures the linear dependence between two variables X and Y and will give you two values; the
correlation coefficient and the p-value

```python
from scipy import stats

pearson_coef, p_value = stats.pearsonr(df['col1'], df['col2'])
``` 

- The coefficient is a value between -1 and 1 inclusive, where:
    - 1: Perfect positive linear correlation.
    - 0: No linear correlation, the two variables most likely do not affect each other.
    - -1: Perfect negative linear correlation.
- The P-value is the probability value that the correlation between these two variables is statistically significant.
  Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between
  the variables is significant.
    - p-value < 0.001 strong certainty about the correlation coefficient that we calculated
    - p-value < 0.05 moderate certainty
    - p-value < 0.1 weak certainty
    - p-value > 0.1 no certainty of correlation at all

**Strong correlation when**

- Correlation coefficient is close to +1 or -1
- P-value is less than 0.001

# Module 4: Model Development

Process of training or **fit** a model to **predict** a *target* variable based on one or more *predictor* variables.
The output of the training are the model parameters that describe the relationship between the predictor variables and
the target variable.

## Linear Regression (SLR and MLR)

Use a linear model to explain the relationship between one continuous target y and one or more predictor variables x.

**Simple linear regression** refers to using one independent variable and **multiple linear regression** uses multiple
independent variables

```
y = b_0 + b_1 * x

y = b_0 + b_1 * x + b_2 * x_2 + ... + b_n * x_n
```

Where we have variables

- **predictor** (independent) variables: `X = [x_1, x_2, ..., x_n]`
- **target** (dependent) variable: `y`

and parameters

- **intercept**: `b_0`
- **slope coefficients** of the regression: `b_1, b_2, ..., b_n`

In code, we can use the `sklearn.linear_model.LinearRegression` class:

```python
from sklearn.linear_model import LinearRegression

lm = LinearRegression()  # create model using constructor
lm.fit(X, y)
Yhat = lm.predict(X)
```

We use `lm.intercept_` and `lm.coef_` to get the intercept `b_0` and slope `b_1` of the regression line.

## Model Evaluation using Visualization

### Boxplot

The boxplot shows the distribution of data points

```python
df.boxplot(column='col1', by='col2')  # pandas version
sns.boxplot(df[['col1', 'col2']], x='col1', y='col2')  # seaborn version
```

- Median (Q2 / 50th Percentile): The horizontal line inside the box. It represents the middle value of the dataset.
- First Quartile (Q1 / 25th Percentile): The bottom edge of the box. 25% of the data falls below this value.
- Third Quartile (Q3 / 75th Percentile): The top edge of the box. 75% of the data falls below this value.
- Interquartile Range (IQR): The height of the box (Q3 - Q1). This represents the middle 50% of your data.
- Whiskers: The vertical lines extending from the box. They typically extend to 1.5 \times IQR from the edges of the
  box. They show the range of the data excluding outliers.
- Outliers: Any data points that fall outside the whiskers are plotted individually as dots or diamonds. These are
  considered outliers (extreme values).

### Regression Plot

Creates a scatterplot with an optional **linear regression line**. Useful for visualizing the relationship between two
numeric variables and spotting trends.

- Plots data points
- Adds a best-fit linear regression line
- Can show confidence intervals
- Works well for quick correlation inspection

```python
sns.regplot(x="horsepower", y="price", data=df)
```

### Residual Plot

Plots residuals from a **linear regression model**. Useful for checking model assumptions like linearity and
homoscedasticity. It shows differences between observed and predicted values and helps detect non-linear patterns or
heteroscedasticity

```python
sns.residplot(x="horsepower", y="price", data=df)
```

If the points in a residual plot are randomly spread out around the x-axis, then a linear model is appropriate for the
data. If these conditions are not met (e.g. residuals depend on x), consider using a different model.

### Distribution Plot with Kernel Density Estimation (KDE)

Kernel Density Estimation (KDE) plots are a valuable tool for visualizing data distributions by estimating their
probability density function (PDF). These plots are particularly useful in regression analysis for comparing actual and
predicted values. With the deprecation of Seaborn distplot, KDE plots serve as a modern and effective method for
assessing model performance.

```python
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.kdeplot(Y, label='Actual Value', fill=True, color='blue')
sns.kdeplot(Yhat, label='Predicted Values', fill=True, color='red')
plt.xlabel('Target')
plt.ylabel('Density')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
```

## Polynomial Regression and Pipelines

Polynomial regression is a technique for fitting a nonlinear relationship between a dependent variable and one or more
independent variables. Which can go from second-order (quadratic) to higher-order (cubic, quartic, etc.) polynomials.

For **one-dimensional polynomial regression** (one independent variable) defined by

```python
Yhat = b0 + b1 * x + b2 * x ^ 2 + ... + bn * x ^ n
```

We can use numpy's polyfit function:

```python
X_flat = X.to_numpy().flatten()  # polyfit requires 1d array (not pd.Series)
f = np.polyfit(X_flat, Y, 2)  # fit a polynomial of degree 2 -> find coefficients a, b and c of ax^2 + bx + c
p = np.poly1d(f)  # convert coefficients into a polynomial object that can be used to predict values
Yhat = p(X_flat)
```

### Multidimensional polynomial regression

For 1-D input features, using the numpy `polyfit` function is equivalent to using scikit-learn `PolynomialFeatures` plus
`LinearRegression`. However, the polyfit function cannot handle multidimensional data. We use the preprocessing library
in scikit-learn to create a polynomial feature object. The constructor takes the degree of the polynomial as a
parameter. Then we transform the features into a polynomial feature with the fit_transform method.

```python
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2, include_bias=False)
X_poly = pr.fit_transform(X)  # learn the polynomial features from the data and return feature matrix 
```

### Preprocessing

We can use the preprocessing module to simplify many tasks. For example, we can standardize each feature simultaneously.
We import standard scalar. We train the object, fit the scale object, then transform the data into a new data frame on
array x_scale.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_poly)
X_scale = scaler.transform(X_poly)
```

There are more normalization methods available in the preprocessing library as well as other transformations.

### Pipelines

There are many steps to getting a prediction. For example, polynomial transform, normalization, and linear regression.
We can simplify the process using a pipeline, which sequentially perform a series of transformations and a prediction
as the last step.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = X.astype(float)  # convert to float to avoid errors
input = [('polynomial', PolynomialFeatures(degree=2)), ('scale', StandardScaler()), ('model', LinearRegression())]
pipeline = Pipeline(input)  # give input to pipeline constructor to create a pipeline object
pipeline.fit(X, y)  # train pipeline object a.k.a. fit the model to the data
yhat = pipeline.predict(X)  # use pipeline object to make predictions
```

### Measures for In-Sample Evaluation

**Mean Squared Error (MSE)** is a common measure of the regression model quality. It is defined as the average of
the squared residuals (errors). However, there a few things to note:

- A lower MSE does not always imply a better fitted model.
- MSE for an MLR model will be smaller than MSE for a SLR model, since the errors decrease with an increasing number of
  variables in the model
- Polynomial regression models have smaller MSE than linear regression models

```python
from sklearn.metrics import mean_squared_error

mean_squared_error(y_true, y_pred)
```

**R-squared (R^2)** also called coefficient of determination. It is a statistical measure of how close the data are to
the fitted regression line by dividing the MSE of regression line by the MSE of data average
`R_squared = 1 - (mean_squared_error(y_true, y_pred) / mean_squared_error(y_true, y_true.mean()))`
R-squared values range normally from 0 to 1, with

- 1 being the best possible prediction
- close to 0, the model does not fit the data well
- < 0 can be due to overfitting the model

```python
lm.score(X, y)  # Evaluate the model lm using the default scoring metric (R^2)
```

### Prediction and Decision-making

To determine whether a model is correct or not, you should always

- check if predicted values make sense
- visualize with regression plot, residual plot, distribution plot, MSE, and R-squared
- evaluate with numerical measures
- compare between different models

# Module 5: Model Evaluation and Refinement

## Out-of-Sample Evaluation and Cross-Validation

**In-sample evaluation** (as done in the previous module) tells us how well our model fits the data already given to
train it but not how well the trained model can predict new data. The solution is to **test-train-split** the data, use
the in-sample data or **training data** to train the model. The rest of the data, called **test data**, is used as
out-of-sample
data. This data is then used to approximate how the model performs in the real world.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

where `test_size` is the proportion of the dataset to include in the test split and `random_state` is the seed for
random data splitting.

TODO **Generalization Error**

To overcome the generalization error, we can use **cross-validation**, which is a technique for evaluating a
model by splitting the original dataset into smaller subsets called **folds**. The model is trained on k-1 folds and
tested on the remaining fold. The process is repeated k times, with each iteration using a different fold as the test
set. The average performance of the k models is then calculated as the cross-validation error. In python we use the *
*cross_val_score** function:

```python
from sklearn.model_selection import cross_val_score

rcross_scores = cross_val_score(model, X, y, cv=3)  # fit model three times, on different partitions of the data
cv_score = rcross_scores.mean()
cv_std_dev = rcross_scores.std()
```

- `model`: type of model (already initialized) for the cross validation.
- `X`: the predictor variable data
- `y`: the target variable data
- `cv`: Number of folds. cv=3 means the data set is split into three equal partitions.

The function returns an array of scores, one for each partition that was chosen as the testing set. We can average the
result together to estimate out-of-sample R^2 using the mean function in NumPy.

```python
from sklearn.model_selection import cross_val_predict

yhat = cross_val_predict(model, X, y, cv=3)
```

If we want to get the predictions for each fold, we can use the **cross_val_predict** function. Inputs are exactly the
same as for cross_val_score, except that it returns an array of predictions instead of an array of scores. With cv=3, it
fits 3 separate models (one per fold), each on 66% of the data, then predicts on the remaining 33%. Yhat is then made by
stitching together those held-out predictions.

## Overfitting, Underfitting, and Model Selection

- **Underfitting**: The model is not complex enough to capture the true relationship between the features and the
  target.
- **Overfitting**: The model is too flexible and fits the noise rather than the underlying pattern.

To select the right degree of complexity for a model, we need to evaluate the model on both in-sample (**training error
**) and out-of-sample (**test error**) data. Training error usually decreases with increasing model complexity. Test
error usually first decreases and then increases again with increasing model complexity. We pick the complexity that
minimizes the test error.

## Ridge Regression

For models with multiple independent features and ones with polynomial feature extrapolation, it is common to have
collinear combinations of features. Left unchecked, this multicollinearity of features can lead the model to overfit the
training data. To control this, the feature sets are typically regularized using hyperparameters.

**Ridge regression** is the process of regularizing the feature set using the hyperparameter `alpha`. It can be utilized
to regularize and reduce standard errors and avoid over-fitting while using a regression model.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(X_train[['attribute_1', 'attribute_2', ...]])
X_test_pr = pr.fit_transform(x_test[['attribute_1', 'attribute_2', ...]])

ridgemodel = Ridge(alpha=1)  # create ridge regression object
ridgemodel.fit(X_train_pr, y_train)
yhat = ridgemodel.predict(X_test_pr)
```

Alpha can be set to any value between 0 and infinity. A higher alpha value results in a more regularized model. The
right value of alpha is when the test error R-squared is minimal (i.e., sweet spot between enough model complexity but
no
overfitting). A common technique to determine alpha is by applying **grid search**.

## Grid Search

Grid search is a technique for finding the optimal hyperparameters of a model. It involves iterating over a grid of
values for each hyperparameter and selecting the combination that yields the best performance.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1, 5, 10, 20, 30, 100]}
ridgemodel = Ridge()
grid_search = GridSearchCV(ridgemodel, parameters, cv=5)  # configure grid search object with 5-fold cross-validation
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_  # best model instance found by grid search
scores = grid_search.cv_results_  # dictionary of results for all iterations
```

The `grid_search.fit` runs cross-validation for every `alpha` in `parameters`:

- split training data into k folds, train on k-1 folds, validate on the remaining fold
- repeat so each fold is used as validation once and average the validation scores

The `grid_search.best_estimator_` attribute returns the model with the best cross-validation score.
By default, the score is R² unless you pass `scoring=`.