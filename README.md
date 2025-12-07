# ibm-data-science
Code for IBM Data Science Professional on Coursera
## Module 2: Data Wrangling
### Import / Export
| Title                             | Description                                  | Command                                                                       |
|-----------------------------------|----------------------------------------------|-------------------------------------------------------------------------------|
| Read CSV w/o header               | Read CSV file into a DataFrame               | `df = pd.read_csv("file.csv")` or `df = pd.read_csv("file.csv", header=None)` |
| Read CSV with custom column names | Provide your own column names                | `df = pd.read_csv("file.csv", names=cols_list)`                               |
| Read Excel                        | Read Excel sheet into a DataFrame            | `df = pd.read_excel("file.xlsx", sheet_name="Sheet1")`                        |
| Read JSON                         | Read JSON file or URL into a DataFrame       | `df = pd.read_json("file.json")`                                              |
| Read from SQL                     | Read from SQL database using a query         | `df = pd.read_sql("SELECT * FROM table", conn)`                               |
| Save to CSV                       | Save DataFrame to a CSV file                 | `df.to_csv("output.csv", index=False)`                                        |
| Save to Excel                     | Save DataFrame to an Excel file              | `df.to_excel("output.xlsx", index=False)`                                     |
| Save to SQL                       | Write DataFrame to SQL table                 | `df.to_sql("table_name", conn, if_exists="replace", index=False)`             |

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
 Descriptive statistical analysis helps to describe basic features of a dataset and obtains a short summary about the sample and measures of the data.
 - `df.describe()` Using the describe function and applying it on your data frame, a describe function automatically computes basic statistics for all numerical variables. 
It shows the mean, the total number of data points, the standard deviation, the quartiles, and the extreme values. Any NaN values are automatically skipped in these statistics
 - `df['col'].value_counts()` The value_counts function takes a Series as input and returns a Series containing counts of unique values.
 - `df['col'].value_counts().to_frame()` saves it to a new DataFrame.
 - `df.boxplot(column='col1', by='col2')` A boxplot is a method for graphically depicting groups of numerical data through their quartiles.
### GroupBy and Pivot Tables
 - `df['drive-wheels'].unique()` returns an array of unique values in a column.
 - `df.groupby(['col1', 'col2'], as_index=False).mean()` The groupby function groups rows of a DataFrame based on some criteria.
 - `df.pivot(index='col1', columns='col2', values='col3')` The pivot_table function creates a spreadsheet-like table of a DataFrame.
### Seaborn functions
 - `sns.regplot(x = 'header_1',y = 'header_2',data= df)` A regression plot draws a scatter plot of two variables, x and y, and then fits the regression model and plots the resulting regression line along with a 95% confidence interval for that regression. The x and y parameters can be shared as the dataframe headers to be used, and the data frame itself is passed to the function as well.
 - 
### Pearson Correlation Coefficient
`pearson_coef, p_value = stats.pearsonr(df['col1'], df['col2'])` The Pearson Correlation measures the linear dependence between two variables X and Y and will give you two values; the correlation coefficient and the p-value.
- The coefficient is a value between -1 and 1 inclusive, where:
  - 1: Perfect positive linear correlation. 
  - 0: No linear correlation, the two variables most likely do not affect each other. 
  - -1: Perfect negative linear correlation.
- The P-value is the probability value that the correlation between these two variables is statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant. 
  - p-value < 0.001 strong certainty about the correlation coefficient that we calculated
  - p-value < 0.05 moderate certainty
  - p-value < 0.1 weak certainty
  - p-value > 0.1 no certainty of correlation at all

*Strong correlation when*
- Correlation coefficient is close to +1 or -1
- P-value is less than 0.001