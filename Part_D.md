prompt: Explain EDA steps using Pandas with examples
LLM: kimi
Output:


# Exploratory Data Analysis (EDA) with Pandas

Exploratory Data Analysis is the critical first step in any data science project. It helps you understand your data's structure, identify patterns, detect anomalies, and form hypotheses. Here's a comprehensive guide using Pandas.

## 1. **Data Loading & Initial Inspection**

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# First look - shape and info
print(f"Dataset shape: {df.shape}")  # (rows, columns)
print(f"\nColumn names: {df.columns.tolist()}")

# Basic info about data types and non-null counts
df.info()
```

**Example Output:**
```
Dataset shape: (891, 12)

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64  # ← Missing values!
```

## 2. **Quick Statistical Overview**

```python
# Descriptive statistics for numerical columns
df.describe()

# Include all columns (even categorical)
df.describe(include='all')

# Specific percentiles
df.describe(percentiles=[0.01, 0.05, 0.95, 0.99])
```

**Key insights from `.describe()`:**
- **Count**: Missing values detection (if count varies across columns)
- **Mean/Median**: Central tendency and potential skewness
- **Min/Max**: Data range and potential outliers
- **Std**: Variability in the data

## 3. **Missing Values Analysis**

```python
# Check missing values
missing = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_table = pd.concat([missing, missing_percent], axis=1, 
                          keys=['Missing Count', 'Missing %'])
print(missing_table[missing_table['Missing Count'] > 0])

# Visualize missing values pattern
import missingno as msno
msno.matrix(df)
msno.heatmap(df)  # Correlation of missingness between columns
```

**Common strategies:**
```python
# Drop if < 5% missing
df.dropna(subset=['small_missing_col'], inplace=True)

# Impute numerical with median (robust to outliers)
df['Age'].fillna(df['Age'].median(), inplace=True)

# Impute categorical with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

## 4. **Data Type Conversion & Cleaning**

```python
# Convert data types
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')  # Memory efficient
df['is_active'] = df['is_active'].map({'yes': 1, 'no': 0})

# Remove duplicates
print(f"Duplicates: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# Clean text data
df['name'] = df['name'].str.strip().str.title()
```

## 5. **Univariate Analysis** (Single Variable)

### Numerical Columns
```python
# Distribution analysis
df['Age'].hist(bins=30, edgecolor='black')
df['Age'].plot(kind='box')

# Key statistics
print(f"Skewness: {df['Age'].skew():.2f}")
print(f"Kurtosis: {df['Age'].kurtosis():.2f}")

# Value counts for discrete numerical
df['Pclass'].value_counts().sort_index()
```

### Categorical Columns
```python
# Frequency analysis
df['Sex'].value_counts()
df['Sex'].value_counts(normalize=True) * 100  # Percentages

# Top categories
df['Cabin'].value_counts().head(10)

# Unique values count
print(f"Unique cabins: {df['Cabin'].nunique()}")
```

## 6. **Bivariate Analysis** (Two Variables)

### Numerical vs Numerical
```python
# Correlation matrix
correlation = df[['Age', 'Fare', 'SibSp', 'Parch']].corr()

# Scatter plot
df.plot.scatter(x='Age', y='Fare', alpha=0.5)

# Group statistics
df.groupby('Pclass')['Fare'].agg(['mean', 'median', 'std', 'count'])
```

### Categorical vs Numerical
```python
# Groupby analysis
df.groupby('Sex')['Age'].mean()

# Multiple aggregations
df.groupby('Pclass').agg({
    'Age': ['mean', 'min', 'max'],
    'Fare': ['mean', 'sum'],
    'Survived': 'mean'  # Survival rate
})

# Box plot by category
df.boxplot(column='Fare', by='Pclass')
```

### Categorical vs Categorical
```python
# Cross-tabulation (contingency table)
pd.crosstab(df['Sex'], df['Survived'], margins=True)

# With percentages
pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100

# Chi-square test for independence
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(pd.crosstab(df['Sex'], df['Survived']))
print(f"P-value: {p:.4f}")  # < 0.05 indicates significant association
```

## 7. **Multivariate Analysis**

```python
# Pivot tables
pivot = df.pivot_table(values='Survived', 
                       index='Sex', 
                       columns='Pclass', 
                       aggfunc='mean')
print(pivot)

# Multiple groupby
df.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack()

# Correlation heatmap for all numerical columns
import seaborn as sns
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', center=0)
```

## 8. **Outlier Detection**

```python
# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df['Fare']))
outliers_z = df[z_scores > 3]

# IQR method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['Fare'] < Q1 - 1.5*IQR) | (df['Fare'] > Q3 + 1.5*IQR)]

print(f"Outliers (IQR method): {len(outliers_iqr)} rows")
```

## 9. **Feature Engineering During EDA**

```python
# Create new features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Binning
df['AgeGroup'] = pd.cut(df['Age'], 
                        bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

# Extract datetime features
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month
df['DayOfWeek'] = df['date'].dt.day_name()
```

## 10. **Complete EDA Function Template**

```python
def quick_eda(df):
    """Comprehensive EDA summary"""
    print("="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n" + "="*50)
    print("MISSING VALUES")
    print("="*50)
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
    
    print("\n" + "="*50)
    print("DATA TYPES")
    print("="*50)
    print(df.dtypes)
    
    print("\n" + "="*50)
    print("NUMERICAL SUMMARY")
    print("="*50)
    print(df.describe())
    
    print("\n" + "="*50)
    print("CATEGORICAL COLUMNS")
    print("="*50)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head(5))
    
    print("\n" + "="*50)
    print("POTENTIAL ISSUES")
    print("="*50)
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print(f"Constant columns: {constant_cols}")
    
    # Check for high cardinality
    high_card = [col for col in cat_cols if df[col].nunique() > 100]
    if high_card:
        print(f"High cardinality columns (>100 unique): {high_card}")

# Usage
quick_eda(df)
```

## Summary Table: EDA Checklist

| Step | Pandas Function | Purpose |
|------|----------------|---------|
| Load | `pd.read_csv()` | Import data |
| Shape | `.shape` | Dimensions |
| Types | `.info()`, `.dtypes` | Data types |
| Stats | `.describe()` | Distribution |
| Missing | `.isnull().sum()` | Data completeness |
| Unique | `.nunique()`, `.value_counts()` | Cardinality |
| Correlation | `.corr()` | Relationships |
| Group | `.groupby()` | Segment analysis |
| Pivot | `.pivot_table()` | Multi-dimensional |
| Sample | `.head()`, `.tail()`, `.sample()` | Manual inspection |

---

**Pro Tips:**
1. Always start with `.head()` and `.info()` — they reveal immediate issues
2. Use `.sample(n)` instead of `.head()` for a more representative view
3. Check for data leakage (target variables in features) during EDA
4. Document your findings — EDA informs all downstream modeling decisions

Would you like me to demonstrate any of these techniques on a specific dataset, or dive deeper into visualization techniques?

Evaluate correctness:

