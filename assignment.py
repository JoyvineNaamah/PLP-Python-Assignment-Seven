# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris()
    # Convert the dataset to a DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})  # Map species to names
except Exception as e:
    print("Error loading the dataset:", e)

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Clean the dataset (if applicable)
df = df.dropna()
print("\nDataset cleaned (if applicable).")

# Compute basic statistics for numerical columns
print("\nBasic Statistics:")
print(df.describe())

# Grouping by 'species' and calculating the mean for each group
grouped = df.groupby('species').mean()
print("\nMean values for each species:")
print(grouped)

# Insights: Identify patterns or findings
print("\nInteresting Findings:")
print("- Setosa species has smaller petal lengths and widths compared to the other species.")
print("- Virginica species tends to have the largest petal and sepal dimensions.")

# Set the visual style for Seaborn
sns.set(style="whitegrid")

# 1. Line Chart: Simulating a trend of petal length over index
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['petal length (cm)'], color='blue', label='Petal Length')
plt.title("Line Chart: Petal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart: Average petal length per species
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='petal length (cm)', data=df, palette='muted')
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram: Distribution of Sepal Length
plt.figure(figsize=(8, 6))
plt.hist(df['sepal length (cm)'], bins=20, color='green', edgecolor='black')
plt.title("Histogram: Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot: Petal Length vs Sepal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='deep')
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
