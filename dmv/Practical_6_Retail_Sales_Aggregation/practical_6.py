"""
=====================================================================
 DMV PRACTICAL 6
 Title   : Data Aggregation
 Problem : Analyzing Sales Performance by Region in a Retail Company
 Aim     : Group sales by region (shopping mall) and product category,
           find the top-performing regions and visualize the results.
 Dataset : customer_shopping_data.csv (Istanbul malls, kept in this folder)
           'price' = total amount of the sale in Turkish Lira
=====================================================================
"""

import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")          # hide library warnings for clean output
pd.set_option("display.width", 200)        # print wide tables on one line
pd.set_option("display.max_columns", 12)   # limit columns shown in the terminal

# Folder of this script, so the dataset is found from anywhere
FOLDER = os.path.dirname(os.path.abspath(__file__))


def heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(" " + text)
    print("=" * 70)


heading("DMV PRACTICAL 6 : SALES PERFORMANCE BY REGION (DATA AGGREGATION)")

# ------------------------------------------------------------------
# STEP 1 : Load and explore the dataset
# ------------------------------------------------------------------
heading("STEP 1 : Load and explore the dataset")

df = pd.read_csv(os.path.join(FOLDER, "customer_shopping_data.csv"))
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nShape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# ------------------------------------------------------------------
# STEP 2 : Identify the variables used for aggregation
# ------------------------------------------------------------------
heading("STEP 2 : Variables used for aggregation")

print("Region           -> shopping_mall :", df["shopping_mall"].nunique(), "malls")
print("Product category -> category      :", df["category"].nunique(), "categories")
print("Sales amount     -> price")
print("Quantity sold    -> quantity")

# ------------------------------------------------------------------
# STEP 3 : Count of records by region and by category
# ------------------------------------------------------------------
heading("STEP 3 : Number of transactions by region and category")

print("Transactions per mall:")
print(df.groupby("shopping_mall")["invoice_no"].count())
print("\nTransactions per category:")
print(df.groupby("category")["invoice_no"].count())

# ------------------------------------------------------------------
# STEP 4 : Total sales by region (mall)
# ------------------------------------------------------------------
heading("STEP 4 : Total sales by region")

# numeric_only=True adds only number columns (not text like names)
branch_sales = df.groupby("shopping_mall").sum(numeric_only=True)[["quantity", "price"]]
branch_sales = branch_sales.sort_values(by="price", ascending=False)
print(branch_sales.round(2))

# ------------------------------------------------------------------
# STEP 5 : Top-performing regions
# ------------------------------------------------------------------
heading("STEP 5 : Top-performing regions")

print("Top 3 malls by total sales:")
for rank, (mall, row) in enumerate(branch_sales.head(3).iterrows(), start=1):
    print(f"  {rank}. {mall:18s} {row['price']:>15,.2f} TL")

share = (branch_sales["price"] / branch_sales["price"].sum() * 100).round(2)
print("\nShare of total sales (%):")
print(share)

# ------------------------------------------------------------------
# STEP 6 : Total sales by product category
# ------------------------------------------------------------------
heading("STEP 6 : Total sales by product category")

category_sales = df.groupby("category").sum(numeric_only=True)[["quantity", "price"]]
category_sales = category_sales.sort_values(by="price", ascending=False)
print(category_sales.round(2))

# ------------------------------------------------------------------
# STEP 7 : Sales by region AND product category
# ------------------------------------------------------------------
heading("STEP 7 : Sales by region and product category")

combined = df.groupby(["shopping_mall", "category"])["price"].sum()
print(combined.head(16).round(2))

# Pivot table: malls as rows, categories as columns
combined_pivot = df.pivot_table(index="shopping_mall", columns="category",
                                values="price", aggfunc="sum")
print("\nBest-selling category in each mall:")
print(combined_pivot.idxmax(axis=1))

# ------------------------------------------------------------------
# STEP 8 : Visualizations
# ------------------------------------------------------------------
heading("STEP 8 : Visualizations")

# 1. Total sales by mall (bar chart)
plt.figure(figsize=(10, 6))
(branch_sales["price"] / 1e6).plot(kind="bar", color="steelblue")
plt.title("Total Sales by Region (Shopping Mall)")
plt.xlabel("Shopping Mall")
plt.ylabel("Sales (Million TL)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# 2. Sales share by mall (pie chart)
plt.figure(figsize=(8, 8))
plt.pie(branch_sales["price"], labels=branch_sales.index, autopct="%1.1f%%")
plt.title("Sales Share by Mall")

# 3. Sales share by category (pie chart)
plt.figure(figsize=(8, 8))
plt.pie(category_sales["price"], labels=category_sales.index, autopct="%1.1f%%")
plt.title("Sales Share by Product Category")

# 4. Grouped bar chart: categories inside each mall
(combined_pivot / 1e6).plot(kind="bar", figsize=(12, 6))
plt.title("Sales of Each Category at Each Mall")
plt.xlabel("Shopping Mall")
plt.ylabel("Sales (Million TL)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()

# 5. Stacked bar chart: total sales split by category
(combined_pivot / 1e6).plot(kind="bar", stacked=True, figsize=(12, 6))
plt.title("Stacked Sales by Mall and Category")
plt.xlabel("Shopping Mall")
plt.ylabel("Sales (Million TL)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()

print("Opening all graphs... close the graph windows to finish.")
plt.show()
