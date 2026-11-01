"""
=====================================================================
 DMV PRACTICAL 1
 Title   : Analyzing Sales Data from Multiple File Formats
 Aim     : Load sales data from CSV, Excel and JSON files, clean it,
           combine it into one table, analyze it and visualize it.
 Dataset : sales_2003.csv, sales_2004.xlsx, sales_2005.json
           (one year of the same sales data in each file format)
=====================================================================
"""

import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")          # hide library warnings for clean output
pd.set_option("display.width", 200)        # print wide tables on one line
pd.set_option("display.max_columns", 12)   # limit columns shown in the terminal

# Folder of this script, so the data files are found from anywhere
FOLDER = os.path.dirname(os.path.abspath(__file__))


def heading(text):
    """Print a clear section heading in the terminal."""
    print("\n" + "=" * 70)
    print(" " + text)
    print("=" * 70)


heading("DMV PRACTICAL 1 : ANALYZING SALES DATA FROM MULTIPLE FILE FORMATS")

# ------------------------------------------------------------------
# STEP 1 : Load the data from CSV, Excel and JSON files
# ------------------------------------------------------------------
heading("STEP 1 : Load data from CSV, Excel and JSON files")

csv_data = pd.read_csv(os.path.join(FOLDER, "sales_2003.csv"))      # CSV file
excel_data = pd.read_excel(os.path.join(FOLDER, "sales_2004.xlsx"))  # Excel file
json_data = pd.read_json(os.path.join(FOLDER, "sales_2005.json"))    # JSON file

print("CSV   file (rows, columns) :", csv_data.shape)
print("Excel file (rows, columns) :", excel_data.shape)
print("JSON  file (rows, columns) :", json_data.shape)

# ------------------------------------------------------------------
# STEP 2 : Explore the data
# ------------------------------------------------------------------
heading("STEP 2 : Explore the data")

print("First 5 rows of the CSV data:")
print(csv_data.head())
print("\nColumn names:")
print(list(csv_data.columns))

# ------------------------------------------------------------------
# STEP 3 : Combine all three files into one unified table
# ------------------------------------------------------------------
heading("STEP 3 : Combine CSV + Excel + JSON into one table")

sales = pd.concat([csv_data, excel_data, json_data], ignore_index=True)
print("Unified data (rows, columns):", sales.shape)
print()
sales.info()

# ------------------------------------------------------------------
# STEP 4 : Check data quality
# ------------------------------------------------------------------
heading("STEP 4 : Check data quality (missing values and duplicates)")

missing = sales.isnull().sum()
print("Columns with missing values:")
print(missing[missing > 0])                 # show only columns that have gaps
print("\nDuplicate rows:", sales.duplicated().sum())

# ------------------------------------------------------------------
# STEP 5 : Clean and standardize the data
# ------------------------------------------------------------------
heading("STEP 5 : Clean and standardize the data")

sales = sales.drop_duplicates()             # remove repeated rows

# Fill missing text values instead of deleting useful rows
text_cols = ["ADDRESSLINE2", "STATE", "POSTALCODE", "TERRITORY"]
sales[text_cols] = sales[text_cols].fillna("Unknown")

sales["POSTALCODE"] = sales["POSTALCODE"].astype(str)   # one type for all codes
sales["STATUS"] = sales["STATUS"].str.strip()           # remove extra spaces
sales["ORDERDATE"] = pd.to_datetime(sales["ORDERDATE"], format="%m/%d/%Y %H:%M")  # text -> date

print("Missing values left :", sales.isnull().sum().sum())
print("ORDERDATE type now  :", sales["ORDERDATE"].dtype)

# ------------------------------------------------------------------
# STEP 6 : Transform the data
# ------------------------------------------------------------------
heading("STEP 6 : Transform the data (add useful columns)")

sales["YEAR_MONTH"] = sales["ORDERDATE"].dt.to_period("M")   # e.g. 2003-02
print(sales[["ORDERNUMBER", "ORDERDATE", "YEAR_MONTH", "SALES"]].head())

# ------------------------------------------------------------------
# STEP 7 : Descriptive analysis
# ------------------------------------------------------------------
heading("STEP 7 : Descriptive analysis of sales")

order_totals = sales.groupby("ORDERNUMBER")["SALES"].sum()   # total value of each order

print(f"Total sales          : {sales['SALES'].sum():,.2f}")
print(f"Number of orders     : {sales['ORDERNUMBER'].nunique()}")
print(f"Average order value  : {order_totals.mean():,.2f}")
print(f"Average sale per line: {sales['SALES'].mean():,.2f}")

print("\nSummary statistics:")
print(sales[["QUANTITYORDERED", "PRICEEACH", "SALES", "MSRP"]].describe())

print("\nTotal sales by year:")
print(sales.groupby("YEAR_ID")["SALES"].sum().round(2))

product_sales = sales.groupby("PRODUCTLINE")["SALES"].sum().sort_values(ascending=False)
print("\nTotal sales by product line:")
print(product_sales.round(2))

order_counts = sales["PRODUCTLINE"].value_counts()
print("\nNumber of order lines by product line:")
print(order_counts)

print("\nTop 5 countries by sales:")
print(sales.groupby("COUNTRY")["SALES"].sum().sort_values(ascending=False).head().round(2))

# ------------------------------------------------------------------
# STEP 8 : Visualizations
# ------------------------------------------------------------------
heading("STEP 8 : Visualizations")

# 1. Number of orders per product line
plt.figure(figsize=(10, 5))
order_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Product Category Distribution")
plt.xlabel("Product Line")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()

# 2. Total sales per product line
plt.figure(figsize=(10, 5))
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title("Total Sales by Product Line")
plt.xlabel("Product Line")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()

# 3. Monthly sales trend
monthly_sales = sales.groupby("YEAR_MONTH")["SALES"].sum()
plt.figure(figsize=(12, 5))
monthly_sales.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.tight_layout()

# 4. Top 10 orders by total value
plt.figure(figsize=(10, 4))
order_totals.sort_values(ascending=False).head(10).plot(kind="bar", color="orange")
plt.title("Top 10 Orders by Total Value")
plt.xlabel("Order Number")
plt.ylabel("Order Value")
plt.tight_layout()

# 5. Distribution of sales amounts
plt.figure(figsize=(8, 4))
sns.histplot(sales["SALES"], kde=True, bins=30)
plt.title("Histogram of SALES")
plt.xlabel("Sales Amount")
plt.ylabel("Frequency")

# 6. Box plot to spot outliers in sales
plt.figure(figsize=(8, 4))
sns.boxplot(x=sales["SALES"])
plt.title("Boxplot of SALES")
plt.xlabel("Sales Amount")

# 7. Relationship between quantity and sales
plt.figure(figsize=(8, 5))
sns.scatterplot(x="QUANTITYORDERED", y="SALES", data=sales)
plt.title("Quantity Ordered vs Sales")
plt.xlabel("Quantity Ordered")
plt.ylabel("Sales")

# 8. Share of orders per product line
plt.figure(figsize=(7, 7))
plt.pie(order_counts, labels=order_counts.index, autopct="%1.1f%%", startangle=140)
plt.title("Orders Distribution by Product Line")
plt.axis("equal")

print("Opening all graphs... close the graph windows to finish.")
plt.show()
