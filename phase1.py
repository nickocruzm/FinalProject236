"""
Phase 1: Data Preparation & EDA in Spark
CS 236 - Database Management Systems
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions as F
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Hotel Bookings EDA") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.host", "localhost") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\nLoading datasets...")

# Load the two CSV files
df1 = spark.read.csv("hotel-booking.csv", header=True, inferSchema=True)
df2 = spark.read.csv("customer-reservations.csv", header=True, inferSchema=True)

print(f"Hotel Booking loaded: {df1.count()} rows")
print(f"Customer Reservations loaded: {df2.count()} rows")

print("\n--- Hotel Booking Schema ---")
df1.printSchema()

print("\n--- Customer Reservations Schema ---")
df2.printSchema()

# Compare column names
cols1 = set(df1.columns)
cols2 = set(df2.columns)

print("\n--- Column Comparison ---")
print(f"Hotel Booking has {len(cols1)} columns")
print(f"Customer Reservations has {len(cols2)} columns")
print(f"Common columns: {len(cols1.intersection(cols2))}")



# FIND UNIQUE COLUMNS
unique_to_1 = cols1 - cols2
unique_to_2 = cols2 - cols1

if unique_to_1:
    print(f"\nUnique to Hotel Booking ({len(unique_to_1)} columns):")
    for column in sorted(unique_to_1):
        print(f"  - {column}")

if unique_to_2:
    print(f"\nUnique to Customer Reservations ({len(unique_to_2)} columns):")
    for column in sorted(unique_to_2):
        print(f"  - {column}")



# NULL VALUE ANALYSIS
print("\nNull Value Analysis...")

def analyze_nulls(df, name):
    """Analyze null values in the dataset"""
    print(f"\n--- {name} Null Analysis ---")
    
    # Calculate null counts for all columns
    null_counts = []
    for c in df.columns:
        null_count = df.filter(col(c).isNull()).count()
        null_counts.append((c, null_count))
    
    # Filter and sort columns with nulls
    columns_with_nulls = [(c, n) for c, n in null_counts if n > 0]
    columns_with_nulls.sort(key=lambda x: x[1], reverse=True)
    
    total_rows = df.count()
    
    if columns_with_nulls:
        print(f"Columns with null values: {len(columns_with_nulls)}/{len(df.columns)}")
        print(f"\n{'Column':<30} {'Null Count':>12} {'Percentage':>12}")
        print("-" * 56)
        for col_name, null_count in columns_with_nulls:
            pct = (null_count / total_rows) * 100
            print(f"{col_name:<30} {null_count:>12,} {pct:>11.2f}%")
    else:
        print("No null values found!")

analyze_nulls(df1, "Hotel Booking")
analyze_nulls(df2, "Customer Reservations")



# DISTINCT VALUE ANALYSIS
print("\nDistinct Value Analysis...")

def analyze_distinct_values(df, name):
    """Analyze distinct values for all columns"""
    print(f"\n--- {name} Distinct Values ---")
    print(f"\n{'Column':<30} {'Distinct Values':>18} {'Uniqueness %':>15}")
    print("-" * 65)
    
    total_rows = df.count()
    
    for c in df.columns:
        distinct_count = df.select(c).distinct().count()
        uniqueness = (distinct_count / total_rows) * 100
        print(f"{c:<30} {distinct_count:>18,} {uniqueness:>14.2f}%")

analyze_distinct_values(df1, "Hotel Booking")
analyze_distinct_values(df2, "Customer Reservations")



# DATA TYPE ANALYSIS
print("\nData Type Analysis...")

def analyze_data_types(df, name):
    """Show data types distribution"""
    print(f"\n--- {name} Data Types ---")
    
    type_counts = {}
    for col_name, col_type in df.dtypes:
        type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    print(f"\n{'Data Type':<20} {'Count':>10}")
    print("-" * 32)
    for dtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{dtype:<20} {count:>10}")

analyze_data_types(df1, "Hotel Bookings")
analyze_data_types(df2, "Customer Reservations")



# DATA TYPE MISMATCHES
print("\nChecking for Data Type Mismatches...")

def compare_column_types(df1, df2):
    """Compare data types of common columns"""
    common_cols = sorted(set(df1.columns).intersection(set(df2.columns)))
    
    print(f"\nComparing {len(common_cols)} common columns...")
    
    df1_types = dict(df1.dtypes)
    df2_types = dict(df2.dtypes)
    
    mismatches = []
    matches = []
    
    for col_name in common_cols:
        type1 = df1_types[col_name]
        type2 = df2_types[col_name]
        
        if type1 != type2:
            mismatches.append((col_name, type1, type2))
        else:
            matches.append(col_name)
    
    if mismatches:
        print(f"\nFound {len(mismatches)} type mismatches:")
        print(f"\n{'Column':<30} {'Hotel Bookings Type':<20} {'Customer Reservations Type':<20}")
        print("-" * 72)
        for col_name, type1, type2 in mismatches:
            print(f"{col_name:<30} {type1:<20} {type2:<20}")
    else:
        print("\nAll common columns have matching types!")
    
    print(f"\n{len(matches)} columns have matching types")
    
    return mismatches

type_mismatches = compare_column_types(df1, df2)


# ==== END SESION ====
print("\nClosing Spark session...")
try:
    spark.stop()
except:
    pass

# Force clean exit on Windows
os._exit(0)