from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "data/clean/"
HOTEL_CSV = f"{DATA_PATH}hotel_df.csv"
CUSTOMER_CSV = f"{DATA_PATH}customer_df.csv"
MERGED_CSV = f"{DATA_PATH}merged_df.csv"

# PostgreSQL connection details
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"
POSTGRES_DB = "hotel_bookings"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"

# JDBC URL for PostgreSQL connection
JDBC_URL = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# ============================================================================
# INITIALIZE SPARK SESSION
# ============================================================================
print("\n" + "="*70)
print("INITIALIZING SPARK SESSION")
print("="*70)

spark = SparkSession.builder \
    .appName("Hotel Bookings Database Population") \
    .config("spark.jars", "/Users/nickocruz/Downloads/postgresql-42.7.3.jar") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.host", "localhost") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Spark session initialized successfully")

# Connection properties for PostgreSQL
jdbc_properties = {
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "driver": "org.postgresql.Driver"
}

# ============================================================================
# LOAD CLEANED DATASETS
# ============================================================================
print("\n" + "="*70)
print("LOADING CLEANED DATASETS")
print("="*70)

try:
    # Load Hotel Booking dataset
    print("\nLoading hotel_df.csv...")
    hotel_df = spark.read.csv(HOTEL_CSV, header=True, inferSchema=True)
    hotel_count = hotel_df.count()
    print(f"Hotel dataset loaded: {hotel_count:,} rows")
    
    # Load Customer Reservations dataset
    print("\nLoading customer_df.csv...")
    customer_df = spark.read.csv(CUSTOMER_CSV, header=True, inferSchema=True)
    customer_count = customer_df.count()
    print(f"Customer dataset loaded: {customer_count:,} rows")
    
    # Load Merged/Unified dataset
    print("\nLoading merged_df.csv...")
    merged_df = spark.read.csv(MERGED_CSV, header=True, inferSchema=True)
    merged_count = merged_df.count()
    print(f"Merged dataset loaded: {merged_count:,} rows")
    
except Exception as e:
    print(f"\nERROR loading datasets: {str(e)}")
    sys.exit(1)

# ============================================================================
# PREPARE DATASETS FOR DATABASE INSERTION
# ============================================================================
print("\n" + "="*70)
print("PREPARING DATASETS FOR DATABASE")
print("="*70)

# Drop the index column
if "_c0" in hotel_df.columns:
    hotel_df = hotel_df.drop("_c0")
    print("Dropped index column from hotel dataset")

if "_c0" in customer_df.columns:
    customer_df = customer_df.drop("_c0")
    print("Dropped index column from customer dataset")

if "_c0" in merged_df.columns:
    merged_df = merged_df.drop("_c0")
    print("Dropped index column from merged dataset")

# Show schema information
print("\n--- Hotel Dataset Schema ---")
print(f"Columns: {', '.join(hotel_df.columns)}")

print("\n--- Customer Dataset Schema ---")
print(f"Columns: {', '.join(customer_df.columns)}")

print("\n--- Merged Dataset Schema ---")
print(f"Columns: {', '.join(merged_df.columns)}")

# ============================================================================
# POPULATE POSTGRESQL DATABASE
# ============================================================================
print("\n" + "="*70)
print("POPULATING POSTGRESQL DATABASE")
print("="*70)

try:
    # Table 1: Hotel Booking Data
    print("\nWriting hotel_data table")
    hotel_df.write \
        .mode("overwrite") \
        .jdbc(url=JDBC_URL, table="hotel_data", properties=jdbc_properties)
    print(f"hotel_data table populated with {hotel_count:,} rows")
    
    # Table 2: Customer Reservations Data
    print("\nWriting customer_data table")
    customer_df.write \
        .mode("overwrite") \
        .jdbc(url=JDBC_URL, table="customer_data", properties=jdbc_properties)
    print(f"customer_data table populated with {customer_count:,} rows")
    
    # Table 3: Unified/Merged Data
    print("\nWriting unified_data table")
    merged_df.write \
        .mode("overwrite") \
        .jdbc(url=JDBC_URL, table="unified_data", properties=jdbc_properties)
    print(f"unified_data table populated with {merged_count:,} rows")
    
except Exception as e:
    print(f"\nERROR writing to database: {str(e)}")
    sys.exit(1)

# ============================================================================
# VERIFY DATABASE POPULATION
# ============================================================================
print("\n" + "="*70)
print("VERIFYING DATABASE POPULATION")
print("="*70)

try:
    # Verify each table
    print("\nReading back from database to verify")
    
    hotel_verify = spark.read \
        .jdbc(url=JDBC_URL, table="hotel_data", properties=jdbc_properties)
    print(f"hotel_data: {hotel_verify.count():,} rows")
    
    customer_verify = spark.read \
        .jdbc(url=JDBC_URL, table="customer_data", properties=jdbc_properties)
    print(f"customer_data: {customer_verify.count():,} rows")
    
    unified_verify = spark.read \
        .jdbc(url=JDBC_URL, table="unified_data", properties=jdbc_properties)
    print(f"unified_data: {unified_verify.count():,} rows")
    
    print("\n" + "="*70)
    print("DATABASE POPULATION COMPLETED SUCCESSFULLY")
    print("="*70)
    
    print("\nSummary:")
    print(f"Hotel dataset: {hotel_count:,} rows")
    print(f"Customer dataset: {customer_count:,} rows")
    print(f"Unified dataset: {merged_count:,} rows")
    print(f"Total records: {hotel_count + customer_count:,} rows")

    
except Exception as e:
    print(f"\nWARNING: Could not verify database: {str(e)}")
    print("  Data may have been written, but verification failed")

# Stop Spark session
spark.stop()
print("\nSpark session stopped")