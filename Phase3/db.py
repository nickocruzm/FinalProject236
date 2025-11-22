import psycopg2
from psycopg2.extras import RealDictCursor
from config import DB_CONFIG


def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        raise


def get_table_data(table_name, limit=50, offset=0):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # Get total count
        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
        total_count = cursor.fetchone()['count']

        # Get paginated data
        query = f"SELECT * FROM {table_name} ORDER BY arrival_date, arrival_year LIMIT %s OFFSET %s"
        cursor.execute(query, (limit, offset))
        rows = cursor.fetchall()

        return rows, total_count

    except psycopg2.Error as e:
        print(f"Error querying table {table_name}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def get_table_columns(table_name):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        cursor.execute(query, (table_name,))
        columns = [row[0] for row in cursor.fetchall()]
        return columns

    except psycopg2.Error as e:
        print(f"Error getting columns for {table_name}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def get_table_statistics(table_name):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        stats = {}

        # Total rows
        cursor.execute(f"SELECT COUNT(*) as total FROM {table_name}")
        stats['total_rows'] = cursor.fetchone()['total']

        # Cancellation rate if is_canceled column exists
        cursor.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s AND column_name = 'is_canceled'
        """, (table_name,))

        if cursor.fetchone():
            cursor.execute(f"""
                SELECT
                    ROUND(AVG(is_canceled) * 100, 2) as cancellation_rate
                FROM {table_name}
            """)
            result = cursor.fetchone()
            stats['cancellation_rate'] = result['cancellation_rate']

        # Date range if arrival_date column exists
        cursor.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s AND column_name = 'arrival_date'
        """, (table_name,))

        if cursor.fetchone():
            cursor.execute(f"""
                SELECT
                    MIN(arrival_date) as min_date,
                    MAX(arrival_date) as max_date
                FROM {table_name}
            """)
            result = cursor.fetchone()
            stats['date_range'] = {
                'min': result['min_date'],
                'max': result['max_date']
            }

        return stats

    except psycopg2.Error as e:
        print(f"Error getting statistics for {table_name}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
