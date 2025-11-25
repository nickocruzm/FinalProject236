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


def get_table_data(table_name, limit=50, offset=0, filters=None):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # Build WHERE clause based on filters
        where_clauses = []
        params = []

        if filters:
            for col, val in filters.items():
                where_clauses.append(f'"{col}"::text ILIKE %s')
                params.append(f"%{val}%")

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Get total count
        count_query = f"SELECT COUNT(*) as count FROM {table_name} {where_sql}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()['count']

        # Get paginated data
        query = f'SELECT * FROM {table_name} {where_sql} ORDER BY "arrival_date", "arrival_year" LIMIT %s OFFSET %s'
        cursor.execute(query, params + [limit, offset])
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


def get_column_metadata(table_name):

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        metadata = {}

        # Get column names and types
        cursor.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))

        columns_info = cursor.fetchall()

        for col_info in columns_info:
            col_name = col_info['column_name']
            col_type = col_info['data_type']

            metadata[col_name] = {
                'type': col_type,
                'filter_type': 'text'  # default
            }

            # Determine filter type based on data type and values
            if col_type == 'date':
                metadata[col_name]['filter_type'] = 'date'

            elif col_type in ('integer', 'bigint', 'smallint'):
                # Check if it's a boolean-like column (only 0 and 1)
                cursor.execute(f"""
                    SELECT DISTINCT "{col_name}"
                    FROM {table_name}
                    WHERE "{col_name}" IS NOT NULL
                    ORDER BY "{col_name}"
                """)
                distinct_vals = [row[col_name] for row in cursor.fetchall()]

                if set(distinct_vals).issubset({0, 1}):
                    metadata[col_name]['filter_type'] = 'boolean'
                    metadata[col_name]['options'] = distinct_vals
                elif len(distinct_vals) <= 20:  # Low cardinality
                    metadata[col_name]['filter_type'] = 'select'
                    metadata[col_name]['options'] = distinct_vals

            elif col_type in ('text', 'character varying'):
                # Check cardinality
                cursor.execute(f"""
                    SELECT COUNT(DISTINCT "{col_name}") as distinct_count
                    FROM {table_name}
                    WHERE "{col_name}" IS NOT NULL
                """)
                distinct_count = cursor.fetchone()['distinct_count']

                if distinct_count <= 20:
                    cursor.execute(f"""
                        SELECT DISTINCT "{col_name}"
                        FROM {table_name}
                        WHERE "{col_name}" IS NOT NULL
                        ORDER BY "{col_name}"
                        LIMIT 20
                    """)
                    metadata[col_name]['filter_type'] = 'select'
                    metadata[col_name]['options'] = [row[col_name] for row in cursor.fetchall()]

            elif col_type in ('double precision', 'numeric', 'real'):
                metadata[col_name]['filter_type'] = 'number'

        return metadata

    except psycopg2.Error as e:
        print(f"Error getting column metadata for {table_name}: {e}")
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
                    ROUND(AVG("is_canceled") * 100, 2) as cancellation_rate
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
                    MIN("arrival_date") as min_date,
                    MAX("arrival_date") as max_date
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
