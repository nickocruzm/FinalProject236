# Phase 3 Report: Web UI and Database Integration


## Project Overview

Phase 3 implements a web-based user interface for the Hotel Bookings Database, allowing users to explore and interact with three datasets:
- **Hotel Data**: Original hotel bookings from 2015-2016
- **Customer Data**: Customer reservations from 2017-2018
- **Unified Data**: Combined dataset spanning 2015-2018

The application provides an intuitive interface for viewing data with features including pagination, search capabilities, and dynamic filtering.

---

## Technology Stack

### Backend
- **Flask 3.0.0**: Lightweight web framework for Python
- **psycopg2-binary 2.9.9**: PostgreSQL database adapter for Python
- **Python 3.11+**

### Frontend
- **HTML5**: Page structure
- **CSS3**: Styling and responsive design
- **JavaScript**: Interactive features (filter toggle)

### Database
- **PostgreSQL**: Relational database management system

---

## Database Connection Implementation

### Configuration (`config.py`)

The configuration file centralizes all application settings:

**Location**: `Phase3/config.py:1-14`

```python
# PostgreSQL Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'hotel_bookings',
    'user': 'postgres',
    'password': 'postgres'
}

# Flask Application Configuration
DEBUG = True

# Pagination settings
ROWS_PER_PAGE = 50
```

### Database Connection Module (`db.py`)

The `db.py` module provides all database interaction functionality through a set of dedicated functions.

#### 1. Database Connection Function

`Phase3/db.py:6-12`

```python
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        raise
```

**How it works**:
- Uses `psycopg2.connect()` with configuration parameters from `config.py`
- The `**DB_CONFIG` syntax unpacks the dictionary as keyword arguments
- Returns a connection object if successful
- Raises an exception if connection fails, with error logging

#### 2. Retrieving Table Data with Pagination

`Phase3/db.py:15-50`

```python
def get_table_data(table_name, limit=50, offset=0, filters=None):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # Build WHERE clause based on filters
        where_clauses = []
        params = []

        if filters:
            for col, val in filters.items():
                where_clauses.append(f"{col}::text ILIKE %s")
                params.append(f"%{val}%")

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Get total count
        count_query = f"SELECT COUNT(*) as count FROM {table_name} {where_sql}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()['count']

        # Get paginated data
        query = f"SELECT * FROM {table_name} {where_sql} ORDER BY arrival_date, arrival_year LIMIT %s OFFSET %s"
        cursor.execute(query, params + [limit, offset])
        rows = cursor.fetchall()

        return rows, total_count

    except psycopg2.Error as e:
        print(f"Error querying table {table_name}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
```

**How it works**:
- **RealDictCursor**: Returns rows as dictionaries instead of tuples, allowing access by column name
- **Dynamic WHERE clause**: Builds filters dynamically based on user input
- **Parameterized queries**: Uses `%s` placeholders to prevent SQL injection
- **ILIKE operator**: Case-insensitive pattern matching in PostgreSQL
- **Type casting**: `::text` converts all column types to text for consistent filtering
- **Two queries**:
  1. Count query: Gets total matching records for pagination
  2. Data query: Retrieves the actual paginated data
- **Resource management**: Always closes cursor and connection in `finally` block

#### 3. Getting Table Columns

`Phase3/db.py:53-73`

```python
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
```

**How it works**:
- Queries PostgreSQL's `information_schema.columns` system catalog
- Returns columns in their defined order (`ordinal_position`)
- Used to dynamically generate table headers in the UI

#### 4. Getting Column Metadata

`Phase3/db.py:76-154`

This function analyzes each column to determine the appropriate filter type for the UI.

```python
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
                    SELECT DISTINCT {col_name}
                    FROM {table_name}
                    WHERE {col_name} IS NOT NULL
                    ORDER BY {col_name}
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
                    SELECT COUNT(DISTINCT {col_name}) as distinct_count
                    FROM {table_name}
                    WHERE {col_name} IS NOT NULL
                """)
                distinct_count = cursor.fetchone()['distinct_count']

                if distinct_count <= 20:
                    cursor.execute(f"""
                        SELECT DISTINCT {col_name}
                        FROM {table_name}
                        WHERE {col_name} IS NOT NULL
                        ORDER BY {col_name}
                        LIMIT 20
                    """)
                    metadata[col_name]['filter_type'] = 'select'
                    metadata[col_name]['options'] = [row[col_name] for row in cursor.fetchall()]

            elif col_type in ('double precision', 'numeric', 'real'):
                metadata[col_name]['filter_type'] = 'number'

        return metadata
```

**How it works**:
- Analyzes each column's data type and distinct values
- Determines the most appropriate UI filter component:
  - **Date columns**: Date picker
  - **Boolean columns** (0/1): Dropdown with Yes/No
  - **Low cardinality columns** (<= 20 distinct values): Dropdown menu
  - **Numeric columns**: Number input
  - **Text columns**: Text search box
- Returns metadata dictionary used by the template to render appropriate filters

#### 5. Getting Table Statistics

**Location**: `Phase3/db.py:157-211`

```python
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
```

**How it works**:
- Computes statistics displayed on the home page
- **Total rows**: Simple count of all records
- **Cancellation rate**: Average of `is_canceled` column (percentage)
- **Date range**: Minimum and maximum arrival dates
- Checks for column existence before querying to handle different table schemas

---

## Web Application Implementation

### Flask Application (`app.py`)

#### Application Setup

`Phase3/app.py:1-13`

```python
from flask import Flask, render_template, request, jsonify
from db import get_db_connection, get_table_data, get_table_columns, get_table_statistics, get_column_metadata
from config import DEBUG, ROWS_PER_PAGE

app = Flask(__name__)
app.config['DEBUG'] = DEBUG

# Available tables in the database
AVAILABLE_TABLES = {
    'hotel_data': 'Hotel Booking Dataset (2015-2016)',
    'customer_data': 'Customer Reservations Dataset (2017-2018)',
    'unified_data': 'Unified Dataset (2015-2018)'
}
```

- Imports Flask framework and database functions
- Creates Flask application instance
- Defines available tables as a dictionary for easy management

#### Route 1: Home Page

`Phase3/app.py:16-36`

```python
@app.route('/')
def index():
    """
    Home page - displays available datasets and their statistics
    """
    try:
        # Get statistics for each table
        table_stats = {}
        for table_name, description in AVAILABLE_TABLES.items():
            stats = get_table_statistics(table_name)
            table_stats[table_name] = {
                'description': description,
                'stats': stats
            }

        return render_template('index.html',
                             tables=AVAILABLE_TABLES,
                             table_stats=table_stats)

    except Exception as e:
        return render_template('error.html', error=str(e))
```

**How it works**:
- Loops through all available tables
- Retrieves statistics for each table from the database
- Passes data to the template for rendering
- Error handling redirects to error page if database connection fails

#### Route 2: View Table Data

`Phase3/app.py:39-83`

```python
@app.route('/view/<table_name>')
def view_table(table_name):
    """
    View data from a specific table with pagination
    """
    # Validate table name
    if table_name not in AVAILABLE_TABLES:
        return render_template('error.html',
                             error=f"Table '{table_name}' not found")

    try:
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        rows_per_page = request.args.get('rows_per_page', ROWS_PER_PAGE, type=int)
        offset = (page - 1) * rows_per_page

        # Extract filter parameters (anything not page or rows_per_page)
        filters = {}
        for key, value in request.args.items():
            if key not in ('page', 'rows_per_page') and value.strip():
                filters[key] = value

        # Get data and total count
        rows, total_count = get_table_data(table_name, limit=rows_per_page, offset=offset, filters=filters)
        columns = get_table_columns(table_name)
        column_metadata = get_column_metadata(table_name)

        # Calculate pagination info
        total_pages = (total_count + rows_per_page - 1) // rows_per_page

        return render_template('view_data.html',
                             table_name=table_name,
                             table_description=AVAILABLE_TABLES[table_name],
                             columns=columns,
                             column_metadata=column_metadata,
                             rows=rows,
                             current_page=page,
                             total_pages=total_pages,
                             total_count=total_count,
                             rows_per_page=rows_per_page,
                             filters=filters)

    except Exception as e:
        return render_template('error.html', error=str(e))
```

**How it works**:
1. **Validation**: Checks if table name is valid
2. **Pagination parameters**:
   - Extracts `page` from URL query string (defaults to 1)
   - Calculates `offset` for SQL query
3. **Filter extraction**:
   - Loops through all query parameters
   - Excludes pagination parameters
   - Builds filters dictionary
4. **Data retrieval**: Calls database functions to get data, columns, and metadata
5. **Pagination calculation**: Computes total pages using ceiling division
6. **Template rendering**: Passes all data to the view template

#### Route 3: API Endpoint

`Phase3/app.py:87-120`

```python
@app.route('/api/table/<table_name>')
def api_table_data(table_name):

    if table_name not in AVAILABLE_TABLES:
        return jsonify({'error': f"Table '{table_name}' not found"}), 404

    try:
        page = request.args.get('page', 1, type=int)
        rows_per_page = request.args.get('rows_per_page', ROWS_PER_PAGE, type=int)
        offset = (page - 1) * rows_per_page

        rows, total_count = get_table_data(table_name, limit=rows_per_page, offset=offset)

        # Convert rows to list of dicts (handles date serialization)
        data = []
        for row in rows:
            row_dict = dict(row)
            # Convert date objects to strings
            for key, value in row_dict.items():
                if hasattr(value, 'isoformat'):
                    row_dict[key] = value.isoformat()
            data.append(row_dict)

        return jsonify({
            'table': table_name,
            'data': data,
            'total_count': total_count,
            'page': page,
            'rows_per_page': rows_per_page,
            'total_pages': (total_count + rows_per_page - 1) // rows_per_page
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

**How it works**:
- Provides JSON API access to table data
- Serializes date objects to ISO format strings for JSON compatibility
- Returns structured JSON response with data and pagination info

#### Route 4: Health Check

`Phase3/app.py:123-139`

```python
@app.route('/health')
def health_check():
    """
    Health check endpoint to verify database connectivity
    """
    try:
        conn = get_db_connection()
        conn.close()
        return jsonify({
            'status': 'healthy',
            'database': 'connected'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500
```

**How it works**:
- Tests database connectivity without querying data
- Useful for monitoring and debugging
- Returns HTTP 200 if healthy, 500 if unhealthy

---

## Search and Pagination Features

### Pagination Implementation

Pagination is implemented through a combination of backend SQL queries and frontend URL parameters.

#### Backend Pagination Logic

**In `db.py:39`**:
```python
query = f"SELECT * FROM {table_name} {where_sql} ORDER BY arrival_date, arrival_year LIMIT %s OFFSET %s"
cursor.execute(query, params + [limit, offset])
```

**Key concepts**:
- **LIMIT**: Restricts the number of rows returned
- **OFFSET**: Skips the first N rows
- Formula: `offset = (page - 1) × rows_per_page`

#### Frontend Pagination Controls

**In `view_data.html:124-146`**:

The template generates pagination links dynamically:

```html
{% if total_pages > 1 %}
<div class="pagination">
    {% if current_page > 1 %}
    <a href="?page=1&{{ request.query_string.decode().replace('page=' ~ current_page, 'page=1') }}" class="btn">First</a>
    <a href="?{{ request.query_string.decode().replace('page=' ~ current_page, 'page=' ~ (current_page - 1)) }}" class="btn">Previous</a>
    {% endif %}

    <span class="page-numbers">
        {% for page_num in range([1, current_page - 2]|max, [total_pages + 1, current_page + 3]|min) %}
            {% if page_num == current_page %}
                <span class="current-page">{{ page_num }}</span>
            {% else %}
                <a href="?{{ request.query_string.decode().replace('page=' ~ current_page, 'page=' ~ page_num) }}">{{ page_num }}</a>
            {% endif %}
        {% endfor %}
    </span>

    {% if current_page < total_pages %}
    <a href="?{{ request.query_string.decode().replace('page=' ~ current_page, 'page=' ~ (current_page + 1)) }}" class="btn">Next</a>
    <a href="?{{ request.query_string.decode().replace('page=' ~ current_page, 'page=' ~ total_pages) }}" class="btn">Last</a>
    {% endif %}
</div>
{% endif %}
```

**How it works**:
- Displays "First" and "Previous" buttons if not on page 1
- Shows a window of page numbers
- Displays "Next" and "Last" buttons if not on last page
- Preserves filter parameters in URLs using `request.query_string`

