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


if __name__ == '__main__':
    print("\n" + "="*70)
    print("STARTING HOTEL BOOKINGS WEB UI")
    print("="*70)
    print("\nAvailable datasets:")
    for table_name, description in AVAILABLE_TABLES.items():
        print(f"  - {table_name}: {description}")
    print("\nAccess the application at: http://localhost:5000")
    print("="*70 + "\n")

    app.run(debug=DEBUG, host='0.0.0.0', port=5000)
