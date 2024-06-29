import os
import requests
from datetime import datetime
import psycopg2
from psycopg2 import sql

# API configuration
API_BASE_URL = "http://localhost:8000"  # URL of our mock API

# Database configuration
DB_NAME = os.getenv('DB_NAME', 'adapted_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'Abbie2504')
DB_HOST = os.getenv('DB_HOST', 'localhost')

def fetch_data(endpoint, params=None):
    """Fetch data from the API"""
    response = requests.get(f"{API_BASE_URL}{endpoint}", params=params)
    response.raise_for_status()
    return response.json()

def fetch_all_pages(endpoint, params=None):
    """Fetch all pages of data from a paginated API endpoint"""
    if params is None:
        params = {}
    all_results = []
    offset = 0
    limit = 100
    
    while True:
        params.update({'offset': offset, 'limit': limit})
        data = fetch_data(endpoint, params)
        results = data.get('results', [])
        all_results.extend(results)
        
        if len(results) < limit:
            break
        
        offset += limit
    
    return all_results

def fetch_users():
    return fetch_all_pages("/learn/api/public/v1/users")

def fetch_courses():
    return fetch_all_pages("/learn/api/public/v1/courses")

def fetch_course_contents(course_id):
    contents = fetch_data(f"/learn/api/public/v1/courses/{course_id}/contents")['results']
    return [
        {
            'id': content['id'],
            'course_id': course_id,
            'title': content['title'],
            'mime_type': content.get('mime_type'),
            'body': content.get('body', 'No description available')
        } for content in contents
    ]

def fetch_course_users(course_id):
    return fetch_data(f"/learn/api/public/v1/courses/{course_id}/users")['results']

def fetch_course_grades(course_id):
    # Note: In a real scenario, we'd need to fetch the column IDs first
    column_id = "mock_column_id"
    return fetch_data(f"/learn/api/public/v2/courses/{course_id}/gradebook/columns/{column_id}/users")['results']

def create_tables(conn):
    """Create necessary tables in the database"""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(255) PRIMARY KEY,
                username VARCHAR(255),
                student_id VARCHAR(255),
                given_name VARCHAR(255),
                family_name VARCHAR(255),
                email VARCHAR(255)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                id VARCHAR(255) PRIMARY KEY,
                course_id VARCHAR(255),
                name VARCHAR(255),
                description TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS contents (
                id VARCHAR(255) PRIMARY KEY,
                course_id VARCHAR(255),
                title VARCHAR(255),
                mime_type VARCHAR(255),
                body TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS enrollments (
                user_id VARCHAR(255),
                course_id VARCHAR(255),
                PRIMARY KEY (user_id, course_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS grades (
                user_id VARCHAR(255),
                course_id VARCHAR(255),
                score FLOAT,
                PRIMARY KEY (user_id, course_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255),
                content_id VARCHAR(255),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (content_id) REFERENCES contents(id)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_content ON interactions(user_id, content_id)
        """)
    conn.commit()

def insert_data(conn, table_name, data):
    """Insert data into a specified table"""
    if not data:
        return
    
    columns = data[0].keys()
    values = [tuple(item[column] for column in columns) for item in data]
    
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("INSERT INTO {} ({}) VALUES {} ON CONFLICT DO NOTHING").format(
                sql.Identifier(table_name),
                sql.SQL(', ').join(map(sql.Identifier, columns)),
                sql.SQL(', ').join([sql.Placeholder()] * len(data))
            ),
            values
        )
    conn.commit()

def main():
    # Establish database connection
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
    
    try:
        # Create tables
        create_tables(conn)
        
        # Fetch and store users
        users = fetch_users()
        insert_data(conn, 'users', [
            {
                'id': user['id'],
                'username': user['userName'],
                'student_id': user['studentId'],
                'given_name': user['name']['given'],
                'family_name': user['name']['family'],
                'email': user['contact']['email']
            } for user in users
        ])
        
        # Fetch and store courses
        courses = fetch_courses()
        insert_data(conn, 'courses', [
            {
                'id': course['id'],
                'course_id': course['courseId'],
                'name': course['name'],
                'description': course['description']
            } for course in courses
        ])
        
        # Fetch and store course contents, enrollments, and grades
        for course in courses:
            course_id = course['id']
            
            contents = fetch_course_contents(course_id)
            print(f"Sample content for course {course_id}:")
            print(contents[0] if contents else "No contents")
            
            insert_data(conn, 'contents', [
                {
                    'id': content['id'],
                    'course_id': course_id,
                    'title': content['title'],
                    'mime_type': content.get('mime_type'),
                    'body': content['body']
                } for content in contents
            ])
            
            enrollments = fetch_course_users(course_id)
            insert_data(conn, 'enrollments', [
                {
                    'user_id': enrollment['userId'],
                    'course_id': course_id
                } for enrollment in enrollments
            ])
            
            grades = fetch_course_grades(course_id)
            insert_data(conn, 'grades', [
                {
                    'user_id': grade['userId'],
                    'course_id': course_id,
                    'score': grade['score']
                } for grade in grades
            ])
        
        print("Data collection and storage completed successfully.")
        print("Sample content after insertion:")
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM contents LIMIT 1")
            print(cur.fetchone())
            
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()