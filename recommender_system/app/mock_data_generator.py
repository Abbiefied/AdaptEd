import random
from datetime import datetime, timedelta
import json
import uuid

def generate_uuid():
    return str(uuid.uuid4())

def generate_users(num_users=100):
    users = []
    for i in range(num_users):
        user = {
            "id": generate_uuid(),
            "uuid": generate_uuid(),
            "externalId": f"student_{i}",
            "dataSourceId": generate_uuid(),
            "userName": f"student_{i}",
            "studentId": f"STU{i:05d}",
            "name": {
                "given": f"Student{i}",
                "family": f"Lastname{i}",
            },
            "contact": {
                "email": f"student_{i}@example.com"
            },
            "availability": {
                "available": "Yes"
            },
        }
        users.append(user)
    return users

def generate_courses(num_courses=20):
    courses = []
    subjects = ['Mathematics', 'Science', 'History', 'Literature', 'Computer Science']
    for i in range(num_courses):
        course = {
            "id": generate_uuid(),
            "uuid": generate_uuid(),
            "externalId": f"course_{i}",
            "dataSourceId": generate_uuid(),
            "courseId": f"COURSE{i:03d}",
            "name": f"Course {i}: {random.choice(subjects)}",
            "description": f"Description for Course {i}",
            "created": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
            "organization": False,
            "ultraStatus": "Classic",
            "allowGuests": False,
            "closedComplete": False,
            "availability": {
                "available": "Yes"
            },
        }
        courses.append(course)
    return courses

def generate_contents(courses, num_contents_per_course=5):
    all_contents = []
    content_formats = ['video/mp4', 'audio/mpeg', 'text/plain', 'application/pdf', 'image/jpeg']
    content_formats = ['video/mp4', 'audio/mpeg', 'text/plain', 'application/pdf', 'image/jpeg']
    for course in courses:
        course_contents = []
        for i in range(num_contents_per_course):
            mime_type = random.choice(content_formats)
            mime_type = random.choice(content_formats)
            content = {
                "id": generate_uuid(),
                "title": f"Content {i} for {course['name']}",
                "body": f"This is the content body for Content {i} in {course['name']}",
                "created": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                "position": i,
                "mime_type": mime_type,
                "attachments": [
                    {
                        "id": generate_uuid(),
                        "fileName": f"file_{i}.{mime_type.split('/')[-1]}",
                        "mimeType": mime_type
                    }
                ],
                "mime_type": mime_type,
                "attachments": [
                    {
                        "id": generate_uuid(),
                        "fileName": f"file_{i}.{mime_type.split('/')[-1]}",
                        "mimeType": mime_type
                    }
                ],
                "links": []
            }
            course_contents.append(content)
        all_contents.append({"courseId": course["id"], "contents": course_contents})
    return all_contents

def generate_enrollments(users, courses):
    enrollments = []
    for course in courses:
        course_enrollments = []
        for user in random.sample(users, k=random.randint(5, len(users))):
            enrollment = {
                "userId": user["id"],
                "courseId": course["id"],
                "dataSourceId": generate_uuid(),
                "created": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                "availability": {
                    "available": "Yes"
                },
            }
            course_enrollments.append(enrollment)
        enrollments.append({"courseId": course["id"], "enrollments": course_enrollments})
    return enrollments

def generate_grades(enrollments):
    all_grades = []
    for course_enrollment in enrollments:
        course_grades = []
        for enrollment in course_enrollment["enrollments"]:
            grade = {
                "userId": enrollment["userId"],
                "columnId": generate_uuid(),
                "status": random.choice(["Graded", "Needs Grading", "In Progress"]),
                "score": random.randint(0, 100),
                "exempt": False,
            }
            course_grades.append(grade)
        all_grades.append({"courseId": course_enrollment["courseId"], "grades": course_grades})
    return all_grades

def generate_all_data():
    users = generate_users()
    courses = generate_courses()
    contents = generate_contents(courses)
    enrollments = generate_enrollments(users, courses)
    grades = generate_grades(enrollments)

    return {
        "users": users,
        "courses": courses,
        "contents": contents,
        "enrollments": enrollments,
        "grades": grades
    }

if __name__ == "__main__":
    data = generate_all_data()
    print(data['contents'][0]['contents'][0])
    with open('blackboard_mock_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("Mock Blackboard Learn data generated and saved to blackboard_mock_data.json")