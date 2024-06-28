from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import json

app = FastAPI()

# Load mock data
with open('blackboard_mock_data.json', 'r') as f:
    mock_data = json.load(f)

@app.get("/learn/api/public/v1/users")
async def get_users(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100)
):
    users = mock_data['users']
    return {
        "results": users[offset:offset+limit],
        "paging": {
            "offset": offset,
            "limit": limit,
            "count": len(users[offset:offset+limit]),
            "total": len(users)
        }
    }

@app.get("/learn/api/public/v1/users/{user_id}")
async def get_user(user_id: str):
    user = next((user for user in mock_data['users'] if user['id'] == user_id), None)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/learn/api/public/v1/courses")
async def get_courses(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100)
):
    courses = mock_data['courses']
    return {
        "results": courses[offset:offset+limit],
        "paging": {
            "offset": offset,
            "limit": limit,
            "count": len(courses[offset:offset+limit]),
            "total": len(courses)
        }
    }

@app.get("/learn/api/public/v1/courses/{course_id}")
async def get_course(course_id: str):
    course = next((course for course in mock_data['courses'] if course['id'] == course_id), None)
    if course is None:
        raise HTTPException(status_code=404, detail="Course not found")
    return course

@app.get("/learn/api/public/v1/courses/{course_id}/contents")
async def get_course_contents(course_id: str):
    contents = next((item['contents'] for item in mock_data['contents'] if item['courseId'] == course_id), None)
    if contents is None:
        raise HTTPException(status_code=404, detail="Course contents not found")
    return {"results": contents}

@app.get("/learn/api/public/v1/courses/{course_id}/users")
async def get_course_users(course_id: str):
    enrollments = next((item['enrollments'] for item in mock_data['enrollments'] if item['courseId'] == course_id), None)
    if enrollments is None:
        raise HTTPException(status_code=404, detail="Course enrollments not found")
    return {"results": enrollments}

@app.get("/learn/api/public/v1/users/{user_id}/courses")
async def get_user_courses(user_id: str):
    user_courses = []
    for course_enrollment in mock_data['enrollments']:
        for enrollment in course_enrollment['enrollments']:
            if enrollment['userId'] == user_id:
                course = next((course for course in mock_data['courses'] if course['id'] == course_enrollment['courseId']), None)
                if course:
                    user_courses.append(course)
    return {"results": user_courses}

@app.get("/learn/api/public/v2/courses/{course_id}/gradebook/columns/{column_id}/users")
async def get_course_grades(course_id: str, column_id: str):
    grades = next((item['grades'] for item in mock_data['grades'] if item['courseId'] == course_id), None)
    if grades is None:
        raise HTTPException(status_code=404, detail="Course grades not found")
    return {"results": grades}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)