import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Database configuration
DB_NAME = os.getenv('DB_NAME', 'adapted_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'Abbie2504')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')

class Recommender:
    def __init__(self):
        self.engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        self.load_data()
        self.prepare_data()

    def load_data(self):
        try:
            self.users = pd.read_sql("SELECT * FROM users", self.engine)
            self.courses = pd.read_sql("SELECT * FROM courses", self.engine)
            self.enrollments = pd.read_sql("SELECT * FROM enrollments", self.engine)
            self.grades = pd.read_sql("SELECT * FROM grades", self.engine)
            self.contents = pd.read_sql("SELECT * FROM contents", self.engine)
            logging.info("Data loaded successfully from the database")
        except Exception as e:
            logging.error(f"Error loading data from database: {e}")
            raise

    def prepare_data(self):
        try:
            # Prepare user-course interaction matrix
            self.user_course_matrix = self.enrollments.pivot(index='user_id', columns='course_id', values='course_id')
            self.user_course_matrix = self.user_course_matrix.notna().astype(int)

            # Prepare course content matrix
            self.tfidf = TfidfVectorizer(stop_words='english')
            course_content = self.courses.set_index('id')['description'].str.lower()
            self.course_content_matrix = self.tfidf.fit_transform(course_content)
            logging.info("Data prepared successfully")
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise

    def get_user_preferences(self, user_id):
        user_courses = self.enrollments[self.enrollments['user_id'] == user_id]['course_id'].tolist()
        user_preferences = self.courses[self.courses['id'].isin(user_courses)]['description'].str.lower().tolist()
        return ' '.join(user_preferences)

    def content_based_filtering(self, user_id, n=5):
        user_preferences = self.get_user_preferences(user_id)
        user_profile = self.tfidf.transform([user_preferences])
        cosine_similarities = cosine_similarity(user_profile, self.course_content_matrix).flatten()
        related_courses_indices = cosine_similarities.argsort()[::-1]
        return self.courses.iloc[related_courses_indices]['id'].tolist()[:n]

    def collaborative_filtering(self, user_id, n=5):
        user_vector = self.user_course_matrix.loc[user_id].values.reshape(1, -1)
        user_similarities = cosine_similarity(user_vector, self.user_course_matrix.values)
        similar_users = user_similarities.argsort().flatten()[::-1][1:11]  # Top 10 similar users
        
        recommended_courses = []
        for similar_user in self.user_course_matrix.index[similar_users]:
            courses = self.user_course_matrix.loc[similar_user][self.user_course_matrix.loc[similar_user] == 1].index.tolist()
            recommended_courses.extend(courses)
        
        recommended_courses = list(set(recommended_courses) - set(self.user_course_matrix.loc[user_id][self.user_course_matrix.loc[user_id] == 1].index.tolist()))
        return recommended_courses[:n]

    def hybrid_recommendations(self, user_id, n=5):
        content_based_recs = self.content_based_filtering(user_id, n)
        collaborative_recs = self.collaborative_filtering(user_id, n)
        
        hybrid_recs = []
        for i in range(n):
            if i < len(content_based_recs):
                hybrid_recs.append(content_based_recs[i])
            if i < len(collaborative_recs):
                hybrid_recs.append(collaborative_recs[i])
        
        return list(dict.fromkeys(hybrid_recs))[:n]  # Remove duplicates and limit to n recommendations

    def get_course_details(self, course_ids):
        return self.courses[self.courses['id'].isin(course_ids)][['id', 'name', 'description']].to_dict('records')

    def recommend(self, user_id, n=5):
        try:
            recommended_course_ids = self.hybrid_recommendations(user_id, n)
            recommendations = self.get_course_details(recommended_course_ids)
            logging.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
        except Exception as e:
            logging.error(f"Error generating recommendations for user {user_id}: {e}")
            return []

if __name__ == "__main__":
    try:
        recommender = Recommender()
        
        # Example usage
        user_id = recommender.users['id'].iloc[0]  # Get the first user_id for demonstration
        recommendations = recommender.recommend(user_id, n=5)
        
        print(f"Recommendations for user {user_id}:")
        for rec in recommendations:
            print(f"Course: {rec['name']}")
            print(f"Description: {rec['description'][:100]}...")  # Print first 100 characters of description
            print()
    except Exception as e:
        logging.error(f"An error occurred: {e}")