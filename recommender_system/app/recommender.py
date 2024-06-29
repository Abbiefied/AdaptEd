import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

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
            self.contents = pd.read_sql("SELECT * FROM contents", self.engine)
            self.interactions = pd.read_sql("SELECT * FROM interactions", self.engine)
            
            print("MIME type distribution:")
            print(self.contents['mime_type'].value_counts())
            
            print("\nSample content:")
            print(self.contents.iloc[0])
            
            logging.info("Data loaded successfully from the database")
        except Exception as e:
            logging.error(f"Error loading data from database: {e}")
            raise

    def prepare_data(self):
        try:
            # Prepare user-content interaction matrix
            self.user_content_matrix = self.interactions.pivot(index='user_id', columns='content_id', values='timestamp')
            self.user_content_matrix = self.user_content_matrix.notna().astype(int)

            # Prepare content feature matrix
            self.tfidf = TfidfVectorizer(stop_words='english')
            content_features = self.contents.set_index('id')['title'] + ' ' + self.contents.set_index('id')['body']
            self.content_feature_matrix = self.tfidf.fit_transform(content_features)

            logging.info("Data prepared successfully")
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise

    def get_user_preferences(self, user_id):
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        if user_interactions.empty:
            return ""
        user_content_ids = user_interactions['content_id'].tolist()
        user_preferences = self.contents[self.contents['id'].isin(user_content_ids)]['title'].tolist()
        return ' '.join(user_preferences)
    
    def get_user_content_type_preferences(self, user_id):
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        content_type_counts = user_interactions.merge(self.contents, left_on='content_id', right_on='id')['content_handler'].value_counts()
        total_interactions = content_type_counts.sum()
        return (content_type_counts / total_interactions).to_dict()

    def content_based_filtering(self, user_id, n=5):
        user_preferences = self.get_user_preferences(user_id)
        if not user_preferences:
            # If the user has no preferences, recommend the most popular content
            popular_content = self.contents.sort_values('id', ascending=False)['id'].tolist()[:n]
            return popular_content

        user_profile = self.tfidf.transform([user_preferences])
        cosine_similarities = cosine_similarity(user_profile, self.content_feature_matrix).flatten()
        related_content_indices = cosine_similarities.argsort()[::-1]
        return self.contents.iloc[related_content_indices]['id'].tolist()[:n]
    
    def collaborative_filtering(self, user_id, n=5):
        user_vector = self.user_content_matrix.loc[user_id].values.reshape(1, -1)
        user_similarities = cosine_similarity(user_vector, self.user_content_matrix.values)
        similar_users = user_similarities.argsort().flatten()[::-1][1:11]  # Top 10 similar users
        
        recommended_contents = []
        for similar_user in self.user_content_matrix.index[similar_users]:
            contents = self.user_content_matrix.loc[similar_user][self.user_content_matrix.loc[similar_user] == 1].index.tolist()
            recommended_contents.extend(contents)
        
        recommended_contents = list(set(recommended_contents) - set(self.user_content_matrix.loc[user_id][self.user_content_matrix.loc[user_id] == 1].index.tolist()))
        return recommended_contents[:n]

    def time_decay_factor(self, timestamp, now=datetime.now()):
        days_since_interaction = (now - timestamp).days
        return np.exp(-days_since_interaction / 30)  # 30-day half-life

    def get_user_mime_type_preferences(self, user_id):
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        mime_type_counts = user_interactions.merge(self.contents, left_on='content_id', right_on='id')['mime_type'].value_counts()
        total_interactions = mime_type_counts.sum()
        return (mime_type_counts / total_interactions).to_dict()

    def hybrid_recommendations(self, user_id, n=5):
        content_based_recs = self.content_based_filtering(user_id, n)
        collaborative_recs = self.collaborative_filtering(user_id, n)
        
        # Get user's MIME type preferences
        mime_type_prefs = self.get_user_mime_type_preferences(user_id)
        
        # Combine and score recommendations
        all_recs = content_based_recs + collaborative_recs
        rec_scores = {}
        
        for content_id in all_recs:
            mime_type = self.contents[self.contents['id'] == content_id]['mime_type'].iloc[0]
            type_score = mime_type_prefs.get(mime_type, 0)
            
            # Check if the user has interacted with this content before
            interaction = self.interactions[(self.interactions['user_id'] == user_id) & 
                                            (self.interactions['content_id'] == content_id)]
            
            if not interaction.empty:
                last_interaction = pd.to_datetime(interaction['timestamp'].iloc[0])
                time_score = self.time_decay_factor(last_interaction)
            else:
                time_score = 1  # No previous interaction, so no decay
            
            rec_scores[content_id] = type_score * time_score
        
        # Sort recommendations by score
        sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
        return [content_id for content_id, score in sorted_recs[:n]]
    
    def get_content_details(self, content_ids):
        return self.contents[self.contents['id'].isin(content_ids)][['id', 'title', 'body', 'mime_type']].to_dict('records')

    def recommend(self, user_id, n=5):
        try:
            # Check if the user exists in the users dataframe
            if user_id not in self.users['id'].values:
                logging.warning(f"User {user_id} not found in the database.")
                return []

            # Check if the user has any interactions
            if self.interactions.empty or user_id not in self.interactions['user_id'].values:
                logging.info(f"User {user_id} has no interactions. Providing content-based recommendations only.")
                recommended_content_ids = self.content_based_filtering(user_id, n)
            else:
                recommended_content_ids = self.hybrid_recommendations(user_id, n)

            recommendations = self.get_content_details(recommended_content_ids)
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
        
        print(f"\nRecommendations for user {user_id}:")
        if recommendations:
            for rec in recommendations:
                print(f"Content: {rec['title']}")
                print(f"MIME Type: {rec['mime_type']}")
                print(f"Description: {rec['body'][:100]}...")
                print()
        else:
            print("No recommendations available for this user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")