import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse.linalg import svds
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime
from evaluation import evaluate_recommendations
import logging
import random
import os

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
            
            print(f"Number of interactions loaded: {len(self.interactions)}")
                    # Remove duplicate interactions
            self.interactions = self.interactions.drop_duplicates(subset=['user_id', 'content_id'])
            
            print(f"Number of interactions after removing duplicates: {len(self.interactions)}")
        
            logging.info("Data loaded successfully from the database")
        except Exception as e:
            logging.error(f"Error loading data from database: {e}")
            raise

    def prepare_data(self):
        try:
            print("Interactions shape:", self.interactions.shape)
            print("Unique users:", self.interactions['user_id'].nunique())
            print("Unique contents:", self.interactions['content_id'].nunique())
            print("Duplicate user-content pairs:", 
                self.interactions.duplicated(subset=['user_id', 'content_id']).sum())
            
            # Prepare user-content interaction matrix
            self.user_content_matrix = self.interactions.pivot_table(
                index='user_id', 
                columns='content_id', 
                values='timestamp',
                aggfunc='count'
            ).fillna(0)

            # Prepare content feature matrix
            self.tfidf = TfidfVectorizer(stop_words='english')
            content_features = self.contents.set_index('id')['title'] + ' ' + self.contents.set_index('id')['body']
            self.content_feature_matrix = self.tfidf.fit_transform(content_features)

            # Apply LDA for topic modeling
            self.lda = LatentDirichletAllocation(n_components=10, random_state=42)
            self.content_topics = self.lda.fit_transform(self.content_feature_matrix)

            # Extract course metadata
            self.contents['subject'] = self.contents['title'].apply(lambda x: x.split(':')[1].strip())
            self.course_subjects = pd.get_dummies(self.contents['subject'])

            # Combine features
            self.content_features = np.hstack([self.content_feature_matrix.toarray(), self.content_topics, self.course_subjects])

            logging.info("Data prepared successfully")
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise
    
    def split_data(self, test_ratio=0.2):
        self.interactions['timestamp'] = pd.to_datetime(self.interactions['timestamp'])
        self.interactions = self.interactions.sort_values('timestamp')
        
        test_interactions = self.interactions.groupby('user_id').last()
        train_interactions = self.interactions[~self.interactions.index.isin(test_interactions.index)]
        
        # Handle users with only one interaction
        single_interaction_users = self.interactions.groupby('user_id').size()[self.interactions.groupby('user_id').size() == 1].index
        for user in single_interaction_users:
            if random.random() < test_ratio:
                test_interactions = test_interactions.append(train_interactions.loc[train_interactions['user_id'] == user].iloc[0])
                train_interactions = train_interactions[train_interactions['user_id'] != user]
            else:
                train_interactions = train_interactions.append(test_interactions.loc[user])
                test_interactions = test_interactions.drop(user)
        
        self.train_data = train_interactions
        self.test_data = test_interactions

        print(f"Total interactions: {len(self.interactions)}")
        print(f"Training set size: {len(self.train_data)}")
        print(f"Test set size: {len(self.test_data)}")

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

    def matrix_factorization(self):
        # Create user-item matrix
        user_item_matrix = self.user_content_matrix.values

        # Perform SVD
        U, sigma, Vt = svds(user_item_matrix, k=50)  # You can adjust the number of factors (k)

        # Convert to diagonal matrix
        sigma = np.diag(sigma)

        # Predict ratings
        user_item_estimated = np.dot(np.dot(U, sigma), Vt)

        # Convert back to dataframe
        self.user_item_predicted = pd.DataFrame(user_item_estimated, columns=self.user_content_matrix.columns, index=self.user_content_matrix.index)  
    
    def collaborative_filtering(self, user_id, n=5):
        if not hasattr(self, 'user_item_predicted'):
            self.matrix_factorization()
        
        user_predictions = self.user_item_predicted.loc[user_id]
        
        # Get contents that the user hasn't interacted with
        already_interacted = self.user_content_matrix.loc[user_id][self.user_content_matrix.loc[user_id] > 0].index
        user_predictions = user_predictions.drop(already_interacted)
        
        top_recommendations = user_predictions.sort_values(ascending=False).index[:n].tolist()
        
        return top_recommendations

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
            
            # Get content-based similarity score
            content_index = self.contents[self.contents['id'] == content_id].index[0]
            content_similarity = cosine_similarity(self.content_features[content_index].reshape(1, -1), 
                                                self.content_features).flatten()[content_index]
            
            # Get collaborative filtering score
            cf_score = self.user_item_predicted.loc[user_id, content_id] if content_id in self.user_item_predicted.columns else 0
            
            # Combine scores
            rec_scores[content_id] = 0.4 * type_score + 0.3 * content_similarity + 0.3 * cf_score
        
        # Sort recommendations by score
        sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
        return [content_id for content_id, score in sorted_recs[:n]]
        
    def get_content_details(self, content_ids):
        return self.contents[self.contents['id'].isin(content_ids)][['id', 'title', 'body', 'mime_type']].to_dict('records')
    
    def recommend(self, user_id, n=5, for_evaluation=False):
        try:
            if for_evaluation:
                # Use only training data for recommendations during evaluation
                original_interactions = self.interactions
                self.interactions = self.train_data
        
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
                return recommendations
            except Exception as e:
                logging.error(f"Error generating recommendations for user {user_id}: {e}")
                return []
            
        finally:
            if for_evaluation:
                # Restore original interactions data
                self.interactions = original_interactions

    def evaluate(self, k=10):
        self.split_data()
        
        y_true = []
        y_pred = []
        
        for user_id in self.test_data.index:
            true_items = self.test_data.loc[user_id, 'content_id']
            if not isinstance(true_items, list):
                true_items = [true_items]
            pred_items = [rec['id'] for rec in self.recommend(user_id, n=k, for_evaluation=True)]
            
            # Ensure both true_items and pred_items are non-empty before appending
            if true_items and pred_items:
                y_true.append(true_items)
                y_pred.append(pred_items)
        
        print(f"Number of users with valid predictions: {len(y_true)}")
        
        if not y_true or not y_pred:
            print("Warning: No valid data for evaluation")
            return {f'precision@{k}': 0, f'recall@{k}': 0, 'MAP': 0}
        
        try:
            return evaluate_recommendations(y_true, y_pred, k)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("y_true lengths:", [len(yt) for yt in y_true])
            print("y_pred lengths:", [len(yp) for yp in y_pred])
            return {f'precision@{k}': 0, f'recall@{k}': 0, 'MAP': 0}

if __name__ == "__main__":
    try:
        recommender = Recommender()
        
        # Evaluate the recommender system
        evaluation_results = recommender.evaluate(k=10)
        print("\nEvaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value:.4f}")
      
      # Get the first user_id for demonstration
        user_id = recommender.users['id'].iloc[0]  
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