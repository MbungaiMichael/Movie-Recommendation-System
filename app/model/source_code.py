import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer


new_file = pd.read_csv(r'C:\Users\UltraBook 3.1\Desktop\data_analysis projects\Olame_projects\movie_recommendation'
                       r'\ml-32m\merged_data1.csv')

# build the model for deployment


class DropDuplicatesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.drop_duplicates(inplace=True)
        return X


class GenresSplitterTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()  # avoid changing the original DataFrame
        X['genres'] = X['genres'].str.split('|')
        return X


class TagVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=300):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit(self, X, y=None):
        self.vectorizer.fit(X['tag'].fillna('Unknown'))
        return self

    def transform(self, X):
        X = X.copy()
        tag_matrix = self.vectorizer.transform(X['tag'].fillna('Unknown'))

        # Calculate the mean TF-IDF score for each tag
        avg_tfidf_scores = tag_matrix.mean(axis=1)
        tag_mean_tfidf = pd.Series([float(score) for score in avg_tfidf_scores], index=X.index)

        # Assign to the DataFrame
        X = X.assign(tag_mean_tfidf=tag_mean_tfidf)

        # Optionally drop the original tag column
        X.drop(columns=['tag'], inplace=True)
        return X


class GenresMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        self.mlb.fit(X['genres'])  # X is expected to be a DataFrame
        return self

    def transform(self, X):
        genre_encoded = self.mlb.transform(X['genres'])
        genre_df = pd.DataFrame(genre_encoded, columns=self.mlb.classes_, index=X.index)
        # Drop the original genre column and concat the new one-hot encoded columns
        X = pd.merge(X, genre_df, left_index=True, right_index=True)
        X.drop('genres', axis=1, inplace=True)
        return X


class TagFrequencyScorer(BaseEstimator, TransformerMixin):
    def __init__(self, tag_column='tag', score_column='Tag_score'):
        self.tag_column = tag_column
        self.score_column = score_column
        self.tag_counts_ = None

    def fit(self, X, y=None):
        self.tag_counts_ = X[self.tag_column].value_counts()
        return self

    def transform(self, X):
        X = X.copy()
        X[self.score_column] = X[self.tag_column].map(self.tag_counts_)
        return X


class KMeansClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        clusters = self.kmeans.predict(X_scaled)

        # Convert X to DataFrame (in case it's still an array)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_transformed = X.copy()
        X_transformed['cluster'] = clusters
        return X_transformed


complete_pipeline = Pipeline([
    ('drop_duplicates', DropDuplicatesTransformer()),
    ('split_genres', GenresSplitterTransformer()),
    ('encode_genres', GenresMultiLabelBinarizer()),
    ('tag_score', TagFrequencyScorer(tag_column='tag')),
    ('fill_vectorize_tags', TagVectorizer()),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('kmeans', KMeansClusteringTransformer(n_clusters=5, random_state=42))
])


df_with_clusters = complete_pipeline.fit_transform(new_file)
#
version = "1.0.0"
file_directory = rf"C:\Users\UltraBook 3.1\PycharmProjects\Movie_rec2\cluster_pipeline_v{version}.pkl"
joblib.dump(complete_pipeline, file_directory)
print(f"Pipeline saved to: {file_directory}")

