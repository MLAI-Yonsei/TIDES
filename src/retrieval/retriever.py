# TIDES/src/retrieval/retriever.py
from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import logging

class BaseRetriever(ABC):
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
        
    @abstractmethod
    def retrieve_documents(self, query, documents, top_k):
        pass
        
    def save_results(self, results, output_path):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

class TFIDFRetriever(BaseRetriever):
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            strip_accents='unicode',
            lowercase=True
        )
        
    def retrieve_documents(self, query, documents, top_k=30):
        try:
            processed_docs = [self.preprocess_text(doc) for doc in documents]
            processed_query = self.preprocess_text(query)
            
            tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
            query_vector = self.vectorizer.transform([processed_query])
            
            similarity_scores = (query_vector @ tfidf_matrix.T).toarray().flatten()
            
            top_indices = similarity_scores.argsort()[-top_k:][::-1]
            
            return {
                'indices': top_indices.tolist(),
                'scores': similarity_scores[top_indices].tolist()
            }
            
        except Exception as e:
            logging.error(f"Error in TF-IDF retrieval: {str(e)}")
            raise

class CosineRetriever(BaseRetriever):
    
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            strip_accents='unicode',
            lowercase=True
        )
        
    def retrieve_documents(self, query, documents, top_k=30):
        try:
            processed_docs = [self.preprocess_text(doc) for doc in documents]
            processed_query = self.preprocess_text(query)
            
            doc_vectors = self.vectorizer.fit_transform(processed_docs)
            query_vector = self.vectorizer.transform([processed_query])
            
            similarity_scores = cosine_similarity(query_vector, doc_vectors)[0]
            
            top_indices = similarity_scores.argsort()[-top_k:][::-1]
            
            return {
                'indices': top_indices.tolist(),
                'scores': similarity_scores[top_indices].tolist()
            }
            
        except Exception as e:
            logging.error(f"Error in cosine similarity retrieval: {str(e)}")
            raise

def get_retriever(method):
    retrievers = {
        'tfidf': TFIDFRetriever,
        'cosine': CosineRetriever
    }
    
    if method not in retrievers:
        raise ValueError(f"Unsupported retrieval method: {method}")
        
    return retrievers[method]()