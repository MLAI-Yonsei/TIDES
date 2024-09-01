import os
import json
import re
import time
import argparse
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

def load_document(doc_name, raw_text_root):
    file_path = os.path.join(raw_text_root, f"{doc_name}.json")
    with open(file_path, "r", encoding="utf-8") as file:
        content = json.load(file)["text"]
    return content

def calculate_tfidf_scores(doc_names, query, raw_text_root):
    documents = [preprocess_text(load_document(doc_name, raw_text_root)) for doc_name in doc_names]
    preprocessed_query = preprocess_text(query)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    query_vector = tfidf_vectorizer.transform([preprocessed_query])
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    return {doc_name: score for doc_name, score in zip(doc_names, similarity_scores)}

def run_benchmark(val_q_a_path, raw_text_root, output_path_root):
    with open(val_q_a_path, 'r', encoding='utf-8') as file:
        question_data_list = json.load(file)

    sorted_doc_lst = []
    for i, question_data in enumerate(tqdm(question_data_list)):
        query = f"{question_data['QUESTION_TITLE']}\n{question_data['QUESTION_TEXT']}"
        tfidf_scores = calculate_tfidf_scores(question_data['DOC_IDS'], query, raw_text_root)
        sorted_scores = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [doc_name for doc_name, score in sorted_scores[:30]]
        sorted_doc_lst.append((question_data['QUESTION_ID'], top_docs))

    for question_id, doc_list in sorted_doc_lst:
        output_path = os.path.join(output_path_root, f"{question_id}_sorted_docs.json")
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(doc_list, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TF-IDF')
    parser.add_argument('--val_q_a_path', type=str, required=True, help='Path to validation reference JSON')
    parser.add_argument('--raw_text_root', type=str, required=True, help='Root path to raw text documents')
    parser.add_argument('--output_path_root', type=str, required=True, help='Root path to save outputs')

    args = parser.parse_args()

    run_benchmark(
        val_q_a_path=args.val_q_a_path,
        raw_text_root=args.raw_text_root,
        output_path_root=args.output_path_root
    )
