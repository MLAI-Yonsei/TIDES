# TIDES/scripts/preprocess_data.py
import argparse
import json
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

def preprocess_techqa(raw_dir: Path, processed_dir: Path):
    logging.info("Processing TechQA dataset...")
    
    validation_path = raw_dir / 'TechQA/validation/validation_reference.json'
    technotes_path = raw_dir / 'TechQA/validation/validation_technotes.json'
    
    try:
        with open(validation_path, 'r', encoding='utf-8') as f:
            validation_data = json.load(f)
            
        with open(technotes_path, 'r', encoding='utf-8') as f:
            technotes = json.load(f)
            
        doc_dir = processed_dir / 'document_val'
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        for item in tqdm(validation_data, desc="Processing validation data"):
            for doc_id in item['DOC_IDS']:
                if doc_id in technotes:
                    doc_path = doc_dir / f'{doc_id}.json'
                    with open(doc_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'id': doc_id,
                            'text': technotes[doc_id]['text']
                        }, f, indent=2, ensure_ascii=False)
        
        logging.info("TechQA preprocessing completed!")
        
    except Exception as e:
        logging.error(f"Error preprocessing TechQA dataset: {str(e)}")
        raise

def preprocess_smart_tv_remote(raw_dir: Path, processed_dir: Path):
    logging.info("Processing manual dataset...")
    
    try:
        corpus_path = raw_dir / 'smart_tv_remote_manual_corpus.json'
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
            
        questions_path = raw_dir / 'smart_tv_remote_50_questions.csv'
        questions_df = pd.read_csv(questions_path)
        
        processed_corpus = {}
        for section_id, content in corpus_data.items():
            title = content['title']
            text = ' '.join(content['text'])
            processed_corpus[section_id] = {
                'id': section_id,
                'title': title,
                'text': f"{title}. {text}"
            }
        
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        corpus_output_path = processed_dir / 'smart_tv_remote_corpus_processed.json'
        with open(corpus_output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_corpus, f, indent=2, ensure_ascii=False)
            
        questions_output_path = processed_dir / 'smart_tv_remote_questions_processed.json'
        questions_data = questions_df.to_dict('records')
        with open(questions_output_path, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)
            
        logging.info("Manual dataset preprocessing completed!")
        
    except Exception as e:
        logging.error(f"Error preprocessing manual dataset: {str(e)}")
        raise

def preprocess_s10(raw_dir: Path, processed_dir: Path):
    logging.info("Processing s10 manual dataset...")
    
    try:
        corpus_path = raw_dir / 's10_manual_corpus.json'
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
            
        questions_path = raw_dir / 's10_50_questions.csv'
        questions_df = pd.read_csv(questions_path)
        
        processed_corpus = {}
        for section_id, content in corpus_data.items():
            title = content['title']
            text = ' '.join(content['text'])
            processed_corpus[section_id] = {
                'id': section_id,
                'title': title,
                'text': f"{title}. {text}"
            }
        
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        corpus_output_path = processed_dir / 's10_corpus_processed.json'
        with open(corpus_output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_corpus, f, indent=2, ensure_ascii=False)
            
        questions_output_path = processed_dir / 's10_questions_processed.json'
        questions_data = questions_df.to_dict('records')
        with open(questions_output_path, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)
            
        logging.info("Manual dataset preprocessing completed!")
        
    except Exception as e:
        logging.error(f"Error preprocessing manual dataset: {str(e)}")
        raise

def calculate_tfidf_scores(corpus: dict, questions: list, output_dir: Path):
    logging.info("Calculating TF-IDF scores...")
    
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        documents = [doc['text'] for doc in corpus.values()]
        section_ids = list(corpus.keys())
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.95,
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, question in enumerate(tqdm(questions, desc="Processing questions")):
            question_vector = vectorizer.transform([question['question']])
            scores = (question_vector @ tfidf_matrix.T).toarray().flatten()
            
            top_indices = scores.argsort()[-30:][::-1]
            top_doc_ids = [section_ids[idx] for idx in top_indices]
            
            output_path = output_dir / f'{i}_sorted_docs.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(top_doc_ids, f, indent=2)
                
        logging.info("TF-IDF calculation completed!")
        
    except Exception as e:
        logging.error(f"Error calculating TF-IDF scores: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Preprocess TIDES datasets')
    parser.add_argument('--dataset', choices=['techqa', 's10', 'smart_tv_remote'], required=True,
                      help='Dataset to preprocess')
    
    args = parser.parse_args()
    setup_logging()
    
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'data/raw'
    processed_dir = base_dir / 'data/processed'
    tfidf_dir = base_dir / 'data/tfidf'
    
    if args.dataset == 'techqa':
        preprocess_techqa(raw_dir / 'techqa', processed_dir / 'techqa')
    elif args.dataset == ' smart_tv_remote':
        preprocess_smart_tv_remote(raw_dir / 'smart_tv_remote', processed_dir / 'smart_tv_remote')
        
        with open(processed_dir / 'smart_tv_remote/smart_tv_remote_corpus_processed.json', 'r') as f:
            corpus = json.load(f)
        with open(processed_dir / 'smart_tv_remote/smart_tv_remote_questions_processed.json', 'r') as f:
            questions = json.load(f)
            
        calculate_tfidf_scores(corpus, questions, tfidf_dir / 'smart_tv_remote')
    else:
        preprocess_s10(raw_dir / 's10', processed_dir / 's10')
        
        with open(processed_dir / 's10/s10_corpus_processed.json', 'r') as f:
            corpus = json.load(f)
        with open(processed_dir / 's10/s10_questions_processed.json', 'r') as f:
            questions = json.load(f)
            
        calculate_tfidf_scores(corpus, questions, tfidf_dir / 's10')    

if __name__ == '__main__':
    main()