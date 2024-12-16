# TIDES/src/data/dataset_utils.py
import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Any

class DatasetLoader:
    def __init__(self, config: Dict[str, Any]):

        self.config = config
        self.base_dir = Path(config['data']['base_path'])
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        self.logger = logging.getLogger(__name__)

    def load_dataset(self, dataset_type: str) -> Dict[str, Any]:
        if dataset_type == 'techqa':
            return self.load_techqa()
        elif dataset_type == 's10':
            return self.load_s10()
        elif dataset_type == 'smart_tv_remote':
            return self.load_smart_tv_remote()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def load_techqa(self) -> Dict[str, Any]:
        try:
            validation_path = self.raw_dir / 'techqa/TechQA/validation/validation_reference.json'
            with open(validation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            processed_data = {
                'questions': [],
                'documents': {},
                'metadata': {
                    'total_questions': len(data),
                    'dataset_type': 'techqa'
                }
            }
            
            for item in data:
                processed_data['questions'].append({
                    'question_id': item.get('QUESTION_ID', ''),
                    'title': item.get('QUESTION_TITLE', ''),
                    'body': item.get('QUESTION_TEXT', ''),
                    'doc_ids': item.get('DOC_IDS', []),
                    'answerable': item.get('ANSWERABLE', 'N'),
                    'answer': item.get('ANSWER', '-'),
                    'document': item.get('DOCUMENT', '-')
                })
            
            self.logger.info(f"Loaded {len(processed_data['questions'])} TechQA questions")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error loading TechQA dataset: {str(e)}")
            raise

    def load_smart_tv_remote(self) -> Dict[str, Any]:
        try:
            questions_path = self.raw_dir / 'smart_tv_remote/smart_tv_remote_50_questions.csv'
            questions_df = pd.read_csv(questions_path)
            
            corpus_path = self.raw_dir / 'smart_tv_remote/smart_tv_remote_manual_corpus.json'
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
            
            processed_data = {
                'questions': [],
                'documents': {},
                'metadata': {
                    'total_questions': len(questions_df),
                    'total_sections': len(corpus_data),
                    'dataset_type': 'smart_tv_remote'
                }
            }
            
            for _, row in questions_df.iterrows():
                processed_data['questions'].append({
                    'question_id': row.name,
                    'question': row['question'],
                    'answer': row['answer']
                })
            
            for section_id, content in corpus_data.items():
                processed_data['documents'][section_id] = {
                    'title': content['title'],
                    'text': ' '.join(content['text'])
                }
            
            self.logger.info(f"Loaded {len(processed_data['questions'])} smart tv remote questions and "
                           f"{len(processed_data['documents'])} document sections")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error loading smart tv remote dataset: {str(e)}")
            raise

    def load_s10(self) -> Dict[str, Any]:
        try:
            questions_path = self.raw_dir / 's10/s10_50_questions.csv'
            questions_df = pd.read_csv(questions_path)
            
            corpus_path = self.raw_dir / 's10/s10_manual_corpus.json'
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
            
            processed_data = {
                'questions': [],
                'documents': {},
                'metadata': {
                    'total_questions': len(questions_df),
                    'total_sections': len(corpus_data),
                    'dataset_type': 's10'
                }
            }
            
            for _, row in questions_df.iterrows():
                processed_data['questions'].append({
                    'question_id': row.name,
                    'question': row['question'],
                    'answer': row['answer']
                })
            
            for section_id, content in corpus_data.items():
                processed_data['documents'][section_id] = {
                    'title': content['title'],
                    'text': ' '.join(content['text'])
                }
            
            self.logger.info(f"Loaded {len(processed_data['questions'])} s10 questions and "
                           f"{len(processed_data['documents'])} document sections")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error loading s10 dataset: {str(e)}")
            raise        
        

    def load_documents(self, doc_ids: List[str]) -> Dict[str, str]:
        documents = {}
        for doc_id in doc_ids:
            try:
                doc_path = self.processed_dir / f'techqa/document_val/{doc_id}.json'
                with open(doc_path, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                    documents[doc_id] = doc_data['text']
            except Exception as e:
                self.logger.error(f"Error loading document {doc_id}: {str(e)}")
                continue
        
        return documents

    def save_processed_data(self, data: Dict[str, Any], output_path: Path) -> None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved processed data to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            raise