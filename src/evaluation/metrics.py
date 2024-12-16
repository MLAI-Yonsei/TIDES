# TIDES/src/evaluation/metrics.py
import bert_score
import evaluate
from typing import List, Dict, Any
import pandas as pd
import json
import os

class MetricsCalculator:
    def __init__(self):
        self.rouge = evaluate.load("rouge")

    def calculate_token_f1(self, predictions: List[str], references: List[str]) -> float:
        f1_scores = []
        for pred, ref in zip(predictions, references):
            if pred == "-" or ref == "-":
                continue
            
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            common_tokens = set(pred_tokens) & set(ref_tokens)
            
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            f1_scores.append(f1)
            
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0

    def calculate_bert_score(self, predictions: List[str], references: List[str]) -> float:
        bert_result = bert_score.score(predictions, references, lang="en", verbose=True)
        bert_f1_scores = bert_result[2].tolist()  # F1 scores
        return sum(bert_f1_scores) / len(bert_f1_scores)

    def calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        return self.rouge.compute(predictions=predictions, references=references)

    def evaluate_techqa(self, output_dir: str, reference_dir: str) -> Dict[str, float]:
        predictions = []
        references = []
        
        for i in range(20):  # TechQA validation set size
                
            # Load reference
            ref_path = os.path.join(reference_dir, f"{str(i).zfill(3)}.json")
            with open(ref_path, "r") as f:
                refer = json.load(f)
            references.append(refer["gt"])
            
            # Load prediction
            try:
                pred_path = os.path.join(output_dir, f"result{str(i).zfill(3)}.json")
                with open(pred_path, "r") as f:
                    result = json.load(f)
                predictions.append(result["stage3"])
            except (FileNotFoundError, json.JSONDecodeError):
                predictions.append("")
        
        # Calculate metrics
        metrics = {
            'bert_score': self.calculate_bert_score(predictions, references) * 100,
            'token_f1': self.calculate_token_f1(predictions, references) * 100
        }
        
        rouge_scores = self.calculate_rouge_scores(predictions, references)
        metrics.update({
            'rouge1': rouge_scores['rouge1'] * 100,
            'rouge2': rouge_scores['rouge2'] * 100,
            'rougeL': rouge_scores['rougeL'] * 100
        })
        
        return metrics

    def evaluate_manual(self, output_dir: str, dataset_type: str) -> Dict[str, float]:
        # Load questions data
        if dataset_type == "s10":
            question_path = "data/raw/s10/s10_50_questions.csv"
        else:  # smart_tv_remote
            question_path = "data/raw/smart_tv_remote/smart_tv_remote_50_questions.csv"
            
        questions_df = pd.read_csv(question_path)
        
        predictions = []
        references = []
        
        for i in range(len(questions_df)):
            references.append(questions_df.loc[i, "answer"])
            
            try:
                pred_path = os.path.join(output_dir, f"result{str(i).zfill(3)}.json")
                with open(pred_path, "r") as f:
                    result = json.load(f)
                predictions.append(result["stage3"])
            except (FileNotFoundError, json.JSONDecodeError):
                predictions.append("")
        
        # Calculate metrics
        metrics = {
            'bert_score': self.calculate_bert_score(predictions, references) * 100,
            'token_f1': self.calculate_token_f1(predictions, references) * 100
        }
        
        rouge_scores = self.calculate_rouge_scores(predictions, references)
        metrics.update({
            'rouge1': rouge_scores['rouge1'] * 100,
            'rouge2': rouge_scores['rouge2'] * 100,
            'rougeL': rouge_scores['rougeL'] * 100
        })
        
        return metrics