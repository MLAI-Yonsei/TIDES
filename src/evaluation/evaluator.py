# TIDES/src/evaluation/evaluator.py
import logging
from typing import Dict, List, Any
import time
import json
from pathlib import Path

class ResponseEvaluator:
    def __init__(self, model_manager, dataset_type, config=None):
        self.model_manager = model_manager
        self.dataset_type = dataset_type
        self.config = config or {}
        self.instructions = self._load_instructions()
        self.logger = logging.getLogger(__name__)

    def _load_instructions(self) -> Dict[str, str]:
        return {
            'step1': """You're an expert in IT and computer science. Provide a rationale in 20 words or less for how the paragraph relates to the question, focusing on key technical details. Finally, give a clear yes or no answer avoiding overconfidence.""",
            
            'step2': """You're an expert in IT and computer science. Given a question and a context, if the context contains information that directly answers the question or provides clear supporting evidence, extract only the relevant section. Do not include any additional explanation or comments. If no relevant information is found, simply respond 'No'.""",
            
            'step3': """Compose a concise answer to the question using the most relevant key words and phrases from the rationales. Aim for a natural response while still aligning closely with the rationales. If the rationales do not sufficiently address the question, respond with 'No answer'."""
        }

    def evaluate(self, question: str, context: Dict[str, str], retrieved_docs: List[str]) -> Dict[str, Any]:
        try:
            start_time = time.time()
            result = {
                'question': question,
                'retrieved_docs': retrieved_docs,
                'timestamps': {}
            }
            self.logger.info("Starting Step 1: Document Selection")
            step1_start = time.time()
            relevant_docs = self._select_relevant_documents(question, context)
            result['stage1'] = relevant_docs
            result['timestamps']['step1'] = time.time() - step1_start

            self.logger.info("Starting Step 2: Evidence Collection")
            step2_start = time.time()
            evidence = self._collect_evidence(question, relevant_docs['selected_docs'])
            result['stage2'] = evidence
            result['timestamps']['step2'] = time.time() - step2_start

            self.logger.info("Starting Step 3: Answer Generation")
            step3_start = time.time()
            final_answer = self._generate_answer(question, evidence['rationales'])
            result['stage3'] = final_answer
            result['timestamps']['step3'] = time.time() - step3_start

            result['total_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in evaluation process: {str(e)}")
            raise

    def _select_relevant_documents(self, question: str, context: Dict[str, str]) -> Dict[str, Any]:
        selected_docs = []
        predictions = []

        for doc_id, content in context.items():
            self.logger.debug(f"Evaluating document {doc_id}")
            pred = self._evaluate_relevance(question, content)
            if pred['is_relevant']:
                selected_docs.append({
                    'doc_id': doc_id,
                    'content': content
                })
            predictions.append({
                'doc_id': doc_id,
                'prediction': pred
            })

        return {
            'selected_docs': selected_docs,
            'predictions': predictions
        }

    def _evaluate_relevance(self, question: str, document: str) -> Dict[str, Any]:
        prompt = self._create_relevance_prompt(question, document)
        response = self.model_manager.call_with_retry(prompt)
        
        return {
            'is_relevant': 'yes' in response.lower(),
            'response': response
        }

    def _collect_evidence(self, question: str, documents: List[Dict[str, str]]) -> Dict[str, List]:
        rationales = []
        
        for doc in documents:
            self.logger.debug(f"Collecting evidence from document {doc['doc_id']}")
            prompt = self._create_evidence_prompt(question, doc['content'])
            response = self.model_manager.call_with_retry(prompt)
            
            if response and response.lower() != 'no':
                rationales.append({
                    'doc_id': doc['doc_id'],
                    'content': response
                })

        return {'rationales': rationales}

    def _generate_answer(self, question: str, rationales: List[Dict[str, str]]) -> str:
        if not rationales:
            return "No answer"
        
        prompt = self._create_answer_prompt(question, rationales)
        return self.model_manager.call_with_retry(prompt)

    def _create_relevance_prompt(self, question: str, document: str) -> str:
        return f"{self.instructions['step1']}\n###Question: {question}\n###Paragraph: {document}\n###Output: "

    def _create_evidence_prompt(self, question: str, document: str) -> str:
        return f"{self.instructions['step2']}\n###Question: {question}\n###Context: {document}\n###Relevant excerpt: "

    def _create_answer_prompt(self, question: str, rationales: List[Dict[str, str]]) -> str:
        rationale_texts = [r['content'] for r in rationales]
        context = "\n".join(rationale_texts)
        return f"{self.instructions['step3']}\nQuestion: {question}\nContext: {context}\nOutput: {{answer summary}}"

    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise