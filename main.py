# TIDES/main.py
import argparse
import logging
from pathlib import Path
import time
from typing import Dict, Any

from src.utils.config import ConfigManager
from src.utils.model_manager import ModelManager
from src.evaluation.evaluator import ResponseEvaluator
from src.retrieval.retriever import get_retriever
from src.data.dataset_utils import DatasetLoader

def setup_logging(output_dir: Path) -> None:
   log_dir = output_dir / 'logs'
   log_dir.mkdir(parents=True, exist_ok=True)
   
   timestamp = time.strftime('%Y%m%d_%H%M%S')
   log_file = log_dir / f'run_{timestamp}.log'
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s [%(levelname)s] %(message)s',
       handlers=[
           logging.FileHandler(log_file),
           logging.StreamHandler()
       ]
   )

def parse_arguments():
   parser = argparse.ArgumentParser(description='TIDES: Technical Information Document Extraction System')
   
   parser.add_argument('--dataset', choices=['techqa', 's10', 'smart_tv_remote'], required=True,
                     help='Dataset to process (techqa or s10 or smart_tv_remote)')
   parser.add_argument('--model-type', choices=['together', 'openai'], required=True,
                     help='Model service to use (together or openai)')
   parser.add_argument('--api-key', required=True,
                     help='API key for the model service')
   
   parser.add_argument('--model-name', type=str,
                     help='Specific model name (optional)')
   parser.add_argument('--retriever', choices=['tfidf', 'cosine'], default='tfidf',
                     help='Document retrieval method')
   parser.add_argument('--config', type=str,
                     help='Path to custom config file')
   parser.add_argument('--output-dir', type=str,
                     help='Custom output directory')
   parser.add_argument('--batch-size', type=int, default=1,
                     help='Batch size for processing')
   parser.add_argument('--start-idx', type=int, default=0,
                     help='Starting index for processing')
   parser.add_argument('--end-idx', type=int,
                     help='Ending index for processing')
   
   return parser.parse_args()

def process_dataset(args: argparse.Namespace, config: Dict[str, Any]):
   try:
       model_manager = ModelManager(
           model_type=args.model_type,
           api_key=args.api_key,
           model_name=args.model_name
       )
       
       retriever = get_retriever(args.retriever)
       evaluator = ResponseEvaluator(model_manager, args.dataset)
       dataset_loader = DatasetLoader(config)
       
       dataset = dataset_loader.load_dataset(args.dataset)
       questions = dataset['questions'][args.start_idx:args.end_idx]
       
       for idx, question in enumerate(questions, start=args.start_idx):
           try:
               logging.info(f"Processing question {idx}")
               
               if args.dataset == 'techqa':
                   docs = dataset_loader.load_documents(question['doc_ids'])
                   retrieved_docs = retriever.retrieve_documents(
                       question['title'] + ' ' + question['body'],
                       docs,
                       config['retrieval']['top_k']
                   )
               else:
                   retrieved_docs = retriever.retrieve_documents(
                       question['question'],
                       dataset['documents'],
                       config['retrieval']['top_k']
                   )
               
               result = evaluator.evaluate(
                   question['question'] if args.dataset in ['s10','smart_tv_remote'] else question['title'] + ' ' + question['body'],
                   docs if args.dataset == 'techqa' else dataset['documents'],
                   retrieved_docs
               )
               
               output_path = Path(config['output']['save_dir']) / f'result_{idx:03d}.json'
               evaluator.save_results(result, output_path)
               
           except Exception as e:
               logging.error(f"Error processing question {idx}: {str(e)}")
               continue
               
   except Exception as e:
       logging.error(f"Error in process_dataset: {str(e)}")
       raise

def main():
   args = parse_arguments()
   
   config_manager = ConfigManager(args)
   config = config_manager.get_config()
   
   output_dir = Path(config['output']['save_dir'])
   output_dir.mkdir(parents=True, exist_ok=True)
   
   setup_logging(output_dir)
   
   logging.info(f"Starting TIDES with configuration:")
   logging.info(f"Dataset: {args.dataset}")
   logging.info(f"Model: {args.model_type}")
   logging.info(f"Retriever: {args.retriever}")
   
   try:
       process_dataset(args, config)
       logging.info("Processing completed successfully!")

        logging.info("Starting evaluation...")
        metrics_calculator = MetricsCalculator()
        
        if args.dataset == 'techqa':
            metrics = metrics_calculator.evaluate_techqa(
                output_dir=output_dir,
                reference_dir='data/raw/techqa/TechQA/validation'
            )
        else:
            metrics = metrics_calculator.evaluate_manual(
                output_dir=output_dir,
                dataset_type=args.dataset
            )
            
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logging.info("Evaluation Results:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.2f}")
   except Exception as e:
       logging.error(f"Error in main process: {str(e)}")
       raise

if __name__ == "__main__":
   main()