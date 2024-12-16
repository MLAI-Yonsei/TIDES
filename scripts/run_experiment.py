# TIDES/scripts/run_experiment.py
import argparse
import json
from pathlib import Path
import logging
from datetime import datetime

def setup_logging(output_dir):
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'experiment_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def run_experiment(args):
    from src.utils.config import load_config
    from src.utils.model_manager import ModelManager
    from src.evaluation.evaluator import ResponseEvaluator
    from src.retrieval.retriever import TFIDFRetriever, CosineRetriever
    from src.data.dataset_utils import DatasetLoader

    config = load_config(args)
    setup_logging(config['data']['output_dir'])
    
    logging.info(f"Starting experiment with config: {config}")
    
    try:
        model_manager = ModelManager(
            model_type=args.model_type,
            api_key=args.api_key,
            model_name=args.model_name
        )
        
        retriever = TFIDFRetriever() if args.retriever == 'tfidf' else CosineRetriever()
        evaluator = ResponseEvaluator(model_manager, args.dataset)
        dataset_loader = DatasetLoader(config)
        
        if args.dataset == 'techqa':
            process_techqa(args, config, model_manager, evaluator, retriever, dataset_loader)
        elif args.dataset == 'smart_tv_remote':
            process_smart_tv_remote(args, config, model_manager, evaluator, retriever, dataset_loader)
        else:
            process_s10(args, config, model_manager, evaluator, retriever, dataset_loader)
            
        logging.info("Experiment completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during experiment: {str(e)}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description='Run TIDES experiment')
    
    parser.add_argument('--dataset', choices=['techqa', 'smart_tv_remote','s10'], required=True,
                      help='Dataset to process')
    parser.add_argument('--retriever', choices=['tfidf', 'cosine'], default='tfidf',
                      help='Document retrieval method')
    parser.add_argument('--model-type', choices=['together', 'openai'], required=True,
                      help='Model service type')
    parser.add_argument('--model-name', type=str,
                      help='Specific model name')
    parser.add_argument('--api-key', type=str, required=True,
                      help='API key for model service')
    
    parser.add_argument('--config', type=str,
                      help='Path to custom config file')
    parser.add_argument('--output-dir', type=str,
                      help='Output directory')
    parser.add_argument('--start-idx', type=int, default=0,
                      help='Starting index for processing')
    parser.add_argument('--end-idx', type=int,
                      help='Ending index for processing')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for processing')
    
    args = parser.parse_args()
    run_experiment(args)

if __name__ == '__main__':
    main()