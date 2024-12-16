# TIDES/src/utils/config.py
import yaml
from pathlib import Path

def load_config(args):
    """설정 파일 로드 및 커맨드 라인 인자와 병합"""
    # 기본 설정 로드
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋별 설정 로드
    dataset_config_path = f'config/{args.dataset}_config.yaml'
    if Path(dataset_config_path).exists():
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
            config.update(dataset_config)
    
    # 사용자 지정 설정 파일이 있다면 로드
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            custom_config = yaml.safe_load(f)
            config.update(custom_config)
    
    # 커맨드 라인 인자로 설정 업데이트
    config['model']['type'] = args.model_type
    if args.model_name:
        config['model']['name'] = args.model_name
    config['model']['api_key'] = args.api_key
    config['retrieval']['method'] = args.retriever
    if args.top_k:
        config['retrieval']['top_k'] = args.top_k
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    
    return config

# # TIDES/config/config.py
# """설정 파일 관리"""

# TECHQA_CONFIG = {
#     'data': {
#         'base_path': 'data',
#         'validation_path': 'TechQA/validation/validation_reference.json',
#         'technotes_path': 'TechQA/validation/validation_technotes.json'
#     },
#     'retrieval': {
#         'top_k': 30
#     },
#     'model': {
#         'name': "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#         'max_tokens': 1000,
#         'temperature': 0
#     }
# }

# MANUAL_CONFIG = {
#     'data': {
#         'base_path': 'data',
#         'questions_path': 'smart_tv_remote_50_questions.csv',
#         'corpus_path': 'smart_tv_remote_manual_corpus.json'
#     },
#     'retrieval': {
#         'top_k': 30
#     },
#     'model': {
#         'name': "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#         'max_tokens': 1000,
#         'temperature': 0
#     }
# }

# def get_config(dataset_type):
#     if dataset_type == 'techqa':
#         return TECHQA_CONFIG
#     return MANUAL_CONFIG