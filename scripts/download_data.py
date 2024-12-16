# TIDES/scripts/download_data.py
import argparse
import requests
import tarfile
from pathlib import Path
import logging
import time

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            start_time = time.time()
            for data in response.iter_content(chunk_size=8192):
                downloaded += len(data)
                f.write(data)
                
                done = int(50 * downloaded / total_size)
                elapsed_time = time.time() - start_time
                speed = downloaded / (1024 * 1024 * elapsed_time)  # MB/s
                
                print(f"\r{desc}: [{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes "
                      f"({speed:.2f} MB/s)", end='')
    print()

def download_techqa():
    output_dir = Path('data/raw/techqa')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://huggingface.co/datasets/PrimeQA/TechQA/resolve/main/TechQA.tar.gz"
    tar_path = output_dir / 'TechQA.tar.gz'
    
    try:
        logging.info("Downloading TechQA dataset...")
        download_file(url, tar_path, "Downloading TechQA")
        
        logging.info("Extracting files...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
        
        tar_path.unlink()
        logging.info("TechQA dataset download completed!")
        
    except Exception as e:
        logging.error(f"Error downloading TechQA dataset: {str(e)}")
        raise

def download_smart_tv_remote():
    output_dir = Path('data/raw/smart_tv_remote')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://raw.githubusercontent.com/abhi1nandy2/EMNLP-2021-Findings/main/data"
    files = {
        'smart_tv_remote_manual_corpus.json': f"{base_url}/smart_tv_remote_manual_corpus.json",
        'smart_tv_remote_50_questions.csv': f"{base_url}/smart_tv_remote_50_questions.csv"
    }
    
    try:
        for filename, url in files.items():
            output_path = output_dir / filename
            logging.info(f"Downloading {filename}...")
            download_file(url, output_path, f"Downloading {filename}")
            
        logging.info("Smart tv remote dataset download completed!")
        
    except Exception as e:
        logging.error(f"Error downloading manual dataset: {str(e)}")
        raise

def download_s10():
    output_dir = Path('data/raw/s10')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://raw.githubusercontent.com/abhi1nandy2/EMNLP-2021-Findings/main/data"
    files = {
        's10_manual_corpus.json': f"{base_url}/s10_manual_corpus.json",
        's10_50_questions.csv': f"{base_url}/s10_50_questions.csv"
    }
    
    try:
        for filename, url in files.items():
            output_path = output_dir / filename
            logging.info(f"Downloading {filename}...")
            download_file(url, output_path, f"Downloading {filename}")
            
        logging.info("S10 dataset download completed!")
        
    except Exception as e:
        logging.error(f"Error downloading s10 dataset: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Download TIDES datasets')
    parser.add_argument('--dataset', choices=['techqa', 's10','smart_tv_remote'], required=True,
                      help='Dataset to download')
    
    args = parser.parse_args()
    setup_logging()
    
    if args.dataset == 'techqa':
        download_techqa()
    elif args.dataset == 's10':
        download_s10()
    else:
        download_smart_tv_remote()

if __name__ == '__main__':
    main()