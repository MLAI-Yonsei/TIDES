# TIDES/scripts/run_all_experiments.sh
#!/bin/bash

# OpenAI GPT-3.5 실험
echo "Running experiment with GPT-3.5..."
python scripts/run_experiment.py \
    --dataset techqa \
    --model-type openai \
    --model-name gpt-3.5-turbo-0125 \
    --api-key $OPENAI_API_KEY \
    --retriever tfidf \
    --output-dir results/techqa_gpt35

# Together AI LLaMA 실험
echo "Running experiment with LLaMA..."
python scripts/run_experiment.py \
    --dataset techqa \
    --model-type together \
    --model-name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --api-key $TOGETHER_API_KEY \
    --retriever tfidf \
    --output-dir results/techqa_llama

# Manual 데이터셋 실험
echo "Running experiment with manual dataset..."
python scripts/run_experiment.py \
    --dataset manual \
    --model-type together \
    --model-name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --api-key $TOGETHER_API_KEY \
    --retriever cosine \
    --output-dir results/manual_llama