import os
import json
import re
import time
import argparse
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from openai import OpenAI

def get_text(text):
    sentences = [line for line in text.split('\n') if line and not line.startswith(('#', '>')) and not line.endswith(':')]
    return '\t'.join(sentences)

def TIDES(api_key, val_q_a_path, raw_text_root, tfidf_list_path_root, output_path_root, step1_instruction, step2_instruction, step3_instruction):
    client = OpenAI(api_key=api_key)

    with open(val_q_a_path, 'r') as f:
        a = json.load(f)

    for i, ele in tqdm(enumerate(a)):
        prediction = {}
        doc_ids = ele['DOC_IDS']
        question = ele['QUESTION_TITLE']
        question += ele['QUESTION_TEXT']
        answer = ele['ANSWER']
        answerable = ele['ANSWERABLE']

        prediction['question'] = question
        prediction['doc_ids'] = doc_ids
        prediction['gt'] = answer
        prediction['answerable'] = answerable

        start = time.time()

        ## STEP0 : TF-IDF document selection
        tfidf_list_path = os.path.join(tfidf_list_path_root, f'{str(i).zfill(3)}.json')
        with open(tfidf_list_path, 'r') as f:
            tfidf_list = json.load(f)

        ## STEP1 associative selection
        associative_pars = []
        predict_list = []
        gts = []
        for doc_id in doc_ids:
            raw_text_path = os.path.join(raw_text_root, f'{doc_id}.json')
            with open(raw_text_path, 'r') as fp:
                raw_text = json.load(fp)['text']
                paragraphs = raw_text.split('\n\n\n')
                ans_split = answer.split('\n\n\n')
            for paragraph in paragraphs:
                par = get_text(paragraph)

                if answer == "-" or answer == "":
                    gts.append('no')
                else:
                    ans_tmp = re.sub(r'[\n\n]+', ' ', answer)
                    par_tmp = re.sub(r'[\n\n]+', ' ', par)
                    paragraph_tmp = re.sub(r'[\n\n]+', ' ', paragraph)
                    if ans_tmp in par_tmp or answer in paragraph_tmp:
                        gts.append('yes')
                    else:
                        gts.append('no')

                if doc_id not in tfidf_list:
                    predict_list.append('no')
                    continue

                msg = step1_instruction + '\n###'
                msg += f'Question: {question}\n###'
                msg += f'Paragraph: {par}\n###'
                msg += 'Output: '

                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview", 
                    messages=[{"role": 'user', "content": msg}]
                )
                pred = response.choices[0].message.content

                pred = re.sub(r'[^\w\s]', '', pred).lower()

                if 'yes' in pred:
                    associative_pars.append(par)
                predict_list.append(pred)
        prediction['gts_location'] = gts
        prediction['stage1'] = predict_list
        prediction['associative_pars'] = associative_pars

        ## STEP2 Rationale Generation
        rationales = []
        for associatve_par in associative_pars:
            msg = step2_instruction + '\n###'
            msg += f'Question: {question}\n###'
            msg += f'Context: {associatve_par}\n###'
            msg += 'Relevant excerpt: '

            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": 'user', "content": msg}],
                temperature=0
            )
            pred = response.choices[0].message.content
            if pred != '':
                rationales.append(pred)
        prediction['stage2'] = rationales

        ## STEP3 Systematic Composition
        while 'No' in rationales:
            rationales.remove('No')
        msg = step3_instruction + '\n'
        msg += f'Question: {question}\n'
        msg += f'Context: {rationales}\n'
        msg += 'Output: {answer summary}'

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": 'user', "content": msg}],
            temperature=0
        )
        final_answer = response.choices[0].message.content
        prediction['stage3'] = final_answer
        end = time.time()
        output_path = os.path.join(output_path_root, f'TIDES_result{str(i).zfill(3)}.json')
        with open(output_path, 'w') as fp:
            json.dump(prediction, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TIDES')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--val_q_a_path', type=str, required=True, help='Path to validation reference JSON')
    parser.add_argument('--raw_text_root', type=str, required=True, help='Root path to raw text documents')
    parser.add_argument('--tfidf_list_path_root', type=str, required=True, help='Root path to TF-IDF lists')
    parser.add_argument('--output_path_root', type=str, required=True, help='Root path to save outputs')
    parser.add_argument('--step1_instruction', type=str, required=True, help='Instruction for Step 1')
    parser.add_argument('--step2_instruction', type=str, required=True, help='Instruction for Step 2')
    parser.add_argument('--step3_instruction', type=str, required=True, help='Instruction for Step 3')

    args = parser.parse_args()

    TIDES(
        api_key=args.api_key,
        val_q_a_path=args.val_q_a_path,
        raw_text_root=args.raw_text_root,
        tfidf_list_path_root=args.tfidf_list_path_root,
        output_path_root=args.output_path_root,
        step1_instruction=args.step1_instruction,
        step2_instruction=args.step2_instruction,
        step3_instruction=args.step3_instruction
    )
