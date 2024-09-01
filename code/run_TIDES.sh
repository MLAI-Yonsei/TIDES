
#!/bin/bash

# path and variable
VAL_Q_A_PATH='/path/to/validation_reference.json'
RAW_TEXT_ROOT='/path/to/document_val/'
TFIDF_OUTPUT_PATH='/path/to/tfidf_output/'
API_KEY="your-api-key"
TFIDF_LIST_PATH='/path/to/tfidf_list/'
RESULT_PATH='/path/to/result_path/'

# prompt
INSTRUCTION1='You’re an expert in IT and computer science. Provide a rationale in 20 words or less for how the paragraph relates to the question, focusing on key technical details. Finally, give a clear yes or no answer avoiding overconfidence.'

INSTRUCTION2='You’re an expert in IT and computer science. Given a question and a context, if the context contains information that directly answers the question or provides clear supporting evidence, extract only the relevant section. Do not include any additional explanation or comments. If no relevant information is found, simply respond "No".'

INSTRUCTION3='Compose a concise answer to the question using the most relevant key words and phrases from the rationales. Aim for a natural response while still aligning closely with the rationales. If the rationales do not sufficiently address the question, respond with "No answer".'

# Run TF-IDF
python /path/to/TFIDF.py --val_q_a_path $VAL_Q_A_PATH --raw_text_root $RAW_TEXT_ROOT --output_path_root $TFIDF_OUTPUT_PATH

# Run TIDES
python /path/to/TIDES.py --api_key $API_KEY --val_q_a_path $VAL_Q_A_PATH --raw_text_root $RAW_TEXT_ROOT --tfidf_list_path_root $TFIDF_LIST_PATH --output_path_root $RESULT_PATH --step1_instruction "$INSTRUCTION1" --step2_instruction "$INSTRUCTION2" --step3_instruction "$INSTRUCTION3"
