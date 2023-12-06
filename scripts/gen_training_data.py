import re
import csv
from tqdm import tqdm
import uuid
import langchain.llms
import prompts
import jsonlines


def gen_q_from_context(texts,
                       llm,
                       num_questions_per_chunk=1,
                       file=None):
    """
    Generate num_questions_per_chunk given the input context.
    """

    queries = {}
    relevant_docs = {}
    qa_train_data=[]
    i_text=0
    for text in tqdm(texts):
        i_text=i_text+1
        answer=text.page_content+" "+str(text.metadata)
        query = prompts.QA_GENERATE_PROMPT.format(
            context_str=answer,
            num_questions_per_chunk=num_questions_per_chunk
        )

        response = llm(query)

        questions=[]
        results = str(response).strip().split("\n")
        for result in results:
            if result.startswith('QUESTION:'):
                questions.append(result.replace('QUESTION:',''))

        for question in questions:
            qa_train_data.append({'question':question,
                                  'answer':answer})

    if file:
        with jsonlines.open(file, mode='w') as writer:
            writer.write_all(qa_train_data)

    return qa_train_data

def format_dataset(format,
                   file_in=None,
                   training_data=None,
                   file_out=None):
    """
    Formats dataset for training in huggingface. Use either file or training_data, not both.
    """
    # Read from file if argument was used
    if file_in and training_data is None:
        training_data = []
        with jsonlines.open(file_in) as reader:
            for line in reader:
                training_data.append(line)
    
    # Format into datasets used to train llms on huggingface. Info here: https://huggingface.co/docs/autotrain/main/en/llm_finetuning
    if format=='LLM-generic':
        # Based on this reference dataset: https://huggingface.co/datasets/timdettmers/openassistant-guanaco
        tune_data=['text']
        for data in training_data:
            tune_data.append('### Human:'+data['question']+'### Assistant:'+data['answer'])
    
    if file_out:
        with open(file_out, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Write each line as a single cell in the CSV file
            for data in tune_data:
                writer.writerow([data])
    
    return tune_data