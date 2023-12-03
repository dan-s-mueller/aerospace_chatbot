import re
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
                                  'id':str(uuid.uuid4()),
                                  'answer':answer})

    if file:
        with jsonlines.open(file, mode='w') as writer:
            writer.write_all(qa_train_data)

    return qa_train_data