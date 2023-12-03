import re
from tqdm import tqdm
import uuid
import langchain.llms
import prompts


def gen_q_from_context(texts,
                       llm,
                       num_questions_per_chunk=1):
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

        # print("Processing text "+str(i_text))
        response = llm(query)

        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]

        for question in questions:
            qa_train_data.append({'question':question,
                                  'id':str(uuid.uuid4()),
                                  'answer':answer})
        # print("Completed processing of text "+str(i_text))
    return qa_train_data