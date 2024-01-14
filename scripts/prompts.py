from langchain import hub

CONDENSE_QUESTION_PROMPT = hub.pull("dmueller/ams-chatbot-qa-condense-history")
QA_PROMPT=hub.pull("dmueller/ams-chatbot-qa-retrieval")
QA_WSOURCES_PROMPT=hub.pull("dmueller/ams-chatbot-qa-retrieval-wsources")
QA_GENERATE_PROMPT=hub.pull("dmueller/generate_qa_prompt")