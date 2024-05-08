from langchain import hub
from langchain.prompts.prompt import PromptTemplate

# Prompts on the hub: https://smith.langchain.com/hub/my-prompts?organizationId=45eb8917-7353-4296-978d-bb461fc45c65
CONDENSE_QUESTION_PROMPT = hub.pull("dmueller/ams-chatbot-qa-condense-history")
QA_PROMPT=hub.pull("dmueller/ams-chatbot-qa-retrieval")
QA_WSOURCES_PROMPT=hub.pull("dmueller/ams-chatbot-qa-retrieval-wsources")
QA_GENERATE_PROMPT=hub.pull("dmueller/generate_qa_prompt")
SUMMARIZE_TEXT=hub.pull("dmueller/summarize_text")
TEST_QUERY_PROMPT='What are examples of adhesives to use when potting motors for launch vehicle or spacecraft mechanisms?'
GENERATE_SIMILAR_QUESTIONS=hub.pull("dmueller/generate_similar_questions")
GENERATE_SIMILAR_QUESTIONS_W_CONTEXT=hub.pull("dmueller/generate_similar_questions_w_context")
CLUSTER_LABEL=hub.pull("dmueller/cluster-label")

# Prompts defined here only
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")