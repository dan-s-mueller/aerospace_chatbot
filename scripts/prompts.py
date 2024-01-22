from langchain import hub
from langchain.prompts.prompt import PromptTemplate

# Prompts on the hub: https://smith.langchain.com/hub/my-prompts?organizationId=45eb8917-7353-4296-978d-bb461fc45c65
CONDENSE_QUESTION_PROMPT = hub.pull("dmueller/ams-chatbot-qa-condense-history")
QA_PROMPT=hub.pull("dmueller/ams-chatbot-qa-retrieval")
QA_WSOURCES_PROMPT=hub.pull("dmueller/ams-chatbot-qa-retrieval-wsources")
QA_GENERATE_PROMPT=hub.pull("dmueller/generate_qa_prompt")

# Prompts defined here only
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
TEST_QUERY_PROMPT='What are examples of adhesives to use when potting motors for launch vehicle or spacecraft mechanisms?'