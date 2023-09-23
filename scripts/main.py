"""
@author: dsmueller3760
Read documents for aerosapce engineers and do things with them.
"""
import os
import glob
import data_import
import queries
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

# Load API keys
load_dotenv(find_dotenv(),override=True)
# print(os.getenv('PINECONE_ENVIRONMENT'))
# print(os.getenv('PINECONE_API_KEY'))

# OpenAI
llm = OpenAI(temperature=0)
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=os.getenv('OPENAI_API_KEY'))

# Pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT') 
)
index_name = 'langchain-quickstart'

# Find all docs in data folder and import them
current_path=os.path.dirname(os.path.abspath(__file__))
data_folder='/../data/'
docs = glob.glob(current_path+data_folder+'*.pdf')   # Only get the PDFs in the directory

refresh=False
if refresh:
    data_import.load_docs(index_name,embeddings_model,docs)


# filter_list={'source':'AMS_2006.pdf'}
qa_model_obj=queries.QA_Model(index_name,
                    embeddings_model,
                    llm,
                    k=6,
                    search_type='mmr',
                    verbose=False,
                    filter=None)


query = 'What can you tell me about latch mechanism design failures which have occurred'
qa_model_obj.query_docs(query)

query_followup='Which one of the sources discussed volatile spherical joint interfaces'

# Response without filter
# qa_model_obj.query_docs(query_followup)

# Response with filter
# {'page':'GPT-4 Technical Report'}
# {"filter": {"source": ["source1", "source2"]}}
# {"filter": {"source": ["source1", "source2"]}}
#         search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
filter_list = list(set(item['source'] for item in qa_model_obj.sources[-1])) # Beware, if you have a typo it will run but not work (e.g. 'source:' instead of 'source')
# Filter by multiple found items in previous response: https://docs.pinecone.io/docs/metadata-filtered-search
filter_items=[]
for item in filter_list:
    filter_item={'source': item}
    filter_items.append(filter_item)
filter_dict={'$or':filter_items}

qa_model_obj.update_model(search_type='similarity',
                          filter=filter_dict)
qa_model_obj.query_docs(query_followup)