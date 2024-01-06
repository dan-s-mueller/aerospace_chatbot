import os
import glob
import re
import pinecone
import uuid
import json, jsonlines
from tqdm import tqdm
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document as lancghain_Document
from canopy.models.data_models import Document as canopy_Document

def load_docs(docs,
              index_name=None,
              embeddings_model=None,
              PINECONE_API_KEY=None,
              PINECONE_ENVIRONMENT=None,
              chunk_size=5000,
              chunk_overlap=0,
              clear=True,
              destination='langchain',
              file=None):
    """
    Loads PDF documents. If index_name is blank, it will return a list of the data (texts). If it is a name of a pinecone storage, it will return the vector_store.    
    destination: 
        'langchain' uses the Document object definition from langchain_core.documents import Document
        'canopy' uses the Document object definition from canopy.models.data_models import Document
    """

    if index_name:
        # Import and initialize Pinecone client
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT') 
        )

        # Find the existing index, clear for new start
        if clear:
            index=pinecone.Index(index_name)
            index.delete(delete_all=True) # Clear the index first, then upload
            # print('Cleared database.')

    # Read docs
    docs_out=[]
    for doc in tqdm(docs,desc='Reading and parsing docs'):
        # print('Parsing: '+doc)
        loader = PyPDFLoader(doc)
        data = loader.load_and_split()

        # This is optional, but needed to play with the data parsing.
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        pages = text_splitter.split_documents(data)

        # Tidy up text by removing unnecessary characters
        for page in pages:
            page.metadata['source']=os.path.basename(page.metadata['source'])   # Strip path
            page.metadata['page']=int(page.metadata['page'])+1   # Pages are 0 based, update
            # Merge hyphenated words
            page.page_content=re.sub(r"(\w+)-\n(\w+)", r"\1\2", page.page_content)
            # Fix newlines in the middle of sentences
            page.page_content = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", page.page_content.strip())
            # Remove multiple newlines
            page.page_content = re.sub(r"\n\s*\n", "\n\n", page.page_content)

            if destination=='canopy':   # text stored as text key
                doc_temp=canopy_Document(id=page.metadata['source']+"_"+str(page.metadata['page'])+str(uuid.uuid4()),
                                        text=page.page_content,
                                        source=page.metadata['source'],
                                        metadata={'page':str(page.metadata['page'])})
            elif destination=='langchain':  # text stored as page_content key
                doc_temp=lancghain_Document(page_content=page.page_content,
                                            source=page.metadata['source'],
                                            metadata=page.metadata)
            
            if has_meaningful_content(page,destination=destination):
                docs_out.append(doc_temp)
        
    if index_name:
        vectorstore = Pinecone.from_documents(docs_out, embeddings_model, index_name=index_name)

    if file:
        # Write to a jsonl file, save it.
        with jsonlines.open(file, mode='w') as writer:
            for doc in docs_out: 
                writer.write(doc.dict())

    if index_name:
        return vectorstore
    else:
        return docs_out

def update_database():
    # Executed when this module is run to update the database.
    # Pinecone and embeddings model
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENVIRONMENT') 
    )
    index_name = 'langchain-quickstart'
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=os.getenv('OPENAI_API_KEY'))

    # Find all docs in data folder and import them
    current_path=os.path.dirname(os.path.abspath(__file__))
    data_folder='/../data/'
    docs = glob.glob(current_path+data_folder+'*.pdf')   # Only get the PDFs in the directory
    load_docs(index_name,embeddings_model,docs)

def read_docs(file,destination='langchain'):
    """
    Reads the tile output from load_docs and formats it into a list of documents.
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    list_of_docs = []
    for line in lines:
        dict_ = json.loads(line)
        if destination=='canopy':   # text stored as text key
            doc_=canopy_Document(id=dict_['id'],
                                    text=dict_['page_content'],
                                    source=dict_['metadata']['source'],
                                    metadata=dict_['metadata'])
        elif destination=='langchain':  # text stored as page_content key
            doc_=lancghain_Document(page_content=dict_['page_content'],
                                        source=dict_['metadata']['source'],
                                        metadata=dict_['metadata'])
        list_of_docs.append(doc_)
    return list_of_docs


def has_meaningful_content(page,destination='langchain'):
    """
    Test whether the page has more than 30% words and is more than 5 words.
    """

    if destination=='langchain':
        text=page.page_content
    elif destination=='canopy':
        text=page.text
    
    num_words = len(text.split())
    alphanumeric_pct = sum(c.isalnum() for c in text) / len(text)
    if num_words < 5 or alphanumeric_pct < 0.3:
        return False
    else:
        return True