from prompts import SUMMARIZE_TEXT

import os, logging, re, shutil, random
import uuid
from pathlib import Path
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from pinecone import Pinecone as pinecone_client
from pinecone import PodSpec

import chromadb
from chromadb import PersistentClient

import json, jsonlines

import streamlit as st

from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from ragatouille import RAGPretrainedModel

from ragxplorer import RAGxplorer, rag
import pandas as pd


def load_docs(index_type:str,
              docs,
              query_model,
              rag_type:str='Standard',
              index_name:str=None,
              chunk_method:str='character_recursive',
              chunk_size:int=500,
              chunk_overlap:int=0,
              clear:bool=False,
              file_out:str=None,
              batch_size:int=50,
              local_db_path:str='.',
              llm=None,
              show_progress:bool=False):
    """
    Loads documents into the specified index.

    Args:
        index_type (str): The type of index to use.
        docs: The documents to load.
        query_model: The query model to use.
        rag_type (str, optional): The type of RAG (Retrieval-Augmented Generation) to use. Defaults to 'Standard'.
        index_name (str, optional): The name of the index. Defaults to None.
        chunk_method (str, optional): The method to chunk the documents. Defaults to 'character_recursive'.
        chunk_size (int, optional): The size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 0.
        clear (bool, optional): Whether to clear the index before loading new documents. Defaults to False.
        file_out (str, optional): The output file path. Defaults to None.
        batch_size (int, optional): The batch size for upserting documents. Defaults to 50.
        local_db_path (str, optional): The local database path. Defaults to '../../db'.
        llm (optional): The language model to use. Defaults to None.
        show_progress (bool, optional): Whether to show progress during the loading process. Defaults to False.

    Returns:
        vectorstore: The updated vectorstore.
    """
    # Check for illegal things
    if not clear and (rag_type == 'Parent-Child' or rag_type == 'Summary'):
        raise ValueError('Parent-Child databases must be cleared before loading new documents.')

    # Chunk docs
    chunker=chunk_docs(docs,
                       rag_type=rag_type,
                       chunk_method=chunk_method,
                       chunk_size=chunk_size,
                       chunk_overlap=chunk_overlap,
                       file_out=file_out,
                       llm=llm,
                       show_progress=show_progress)
        
    # Set index names for special databases
    if rag_type == 'Parent-Child':
        index_name = index_name + '-parent-child'
    if rag_type == 'Summary':
        index_name = index_name + '-summary-' + llm.model_name.replace('/', '-')

    # Initialize client an upsert docs
    vectorstore = initialize_database(index_type, 
                                      index_name, 
                                      query_model, 
                                      rag_type=rag_type,
                                      clear=clear, 
                                      local_db_path=local_db_path,
                                      init_ragatouille=True,
                                      show_progress=show_progress)
    vectorstore, retriever = upsert_docs(index_type,
                                         index_name,
                                         vectorstore,
                                         chunker,
                                         batch_size=batch_size,
                                         show_progress=show_progress,
                                         local_db_path=local_db_path)
    logging.info(f"Documents upserted to {index_name}.")
    return vectorstore
def chunk_docs(docs: List[str],
               rag_type:str='Standard',
               chunk_method:str='character_recursive',
               file_out:str=None,
               chunk_size:int=500,
               chunk_overlap:int=0,
               k_parent:int=4,
               llm=None,
               show_progress:bool=False):
    """
    Chunk the given list of documents into smaller chunks based on the specified parameters.

    Args:
        docs (List[str]): The list of document paths to be chunked.
        rag_type (str, optional): The type of chunking method to be used. Defaults to 'Standard'.
        chunk_method (str, optional): The method of chunking to be used. Defaults to 'character_recursive'.
        file_out (str, optional): The output file path to save the chunked documents. Defaults to None.
        chunk_size (int, optional): The size of each chunk in tokens. Defaults to 500.
        chunk_overlap (int, optional): The overlap between chunks in tokens. Defaults to 0.
        k_parent (int, optional): The number of parent chunks to split into child chunks for 'Parent-Child' rag_type. Defaults to 4.
        llm (None, optional): The language model to be used for generating summaries. Defaults to None.
        show_progress (bool, optional): Whether to show the progress bar during chunking. Defaults to False.

    Returns:
        dict: A dictionary containing the chunking results based on the specified rag_type.
    """
    if show_progress:
        progress_text = "Chunking in progress..."
        my_bar = st.progress(0, text=progress_text)
    pages=[]
    chunks=[]
    logging.info('No jsonl found. Reading and parsing docs.')
    logging.info('Chunk size (tokens): '+str(chunk_size))
    logging.info('Chunk overlap (tokens): '+str(chunk_overlap))

    # Parse doc pages
    for i, doc in enumerate(docs):
        logging.info('Parsing: '+doc)
        loader = PyPDFLoader(doc)
        page_data = loader.load()

        # Clean up page info, update some metadata
        for page in page_data:
            page=_sanitize_raw_page_data(page)
            if page is not None:
                pages.append(page)
        if show_progress:
            progress_percentage = i / len(docs)
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
    
    # Process pages
    if rag_type=='Standard': 
        if chunk_method=='character_recursive':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise NotImplementedError
        page_chunks = text_splitter.split_documents(pages)

        for chunk in page_chunks:
            chunk.page_content += str(chunk.metadata)    # Add metadata to the end of the page content, some RAG models don't have metadata.
            chunks.append(chunk)    # Not sanitized because the page already was
        logging.info('Parsed: '+doc)
        logging.info('Sample entries:')
        logging.info(str(chunks[0]))
        logging.info(str(chunks[-1]))
        if file_out:
            # Write to a jsonl file, save it.
            logging.info('Writing to jsonl file: '+file_out)
            with jsonlines.open(file_out, mode='w') as writer:
                for doc in chunks: 
                    writer.write(doc.dict())
            logging.info('Written: '+file_out)
        if show_progress:
            my_bar.empty()
        return {'rag':'Standard',
                'pages':pages,
                'chunks':chunks,
                'splitters':text_splitter}
    elif rag_type=='Parent-Child': 
        if chunk_method=='character_recursive':
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size*k_parent, chunk_overlap=chunk_overlap)
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise NotImplementedError
        
        # Split up parent chunks
        parent_chunks = parent_splitter.split_documents(pages)
        doc_ids = [str(uuid.uuid4()) for _ in parent_chunks]
        
        # Split up child chunks
        id_key = "doc_id"
        chunks = []
        for i, doc in enumerate(parent_chunks):
            _id = doc_ids[i]
            _chunks = child_splitter.split_documents([doc])
            for _doc in _chunks:
                _doc.metadata[id_key] = _id
            chunks.extend(_chunks)

        if show_progress:
            my_bar.empty()
        return {'rag':'Parent-Child',
                'pages':{'doc_ids':doc_ids,'parent_chunks':parent_chunks},
                'chunks':chunks,
                'splitters':{'parent_splitter':parent_splitter,'child_splitter':child_splitter}}
    elif rag_type == 'Summary':
        if show_progress:
            my_bar.empty()
            my_bar = st.progress(0, text='Generating summaries...')

        id_key = "doc_id"
        doc_ids = [str(uuid.uuid4()) for _ in pages]
        chain = (
            {"doc": lambda x: x.page_content}
            | SUMMARIZE_TEXT
            | llm
            | StrOutputParser()
        )

        summaries = []
        for i, page in enumerate(pages):
            summary = chain.invoke(page)
            summaries.append(summary)
            if show_progress:
                progress_percentage = i / len(pages)
                my_bar.progress(progress_percentage, text=f'Generating summaries...{progress_percentage*100:.2f}%')

        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]
        if show_progress:
            my_bar.empty()
        return {'rag':'Summary',
                'pages':{'doc_ids':doc_ids,'docs':pages},
                'summaries':summary_docs,
                'llm':llm}
    else:
        raise NotImplementedError
def initialize_database(index_type: str, 
                        index_name: str, 
                        query_model: str, 
                        rag_type: str,
                        local_db_path: str = None, 
                        clear: bool = False,
                        init_ragatouille: bool = False,
                        show_progress: bool = False):
    """Initializes the database based on the specified parameters.

    Args:
        index_type (str): The type of index to use (e.g., "Pinecone", "ChromaDB", "RAGatouille").
        index_name (str): The name of the index.
        query_model (str): The query model to use.
        rag_type (str): The type of RAG model to use.
        local_db_path (str, optional): The path to the local database. Defaults to None.
        clear (bool, optional): Whether to clear the index. Defaults to False.
        init_ragatouille (bool, optional): Whether to initialize the RAGatouille model. Defaults to False.
        show_progress (bool, optional): Whether to show the progress bar. Defaults to False.

    Returns:
        vectorstore: The initialized vector store.

    Raises:
        NotImplementedError: If the specified index type is not implemented.
    """
    
    if show_progress:
        progress_text = "Database initialization..."
        my_bar = st.progress(0, text=progress_text)

    if index_type == "Pinecone":

        if clear:
            delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)
        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        
        try:
            pc.describe_index(index_name)
        except:
            pc.create_index(index_name,
                            dimension=_embedding_size(query_model),
                            spec=PodSpec(environment="us-west1-gcp", pod_type="p1.x1"))
        
        index = pc.Index(index_name)
        vectorstore=PineconeVectorStore(index,
                                        index_name=index_name, 
                                        embedding=query_model,
                                        text_key='page_content',
                                        pinecone_api_key=os.getenv('PINECONE_API_KEY'))
        if show_progress:
            progress_percentage = 1
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
    elif index_type == "ChromaDB":
        if clear:
            delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)
        persistent_client = chromadb.PersistentClient(path=os.path.join(local_db_path,'chromadb'))            
        vectorstore = Chroma(client=persistent_client,
                                collection_name=index_name,
                                embedding_function=query_model)     
        if show_progress:
            progress_percentage = 1
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')   
    elif index_type == "RAGatouille":
        if clear:
            delete_index(index_type, index_name, rag_type, local_db_path=local_db_path)
        if init_ragatouille:    # Used if the index is not already set
            vectorstore = RAGPretrainedModel.from_pretrained(query_model,verbose=0)
        else:   # Used if the index is already set
            vectorstore = query_model    # The index is picked up directly.
        if show_progress:
            progress_percentage = 1
            my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
    else:
        raise NotImplementedError

    if show_progress:
        my_bar.empty()
    return vectorstore
@retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=1,max=60))
def upsert_docs(index_type: str, 
                index_name: str,
                vectorstore: any, 
                chunker: dict, 
                batch_size: int = 50, 
                show_progress: bool = False,
                local_db_path: str = '.'):
    """
    Upserts documents into the specified index. Uses tenacity with exponential backoff to retry upserting documents.

    Args:
        index_type (str): The type of index to upsert the documents into.
        index_name (str): The name of the index.
        vectorstore (any): The vectorstore object to add documents to.
        chunker (dict): The chunker dictionary containing the documents to upsert.
        batch_size (int, optional): The batch size for upserting documents. Defaults to 50.
        show_progress (bool, optional): Whether to show progress during the upsert process. Defaults to False.
        local_db_path (str, optional): The local path to the database folder. Defaults to '.'.

    Returns:
        tuple: A tuple containing the updated vectorstore and retriever objects.
    """
    if show_progress:
        progress_text = "Upsert in progress..."
        my_bar = st.progress(0, text=progress_text)

    if chunker['rag'] == 'Standard':
        # Upsert each chunk in batches
        if index_type == "Pinecone":
            for i in range(0, len(chunker['chunks']), batch_size):
                chunk_batch = chunker['chunks'][i:i + batch_size]
                vectorstore.add_documents(chunk_batch)
                if show_progress:
                    progress_percentage = i / len(chunker['chunks'])
                    my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
            logging.info(f"Index created: {vectorstore}")
            retriever = vectorstore.as_retriever()
        elif index_type == "ChromaDB":
            for i in range(0, len(chunker['chunks']), batch_size):
                chunk_batch = chunker['chunks'][i:i + batch_size]
                vectorstore.add_documents(chunk_batch)
                if show_progress:
                    progress_percentage = i / len(chunker['chunks'])
                    my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
            logging.info(f"Index created: {vectorstore}")
            retriever = vectorstore.as_retriever()
        elif index_type == "RAGatouille":
            logging.info(f"Creating index {index_name} from RAGatouille.")
            # Create an index from the vectorstore.
            vectorstore.index(
                collection=[chunk.page_content for chunk in chunker['chunks']],
                index_name=index_name,
                max_document_length=chunker['splitters']._chunk_size,
                overwrite_index=True,
                split_documents=True,
            )
            logging.info(f"Index created: {vectorstore}")

            # Move the directory to the db folder
            try:
                shutil.move('.ragatouille', local_db_path)
            except shutil.Error:
                pass    # If it already exists, don't do anything
            retriever = vectorstore.as_langchain_retriever()
        else:
            raise NotImplementedError
    elif chunker['rag'] == 'Parent-Child':
        if index_type == "ChromaDB" or index_type == "Pinecone":
            lfs_path = Path(local_db_path).resolve() / 'local_file_store' / index_name
            store = LocalFileStore(lfs_path)
            
            id_key = "doc_id"
            retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=store, id_key=id_key)

            for i in range(0, len(chunker['chunks']), batch_size):
                chunk_batch = chunker['chunks'][i:i + batch_size]
                retriever.vectorstore.add_documents(chunk_batch)
                if show_progress:
                    progress_percentage = i / len(chunker['chunks'])
                    my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
            
            # Index parent docs all at once
            retriever.docstore.mset(list(zip(chunker['pages']['doc_ids'], chunker['pages']['parent_chunks'])))
        elif index_type == "RAGatouille":
            raise Exception('RAGAtouille only supports standard RAG.')
        else:
            raise NotImplementedError
    elif chunker['rag'] == 'Summary':
        if index_type == "ChromaDB" or index_type == "Pinecone":
            lfs_path = Path(local_db_path).resolve() / 'local_file_store' / index_name
            store = LocalFileStore(lfs_path)
            
            id_key = "doc_id"
            retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=store, id_key=id_key)

            for i in range(0, len(chunker['summaries']), batch_size):
                summary_batch = chunker['summaries'][i:i + batch_size]
                retriever.vectorstore.add_documents(summary_batch)
                if show_progress:
                    progress_percentage = i / len(chunker['summaries'])
                    my_bar.progress(progress_percentage, text=f'{progress_text}{progress_percentage*100:.2f}%')
            
            # Index parent docs all at once
            retriever.docstore.mset(list(zip(chunker['pages']['doc_ids'], chunker['pages']['docs'])))
        elif index_type == "RAGatouille":
            raise Exception('RAGAtouille only supports standard RAG.')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    if show_progress:
        my_bar.empty()
    return vectorstore, retriever
def delete_index(index_type: str, 
                 index_name: str, 
                 rag_type: str,
                 local_db_path: str = '.'):
    """
    Deletes an index based on the specified index type.

    Args:
        index_type (str): The type of index to delete. Valid values are "Pinecone", "ChromaDB", or "RAGatouille".
        index_name (str): The name of the index to delete.
        rag_type (str): The type of RAG (RAGatouille) to delete. Valid values are "Parent-Child" or "Summary".
        local_db_path (str, optional): The path to the local database. Defaults to '.'.

    Raises:
        NotImplementedError: If the index_type is not supported.

    """
    if index_type == "Pinecone":
        pc = pinecone_client(api_key=os.getenv('PINECONE_API_KEY'))
        try:
            pc.describe_index(index_name)
            pc.delete_index(index_name)
        except Exception as e:
            # print(f"Error occurred while deleting Pinecone index: {e}")
            pass
        if rag_type == 'Parent-Child' or rag_type == 'Summary':
            try:
                shutil.rmtree(Path(local_db_path).resolve() / 'local_file_store' / index_name)
            except Exception as e:
                # print(f"Error occurred while deleting ChromaDB local_file_store collection: {e}")
                pass    # No need to do anything if it doesn't exist
    elif index_type == "ChromaDB":  
        try:
            persistent_client = chromadb.PersistentClient(path=os.path.join(local_db_path,'chromadb'))
            indices = persistent_client.list_collections()
            for idx in indices:
                if index_name in idx.name:
                    persistent_client.delete_collection(name=idx.name)
        except Exception as e:
            # print(f"Error occurred while deleting ChromaDB collection: {e}")
            pass
        # Delete local file store if they exist
        if rag_type == 'Parent-Child' or rag_type == 'Summary':
            try:
                shutil.rmtree(Path(local_db_path).resolve() / 'local_file_store' / index_name)
            except Exception as e:
                # print(f"Error occurred while deleting ChromaDB local_file_store collection: {e}")
                pass    # No need to do anything if it doesn't exist
    elif index_type == "RAGatouille":
        try:
            ragatouille_path = os.path.join(local_db_path, '.ragatouille')
            shutil.rmtree(ragatouille_path)
        except Exception as e:
            # print(f"Error occurred while deleting RAGatouille index: {e}")
            pass
    else:
        raise NotImplementedError
def reduce_vector_query_size(rx_client:RAGxplorer,
                             chroma_client:chromadb,
                             vector_qty:int,
                             verbose:bool=False):
    """
    Reduces the number of vectors in the RAGxplorer client to a specified quantity.

    Args:
        rx_client (RAGxplorer): The RAGxplorer client.
        chroma_client (chromadb): The chromadb client.
        vector_qty (int): The desired number of vectors.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        RAGxplorer: The updated RAGxplorer client with the reduced number of vectors.
    """

    ids = rx_client._vectordb.get()['ids']
    embeddings = rag.get_doc_embeddings(rx_client._vectordb)
    text = rag.get_docs(rx_client._vectordb)

    if verbose:
        print('Reducing the number of vectors from '+str(len(embeddings))+' to '+str(vector_qty)+'...')
    indices = random.sample(range(len(embeddings)), vector_qty)
    id = str(uuid.uuid4())[:8]
    temp_index_name=rx_client._vectordb.name+'-'+id
    
    # Create a temporary index with the reduced number of vectors
    chroma_client.create_collection(name=temp_index_name,embedding_function=rx_client._chosen_embedding_model)
    temp_collection = chroma_client.get_collection(name=temp_index_name,embedding_function=rx_client._chosen_embedding_model)
    temp_collection.add(
        ids=[ids[i] for i in indices],
        embeddings=[embeddings[i] for i in indices],
        documents=[text[i] for i in indices]
    )

    # Replace the original index with the temporary one
    rx_client._vectordb = temp_collection
    rx_client._documents.embeddings = rag.get_doc_embeddings(rx_client._vectordb)
    rx_client._documents.text = rag.get_docs(rx_client._vectordb)
    rx_client._documents.ids = rx_client._vectordb.get()['ids']

    if verbose:
        print('Reduced number of vectors to '+str(len(rx_client._documents.embeddings))+' ✓')
    return rx_client
def export_data_viz(rx_client: RAGxplorer, df_export_path: str):
    """
    Exports the visualization data to a JSON file.

    Args:
        rx_client (RAGxplorer): The RAGxplorer object containing the visualization data.
        df_export_path (str): The file path to export the JSON data.

    Returns:
        None
    """
    export_data = {
        'visualization_index_name': rx_client._vectordb.name,
        'umap_params': rx_client._projector.get_params(),
        'viz_data': rx_client._VizData.base_df.to_json(orient='split')
    }

    # Save the data to a JSON file
    with open(df_export_path, 'w') as f:
        json.dump(export_data, f, indent=4)
def create_data_viz(index_selected: str,
                    rx_client: RAGxplorer,
                    chroma_client: PersistentClient,
                    umap_params: dict = {'n_neighbors': 5, 'n_components': 2, 'random_state': 42},
                    limit_size_qty: int = None,
                    df_export_path: str = None,
                    show_progress: bool = False):
    """
    Creates data visualization using RAGxplorer and PersistentClient.

    Args:
        index_selected (str): The name of the collection to load from chroma_client.
        rx_client (RAGxplorer): An instance of RAGxplorer class.
        chroma_client (PersistentClient): An instance of PersistentClient class.
        umap_params (dict, optional): UMAP parameters for embedding. Defaults to {'n_neighbors': 5, 'n_components': 2, 'random_state': 42}.
        limit_size_qty (int, optional): The maximum number of vectors to include in the visualization. Defaults to None.
        df_export_path (str, optional): The file path to export the visualization data. Defaults to None.
        show_progress (bool, optional): Whether to show progress bar. Defaults to False.

    Returns:
        tuple: A tuple containing the updated rx_client and chroma_client instances.
    """
    if show_progress:
        my_bar = st.progress(0, text='Loading collection...')
    
    # Load collection from chroma_client
    collection = chroma_client.get_collection(name=index_selected,
                                              embedding_function=rx_client._chosen_embedding_model)
    rx_client.load_chroma(collection,
                          umap_params=umap_params,
                          initialize_projector=True)
    
    if limit_size_qty:
        if show_progress:
            my_bar.progress(0.25, text='Reducing vector query size...')
        
        # Reduce vector query size
        rx_client = reduce_vector_query_size(rx_client,
                                             chroma_client,
                                             vector_qty=limit_size_qty,
                                             verbose=True)
    
    if show_progress:
        my_bar.progress(0.5, text='Projecting embeddings for visualization...')
    
    # Run projector
    rx_client.run_projector()
    
    if show_progress:
        my_bar.progress(1, text='Projecting complete!')
    
    if df_export_path:
        # Export data visualization
        export_data_viz(rx_client, df_export_path)
    
    if show_progress:
        my_bar.empty()
    
    return rx_client, chroma_client
def visualize_data(index_selected: str,
                   rx_client: RAGxplorer,
                   chroma_client: PersistentClient,
                   query: str,
                   umap_params: dict = {'n_neighbors': 5, 'n_components': 2, 'random_state': 42},
                   import_file: bool = True):
    """
    Visualizes data using RAGxplorer and PersistentClient.

    Args:
        index_selected (str): The name of the selected index.
        rx_client (RAGxplorer): An instance of the RAGxplorer class.
        chroma_client (PersistentClient): An instance of the PersistentClient class.
        query (str): The query to visualize.
        umap_params (dict, optional): UMAP parameters for data visualization. Defaults to {'n_neighbors': 5, 'n_components': 2, 'random_state': 42}.
        import_file (bool, optional): Whether to import data from a file. Defaults to True.

    Returns:
        Tuple[RAGxplorer, PersistentClient]: A tuple containing the updated instances of RAGxplorer and PersistentClient.
    """
    if import_file:
        with open(import_file, 'r') as f:
            data = json.load(f)
        viz_data = pd.read_json(data['viz_data'], orient='split')
        collection = chroma_client.get_collection(name=index_selected, embedding_function=rx_client._chosen_embedding_model)
        rx_client.load_chroma(collection, umap_params=umap_params, initialize_projector=True)
        fig = rx_client.visualize_query(query, import_projection_data=viz_data)
    else:
        fig = rx_client.visualize_query(query)
    st.plotly_chart(fig, use_container_width=True)

    return rx_client, chroma_client
def _sanitize_raw_page_data(page):
    """
    Sanitizes the raw page data by removing unnecessary information and checking for meaningful content.

    Args:
        page (Page): The raw page data to be sanitized.

    Returns:
        Page or None: The sanitized page data if it contains meaningful content, otherwise None.
    """
    
    # Yank out some things you'll never care about
    page.metadata['source'] = os.path.basename(page.metadata['source'])   # Strip path
    page.metadata['page'] = int(page.metadata['page']) + 1   # Pages are 0 based, update
    page.page_content = re.sub(r"(\w+)-\n(\w+)", r"\1\2", page.page_content)   # Merge hyphenated words
    page.page_content = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", page.page_content.strip())  # Fix newlines in the middle of sentences
    page.page_content = re.sub(r"\n\s*\n", "\n\n", page.page_content)   # Remove multiple newlines

    # Test if there is meaningful content
    text = page.page_content
    num_words = len(text.split())
    if len(text) == 0:
        return None
    alphanumeric_pct = sum(c.isalnum() for c in text) / len(text)
    if num_words < 5 or alphanumeric_pct < 0.3:
        return None
    else:
        return page
def _embedding_size(embedding_model:any):
    """
    Returns the size of the embedding for a given embedding model.

    Args:
        embedding_model (object): The embedding model to get the size for.

    Returns:
        int: The size of the embedding.

    Raises:
        NotImplementedError: If the embedding model is not supported.
    """
    if isinstance(embedding_model,OpenAIEmbeddings):
        return 1536 # https://platform.openai.com/docs/models/embeddings, test-embedding-ada-002
    elif isinstance(embedding_model,VoyageAIEmbeddings):
        return 1024 # https://docs.voyageai.com/embeddings/, voyage-02
    else:
        raise NotImplementedError