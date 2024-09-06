import os, sys
import json
from pathlib import Path
import pickle

from ragas.testset import TestsetGenerator
from ragas import RunConfig
from dotenv import load_dotenv,find_dotenv
import chromadb
from chromadb import PersistentClient
from pinecone import Pinecone as pinecone_client, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents import Document
import pandas as pd

from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness, context_recall
from datasets import Dataset

from renumics import spotlight
from renumics.spotlight import Embedding
import pandas as pd

from umap import UMAP
import numpy as np

# Import local packages
sys.path.append('../src/aerospace_chatbot')
import queries
import data_processing

# TODO after tests are written, write docstrings

# Set environment variables with .env
load_dotenv(find_dotenv(), override=True)

def write_dict_to_file(data_dict, filename):
    """write a dictionary as a json line to a file - allowing for appending"""
    with open(filename, "a") as f:
        f.write(json.dumps(data_dict) + "\n")

def read_dicts_from_file(filename):   
    # For this to work, each dict must be on a separate line!
    # Initialize an empty list to hold the JSON data
    json_data = []

    # Open the file and read line by line
    with open(filename, 'r') as file:
        for line in file:
            # Parse each line as JSON and append to the list
            json_data.append(json.loads(line))
    return json_data

def add_cached_columns_from_file(df, file_name, merge_on, columns,filter=None):
    """
    Read a file with cached list of dicts data and write it to a dataframe.
    filter is a dict with keys which are column headers and values which are the filters to apply to the columns.
    """
    if Path(file_name).exists():
        df_in=pd.DataFrame(list(read_dicts_from_file(file_name)))
        # Filter the cached file by filter if it exists.
        if filter is not None:  
            # filtered_data = df_in.copy()
            for column, value in filter.items():
                df_in = df_in[df_in[column] == value]

        cached_data = (
            df_in
            .drop_duplicates(
                subset=[merge_on],
            )[columns + [merge_on]]
            .dropna()
            .reset_index(drop=True)
        )
        return df.merge(
            cached_data,
            on=merge_on,
            how="left",
        ).replace({float('nan'): None}).reset_index(drop=True)
    else:
        # Create a copy of the DataFrame
        df_out = df.copy()

        # Add the new columns with the names specified in the 'columns' list
        if isinstance(columns, str):    # Handle where it's a single value, not an array of columns
            columns = [columns]
        for column in columns:
            df_out[column] = None

        # Reorder the columns to place the new columns at the end
        columns = list(df_out.columns)
        for column in columns:
            if column not in columns:
                columns.append(column)
        df_out = df_out[columns]

        return df_out
    
def lcdoc_export(index_type, index, query_model, export_pickle=False):
    if index_type=="ChromaDB":
        # Inspect the first db, save for synthetic test dataset
        all_docs = index.get(include=["metadatas", "documents", "embeddings"])
        lcdocs = [Document(page_content=doc, metadata=metadata) 
                for doc, metadata in zip(all_docs['documents'], all_docs['metadatas'])]
        
        # Format docs into dataframe
        all_docs = index.get(include=["metadatas", "documents", "embeddings"])
        df_docs = pd.DataFrame(
            {
                "id": [data_processing._stable_hash_meta(metadata) for metadata in all_docs["metadatas"]],
                "source": [metadata.get("source") for metadata in all_docs["metadatas"]],
                "page": [metadata.get("page", -1) for metadata in all_docs["metadatas"]],
                "document": all_docs["documents"],
                "embedding": all_docs["embeddings"],
            }
        )
        if export_pickle:
            df_temp=data_processing.archive_db('ChromaDB',index.name,query_model,export_pickle=True)
        
    elif index_type=="Pinecone":
        ids=[]
        for id in index.list():
            ids.extend(id)

        docs=[]
        df_docs = pd.DataFrame()
        chunk_size=200  # Tune to whatever doesn't error out, 200 won't for serverless
        for i in range(0, len(ids), chunk_size):
            print(f"Fetching {i} to {i+chunk_size}")
            vector=index.fetch(ids[i:i+chunk_size])['vectors']
            vector_data = []
            for key, value in vector.items():
                vector_data.append(value)
            docs.extend(vector_data)

            df_doc_temp = pd.DataFrame()
            df_doc_temp["id"]= [vector_elm["id"] for vector_elm in vector_data]
            df_doc_temp["source"]= [vector_elm["metadata"]["source"] for vector_elm in vector_data]
            df_doc_temp["page"]= [vector_elm["metadata"]["page"] for vector_elm in vector_data]
            df_doc_temp["document"]= [vector_elm["metadata"]["page_content"] for vector_elm in vector_data]
            df_doc_temp["embedding"]= [vector_elm["values"] for vector_elm in vector_data]
            df_docs = pd.concat([df_docs, df_doc_temp])

        lcdocs = []
        for data in docs:
            data=data['metadata']
            lcdocs.append(Document(page_content=data['page_content'],
                                metadata={'page':data['page'],'source':data['source']}))
        if export_pickle:
            df_temp=data_processing.archive_db('Pinecone',db['index_name'],query_model,export_pickle=True)
    
    return df_docs, lcdocs
        

def synthetic_dataset_loop(lcdocs,eval_size,n_questions,fname):
    """ 
    Check if testset.csv exists, use, or generate the synthetic dataset. If it doesn't exist, just loop through everything.
    If a partial export exists, it will loop through the remaining indices.
    If a few entries are missing it will loop through that full index.
    """
    if os.path.exists(fname):
        # Import testset.csv into a DataFrame
        df_testset = pd.read_csv(fname)

        full_index_list = list(range(0, len(lcdocs), eval_size))
        index_values = df_testset['index'].unique()
        index_loop = [] # Indices to index over

        for index_value in index_values:
            if len(df_testset[df_testset['index'] == index_value]) != n_questions:
                index_loop.append(index_value)

        index_loop += list(set([index for index in full_index_list if index not in index_values]))
        index_loop = sorted(index_loop)
    else:
        df_testset = pd.DataFrame()  # Initialize an empty DataFrame
        index_loop = list(range(0, len(lcdocs), eval_size))
    return df_testset, index_loop

def generate_testset(lcdocs,generator,eval_size,n_questions,fname,run_config):
    """Loop through eval_size chunks of the dataset to generate the testset. Export to csv along the way."""
    # Initialize the DataFrame to store the testset, determine what parts of the dataset need to be looped over
    df_testset, index_loop = synthetic_dataset_loop(lcdocs,eval_size,n_questions,fname)
    print(f"Index loop: {index_loop}")

    for i in index_loop:
        print(f"Processing index {i} to {i+eval_size}...")
        lcdocs_eval = lcdocs[i:i+eval_size]

        testset = generator.generate_with_langchain_docs(lcdocs_eval, 
                                                        test_size=n_questions,
                                                        with_debugging_logs=True,
                                                        is_async=False, # Avoid rate limit issues
                                                        run_config=run_config,
                                                        raise_exceptions=False)

        df_testset_new = testset.to_pandas()
        df_testset_new['index'] = i
        df_testset_new['eval_size'] = eval_size
        columns = ['index', 'eval_size'] + [col for col in df_testset_new.columns if col not in ['index', 'eval_size']]
        df_testset_new = df_testset_new[columns]

        # Export the testset to a csv file with the index
        if os.path.exists(fname):
            df_testset_new.to_csv(fname, mode='a', header=False, index=False)
        else:
            df_testset_new.to_csv(fname, index=False)
        df_testset = pd.concat([df_testset, df_testset_new])
    return df_testset

def rag_responses(index_type, index_name, query_model, llm, QA_model_params, df_qa, df_docs, testset_name):
    df_qa_out=df_qa.copy()

    # RAGatouille has a different structure than the other models
    if index_type == "RAGatouille":
        query_model_name=query_model.model.checkpoint
    else:
        query_model_name=query_model.model

    # Load cached version of rag responses, filter the responses by the model and parameters being evaluated
    df_qa_out = add_cached_columns_from_file(
        df_qa_out, 
        os.path.join('output',f'rag_response_cache_{testset_name}.jsonl'), "question", 
        ["answer", "source_documents", "answer_by", "query_model", "qa_model_params","index_type","index_name"],
        filter={"answer_by": llm.model_name,
                "query_model": query_model_name, 
                "qa_model_params": QA_model_params,
                "index_type": index_type,
                "index_name": index_name}
    )

    # Generate responses using RAG with input parameters
    for i, row in df_qa_out.iterrows():
        # Loop through each row
        if (row['answer_by'] != llm.model_name) or \
           (row['query_model'] != query_model_name) or \
           (row['qa_model_params'] != str(QA_model_params)):    # Check if the model and parameters are the same

            if row['answer'] is None or pd.isnull(row['answer']) or row['answer']=='':  # Check if the answer is empty
                print(f"Processing question {i+1}/{len(df_qa_out)}")


                # Use the QA model to query the documents
                qa_obj=queries.QA_Model(index_type,
                                index_name,
                                query_model,
                                llm,
                                **QA_model_params)
                qa_obj.query_docs(row['question'])
                response=qa_obj.result

                df_qa_out.loc[df_qa_out.index[i], "answer"] = response['answer'].content

                ids=[data_processing._stable_hash_meta(source_document.metadata)
                    for source_document in response['references']]
                df_qa_out.loc[df_qa_out.index[i], "source_documents"] = ', '.join(ids)
                # df_qa_out.loc[df_qa_out.index[i], "source_documents"] = ids

                df_qa_out.loc[df_qa_out.index[i], "answer_by"] = llm.model_name
                df_qa_out.loc[df_qa_out.index[i], "query_model"] = query_model_name
                df_qa_out.loc[df_qa_out.index[i], "qa_model_params"] = str(QA_model_params)
                df_qa_out.loc[df_qa_out.index[i], "index_type"] = index_type
                df_qa_out.loc[df_qa_out.index[i], "index_name"] = index_name

                # Save the response to cache file
                response_dict = {
                    "question_id": row['question_id'],
                    "question": row['question'],
                    "answer": response['answer'].content,
                    "source_documents": ids,
                    "answer_by": llm.model_name,
                    "query_model": query_model_name,
                    "qa_model_params": QA_model_params,
                    "index_type": index_type,
                    "index_name": index_name
                }
                write_dict_to_file(response_dict, os.path.join('output',f'rag_response_cache_{testset_name}.jsonl'))

    # Get the context documents content for each question
    source_documents_list = []
    for cell in df_qa_out['source_documents']:
        if isinstance(cell, str):
            cell_list = cell.split(', ')
        else:
            cell_list = cell
        context=[]
        for cell in cell_list:
            context.append(df_docs[df_docs["id"] == cell]["document"].values[0])
        source_documents_list.append(context)
    df_qa_out["contexts"]=source_documents_list

    # Addtionaly get embeddings for questions
    if index_type != "RAGatouille":
        if not Path(os.path.join('output',f'question_embeddings_{testset_name}.pickle')).exists():
            question_embeddings = [
                query_model.embed_query(question)
                for question in df_qa_out["question"]
        ]
        df_qa_out["embedding"] = question_embeddings
    else:
        # TODO add RAGatouille encodings
        df_qa_out["embedding"] = None

    return df_qa_out

def eval_rag(df_qa, eval_criterias, testset_name):
    df_qa_out=df_qa.copy()

    # Add answer correctness column, fill in if it exists
    df_qa_out = add_cached_columns_from_file(
        df_qa_out, 
        os.path.join('output',f'ragas_result_cache_{testset_name}.jsonl'), 
        "question", 
        eval_criterias
    )

    # Sometimes ground_truth does not provide a response. Just filter those out.
    df_qa_out = df_qa_out[df_qa_out['ground_truth'].apply(lambda x: isinstance(x, str))]

    # Evaluate the answer correctness if not already done
    fields = ["question", "answer", "contexts", "ground_truth"]

    for i, row in df_qa_out.iterrows():
        print(i, row["question"])
        response_dict={}
        response_dict["question"]=row["question"]

        if any(row[eval_criteria] is None for eval_criteria in eval_criterias):
            for eval_criteria in eval_criterias:
                print(eval_criteria)
                if eval_criteria=="answer_correctness":
                    eval_obj=answer_correctness
                elif eval_criteria=="faithfulness":
                    eval_obj=faithfulness
                elif eval_criteria=="context_recall":
                    eval_obj=context_recall

                if row[eval_criteria] is None or pd.isnull(row[eval_criteria]):
                    evaluation_result = evaluate(
                        Dataset.from_pandas(df_qa_out.iloc[i : i + 1][fields]),
                        [eval_obj],
                    )
                    df_qa_out.loc[i,eval_criteria] = evaluation_result[
                        eval_criteria
                    ]
                    
                response_dict[eval_criteria]=evaluation_result[eval_criteria]
            write_dict_to_file(response_dict, os.path.join('output',f'ragas_result_cache_{testset_name}.jsonl'))

    df_qa=df_qa_out

    return df_qa_out

def data_viz_prep(index_name,df_qa_eval,df_docs):
    """This section adds a column to df_documents containing the ids of the questions that used the document as source. """

    # add the infos about questions using each document to the documents dataframe
    # Explode 'source_documents' so each document ID is in its own row alongside the question ID
    df_questions_exploded = df_qa_eval.explode("source_documents")

    # Group by exploded 'source_documents' (document IDs) and aggregate
    agg = (
        df_questions_exploded.groupby("source_documents")
        .agg(
            num_questions=("id", "count"),  # Count of questions referencing the document
            question_ids=(
                "id",
                lambda x: list(x),
            ),  # List of question IDs referencing the document
        )
        .reset_index()
        .rename(columns={"source_documents": "id"})
    )

    # Merge the aggregated information back into df_documents
    df_documents_agg = pd.merge(df_docs, agg, on="id", how="left")

    # Use apply to replace NaN values with empty lists for 'question_ids'
    df_documents_agg["question_ids"] = df_documents_agg["question_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    # Replace NaN values in 'num_questions' with 0
    df_documents_agg["num_questions"] = df_documents_agg["num_questions"].fillna(0)

    # Concatenate the two dataframes
    df_visualize = pd.concat([df_qa_eval, df_documents_agg], axis=0)

    df_questions = df_visualize[~df_visualize["question"].isna()]
    umap = UMAP(n_neighbors=20, min_dist=0.15, metric="cosine", random_state=42).fit(
        df_questions["embedding"].values.tolist()
    )
    # umap_questions = umap.transform(df_visualize["embedding"].values.tolist())


    df_without_questions = df_visualize[df_visualize["question"].isna()]
    umap = UMAP(n_neighbors=20, min_dist=0.15, metric="cosine", random_state=42).fit(
        df_without_questions["embedding"].values.tolist()
    )
    umap_docs = umap.transform(df_visualize["embedding"].values.tolist())
    df_visualize["umap_docs"] = umap_docs.tolist()

    umap = UMAP(n_neighbors=20, min_dist=0.15, metric="cosine", random_state=42).fit(
        df_visualize["embedding"].values.tolist()
    )
    umap_all = umap.transform(df_visualize["embedding"].values.tolist())
    df_visualize["umap"] = umap_all.tolist()


    # find the nearet question (by embedding) for each document
    question_embeddings = np.array(df_visualize[df_visualize["question"].notna()]["embedding"].tolist())

    df_visualize["nearest_question_dist"] = [  # brute force, could be optimized using ChromaDB
        np.min([np.linalg.norm(np.array(doc_emb) - question_embeddings, axis=1)])
        for doc_emb in df_visualize["embedding"].values
    ]

    # write the dataframe to parquet for later use
    df_visualize.to_parquet(f'df_{index_name}.parquet')

    return df_visualize