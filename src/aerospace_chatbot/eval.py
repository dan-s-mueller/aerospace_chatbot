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
from ragas.metrics import answer_correctness
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

# Set environment variables with .env
load_dotenv(find_dotenv(), override=True)

def write_dict_to_file(data_dict, filename):
    """write a dictionary as a json line to a file - allowing for appending"""
    with open(filename, "a") as f:
        f.write(json.dumps(data_dict) + "\n")

def read_dicts_from_file(filename):
    """Read a json line file as a generator of dictionaries - allowing to load multiple dictionaries as list."""
    with open(filename, "r") as f:
        for line in f:
            yield json.loads(line)

def add_cached_column_from_file(df, file_name, merge_on, column):
    """Read a file with cached list of dicts data write it to a dataframe."""
    if Path(file_name).exists():
        cached_answer_correctness = (
            pd.DataFrame(list(read_dicts_from_file(file_name)))
            .drop_duplicates(
                subset=[merge_on],
            )[[column, merge_on]]
            .dropna()
            .reset_index(drop=True)
        )
        return df.merge(
            cached_answer_correctness,
            on=merge_on,
            how="left",
        ).reset_index(drop=True)
    else:
        # Create a copy of the DataFrame
        df_out = df.copy()

        # Add the new column with the name of the variable 'column'
        df_out[column] = None

        # Reorder the columns to place the new column at the end
        columns = list(df_out.columns)
        columns.remove(column)
        columns.append(column)
        df_out = df_out[columns]
        
        return df_out
    
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

def rag_responses(index_type, index_name, query_model, llm, QA_model_params, df_qa, df_docs):
    # Generate responses using RAG with input parameters
    for i, row in df_qa.iterrows():
        if row['answer'] is None or pd.isnull(row['answer']) or row['answer']=='':
            print(f"Processing question {i+1}/{len(df_qa)}")

            # Use the QA model to query the documents
            qa_obj=queries.QA_Model(index_type,
                            index_name,
                            query_model,
                            llm,
                            **QA_model_params)
            qa_obj.query_docs(row['question'])
            response=qa_obj.result

            df_qa.loc[df_qa.index[i], "answer"] = response['answer'].content

            ids=[data_processing._stable_hash_meta(source_document.metadata)
                for source_document in response['references']]
            df_qa.loc[df_qa.index[i], "source_documents"] = ', '.join(ids)

            # Save the response to cache file
            response_dict = {
                "question": row['question'],
                "answer": response['answer'].content,
                "source_documents": ids,
            }
            write_dict_to_file(response_dict, os.path.join('output',f'rag_response_cache_{index_name}.json'))

    # Get the context documents content for each question
    source_documents_list = []
    for cell in df_qa['source_documents']:
        cell_list = cell.strip('[]').split(', ')
        context=[]
        for cell in cell_list:
            context.append(df_docs[df_docs["id"] == cell]["document"].values[0])
        source_documents_list.append(context)
    df_qa["contexts"]=source_documents_list

    # Addtionaly get embeddings for questions
    if not Path(os.path.join('output',f'question_embeddings_{index_name}.pickle')).exists():
        question_embeddings = [
            query_model.embed_query(question)
            for question in df_qa["question"]
        ]
        with open(os.path.join('output',f'question_embeddings_{index_name}.pickle'), "wb") as f:
            pickle.dump(question_embeddings, f)

    question_embeddings = pickle.load(open(os.path.join('output',f'question_embeddings_{index_name}.pickle'), "rb"))
    df_qa["embedding"] = question_embeddings
    return df_qa

def eval_rag(index_name, df_qa):
    # Add answer correctness column, fill in if it exists
    df_qa = add_cached_column_from_file(
        df_qa, os.path.join('output',f'ragas_result_cache_{index_name}.json', "question"), "answer_correctness"
    )

    # Sometimes ground_truth does not provide a response. Just filter those out.
    df_qa = df_qa[df_qa['ground_truth'].apply(lambda x: isinstance(x, str))]
    df_qa

    # Prepare the dataframe for evaluation
    df_qa_eval = df_qa.copy()

    # Evaluate the answer correctness if not already done
    fields = ["question", "answer", "contexts", "ground_truth"]
    for i, row in df_qa_eval.iterrows():
        print(i, row["question"])
        # TODO add multiple eval criteria
        if row["answer_correctness"] is None or pd.isnull(row["answer_correctness"]):
            evaluation_result = evaluate(
                Dataset.from_pandas(df_qa_eval.iloc[i : i + 1][fields]),
                [answer_correctness],
            )
            df_qa_eval.loc[i, "answer_correctness"] = evaluation_result[
                "answer_correctness"
            ]

            # optionally save the response to cache
            response_dict = {
                "question": row["question"],
                "answer_correctness": evaluation_result["answer_correctness"],
            }
            write_dict_to_file(response_dict, os.path.join('output',f'ragas_result_cache_{index_name}.json'))

    # write the answer correctness to the original dataframe
    df_qa["answer_correctness"] = df_qa_eval["answer_correctness"]

    return df_qa_eval, df_qa

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
    umap_questions = umap.transform(df_visualize["embedding"].values.tolist())


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