import os
import jsonlines
from uuid import uuid4
import pandas as pd

from datasets import load_dataset
import subprocess
from tqdm.notebook import tqdm

# from dotenv import load_dotenv,find_dotenv
# load_dotenv(find_dotenv(),override=True)

# Load dataset
dataset_name = 'ai-aerospace/ams_data_train_generic_v0.1_100'
dataset=load_dataset(dataset_name)

# Write dataset files into data directory
data_directory = '../fine_tune_data/'

# Create the data directory if it doesn't exist
os.makedirs(data_directory, exist_ok=True)

# Write the train data to a CSV file
train_data='train_data.csv'
train_filename = os.path.join(data_directory, train_data)
dataset['train'].to_pandas().to_csv(train_filename, columns=['text'], index=False)

# Write the validation data to a CSV file
validation_data='validation_data.csv'
validation_filename = os.path.join(data_directory, validation_data)
dataset['validation'].to_pandas().to_csv(validation_filename, columns=['text'], index=False)

# Define project parameters
username='ai-aerospace'
project_name='./llms/'+'ams_data_train-100_'+str(uuid4())
repo_name='ams_data_train-100_'+str(uuid4())

model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1'
# model_name='mistralai/Mistral-7B-v0.1'

# Save parameters to environment variables
os.environ["project_name"] = project_name
os.environ["model_name"] = model_name
os.environ["repo_id"] = username+'/'+repo_name
os.environ["train_data"] = train_data   
os.environ["validation_data"] = validation_data

# Set .venv and execute the autotrain script
# !autotrain llm --train --project_name my-llm --model TinyLlama/TinyLlama-1.1B-Chat-v0.1 --data_path . --use-peft --use_int4 --learning_rate 2e-4 --train_batch_size 6 --num_train_epochs 3 --trainer sft
# The training dataset to be used must be called training.csv and be located in the data_path folder.
command="""
source ../.venv/bin/activate && autotrain llm --train \
    --project_name ${project_name} \
    --model ${model_name} \
    --data_path ../fine_tune_data \
    --train_split ${train_data} \
    --valid_split ${validation_data} \
    --use-peft \
    --learning_rate 2e-4 \
    --train_batch_size 6 \
    --num_train_epochs 3 \
    --trainer sft \
    --push_to_hub \
    --repo_id ${repo_id} \
    --token $HUGGINGFACE_TOKEN
"""

# Use subprocess.run() to execute the command
subprocess.run(command, shell=True, check=True, env=os.environ)