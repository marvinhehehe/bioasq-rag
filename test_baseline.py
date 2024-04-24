import re
import os 
import random

import torch
import numpy as np
from llama_index.core import Settings
from torchmetrics.text.rouge import ROUGEScore
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from datasets import load_dataset


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def extract_answers(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # Pattern to capture the text between [ANSWER] and the next [QUESTION] or end of file
    pattern = re.compile(r'\[ANSWER\]\s*([\s\S]*?)(?=\[QUESTION\]|\Z)')

    # Find all matches in the file content
    answers = pattern.findall(content)
    
    return answers


seed_everything(42)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", embed_batch_size=16, cache_folder="./bge-large-en-v1.5")
# Example usage:
biomedlm_file = 'baselines/biomedlm.txt'
biomistral_file = 'baselines/biomistral.txt'

biomedlm_answers = extract_answers(biomedlm_file)
biomistral_answers = extract_answers(biomistral_file)

dataset = load_dataset("rag-datasets/mini-bioasq", 'question-answer-passages')
df_qa = dataset['test'].to_pandas()

df_qa = df_qa[['question', 'answer']]

rouge = ROUGEScore()
answer_list = []
for answer in df_qa["answer"]:
    answer_list.append(answer)

sementic_evaluator = SemanticSimilarityEvaluator()
print("biomedlm result:")
biomedlm_metric_score = rouge(biomedlm_answers, answer_list)

for key, score in biomedlm_metric_score.items():
    print(key + " : " + str(score.item()))

sementic_score = 0.

for answer, response in zip(df_qa["answer"], biomedlm_answers):
    sementic_result = sementic_evaluator.evaluate(response=response, reference=answer)
    sementic_score += sementic_result.score

sementic_score /= len(biomedlm_answers)

print(f"Sementic similarity score: {sementic_score}")

print("***********************************")
print("biomistral result:")
biomistral_metric_score = rouge(biomistral_answers, answer_list)

for key, score in biomistral_metric_score.items():
    print(key + " : " + str(score.item()))

sementic_score = 0.

for answer, response in zip(df_qa["answer"], biomistral_answers):
    sementic_result = sementic_evaluator.evaluate(response=response, reference=answer)
    sementic_score += sementic_result.score

sementic_score /= len(biomistral_answers)

print(f"Sementic similarity score: {sementic_score}")

