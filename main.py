import os
import argparse

import random
import numpy as np
import torch
import openai
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from torchmetrics.text.rouge import ROUGEScore
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from datasets import load_dataset
from tqdm.contrib import tzip

from huggingfacellm import get_huggingfacellm


def download_corpus(corpus_path):
    corpus_dataset = load_dataset("rag-datasets/mini-bioasq", 'text-corpus')
    df = corpus_dataset['passages'].to_pandas()
    df.to_csv(os.path.join(corpus_path, "storage.csv"))
    for passage in df['passage']:
        if isinstance(passage, str):
            pass
            # print(passage)

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--llm', type=str, default="chatgpt", help='llm model')
parser.add_argument('--download_corpus', action='store_true', help='download coupus')
parser.add_argument('--no_context', action='store_true', help='query directly without context')
parser.add_argument('--chunk_size', type=int, default=256, help='chunk size')
parser.add_argument('--source_dir', type=str, default="./corpus/", help='source directory')
parser.add_argument('--persist_dir', type=str, default="./bioasq_embedding", help='persist directory')
args = parser.parse_args()


openai.api_key = ""

llm_dict = {
    'chatgpt': "gpt-3.5-turbo", 
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "vicunav1.3": "lmsys/vicuna-7b-v1.3",
    "vicunav1.5": "lmsys/vicuna-7b-v1.5",
    "chatglm": "THUDM/chatglm-6b",
    "chatglm2": "THUDM/chatglm2-6b",
    "chatglm3": "THUDM/chatglm3-6b",
    "balle": "BelleGroup/BELLE-7B-2M",
    "qwen": "Qwen/Qwen1.5-7B-Chat",
}


seed_everything(42)
model_name = args.llm
if args.download_corpus:
    download_corpus(args.source_dir)
reader = SimpleDirectoryReader(args.source_dir)
documents = reader.load_data()
splitter = SentenceSplitter(
            chunk_overlap=20,
        )
# nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", embed_batch_size=16, cache_folder="./bge-large-en-v1.5")
Settings.llm = OpenAI(model=llm_dict[model_name]) if model_name == 'chatgpt' else get_huggingfacellm(llm_dict[model_name])
Settings.chunk_size = args.chunk_size
# service_context = ServiceContext.from_defaults(llm=llm, chunk_size=256)
if not os.path.exists("./bioasq_embedding"):
    vector_index = VectorStoreIndex.from_documents(
        documents, transformations=[splitter], show_progress=True
    )
    vector_index.storage_context.persist(persist_dir=args.persist_dir)
else:
    storage_context = StorageContext.from_defaults(persist_dir=args.persist_dir)
    vector_index = load_index_from_storage(storage_context)


query_engine = vector_index.as_query_engine(response_mode='generation') if args.no_context else vector_index.as_query_engine()
dataset = load_dataset("rag-datasets/mini-bioasq", 'question-answer-passages')
df_qa = dataset['test'].to_pandas()

df_qa = df_qa[['question', 'answer']]

rouge = ROUGEScore()
res_str_list = []
answer_list = []
for question, answer in tzip(df_qa["question"], df_qa["answer"]):
    res = query_engine.query(question)
    res_str_list.append(res.response)
    answer_list.append(answer)
    print(res.response)


metric_score = rouge(res_str_list, answer_list)

for key, score in metric_score.items():
    print(key + " : " + str(score.item()))

sementic_evaluator = SemanticSimilarityEvaluator()

sementic_score = 0.

for answer, response in tzip(df_qa["answer"], res_str_list):
    sementic_result = sementic_evaluator.evaluate(response=response, reference=answer)
    sementic_score += sementic_result.score

sementic_score /= len(res_str_list)

print(f"Sementic similarity score: {sementic_score}")




'''
res = query_engine.query('Is Hirschsprung disease a mendelian or a multifactorial disorder?	')
print(res)
'''
