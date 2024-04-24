import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import MistralForCausalLM, GPT2LMHeadModel, AutoTokenizer, GPT2Tokenizer

# Device
device = torch.device("cuda")

# Dataset
text_corpus = load_dataset("rag-datasets/mini-bioasq", 'text-corpus')
passages = text_corpus['passages'].to_pandas()
qa_passages = load_dataset("rag-datasets/mini-bioasq", 'question-answer-passages')
qa = qa_passages['test'].to_pandas()
question = qa["question"]
answer = qa["answer"]
print(question[0])
print("\n")
print(answer[0])
print(100 * "-")

# BioMistral (Question Answering)
tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
model = MistralForCausalLM.from_pretrained("BioMistral/BioMistral-7B").to(device)
output_list = []
for q in tqdm(question, total=len(question)):
    input_ids = tokenizer.encode(q, return_tensors="pt").to(device)
    sample_output = model.generate(input_ids, do_sample=True, max_new_tokens =64, temperature= 0.7, top_k= 50, top_p= 0.95, no_repeat_ngram_size=2)
    output = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    output_q = output[:len(q)]
    output_a = output[len(q):]
    output_list.append(f"[QUESTION]\n{output_q}\n[ANSWER]\n{output_a}\n")
with open('biomistral.txt', 'w', encoding='utf-8') as file:
    for output in tqdm(output_list):
        file.write(output + '\n')

# BioMedLM (Text Completion)
tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM")
model = GPT2LMHeadModel.from_pretrained("stanford-crfm/BioMedLM").to(device)
output_list = []
for q in tqdm(question, total=len(question)):
    input_ids = tokenizer.encode(q, return_tensors="pt").to(device)
    sample_output = model.generate(input_ids, do_sample=True, max_new_tokens =64, temperature= 0.7, top_k= 50, top_p= 0.95, no_repeat_ngram_size=2)
    output = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    output_q = output[:len(q)]
    output_a = output[len(q):]
    output_list.append(f"[QUESTION]\n{output_q}\n[ANSWER]\n{output_a}\n")
with open('biomedlm.txt', 'w', encoding='utf-8') as file:
    for output in tqdm(output_list):
        file.write(output + '\n')
