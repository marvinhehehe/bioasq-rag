from functools import partial

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from llama_index.llms.huggingface import HuggingFaceLLM

AUTH_TOKEN = ""

load_tokenizer = []


def llama_model_and_tokenizer(name, auth_token):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, token=auth_token, torch_dtype=torch.float16,
                                                 rope_scaling={"type": "dynamic", "factor": 2},
                                                 load_in_8bit=True, device_map="auto").eval()

    return tokenizer, model


def llama_completion_to_prompt(completion):
    return f"""<s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as 
        helpfully as possible, while being safe. Your answers should not include
        any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain 
        why instead of answering something not correct. If you don't know the answer 
        to a question, please don't share false information.

        Your goal is to provide answers relating to the financial performance of 
        the company.<</SYS>>
        {completion} [/INST]"""


def vicuna_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer =  AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True).eval()

    return tokenizer, model


def vicuna_completion_to_prompt(completion):
    system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    return f'''{system} 

        USER: {completion}
        ASSISTANT:
        '''

def belle_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer =  AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True).eval()

    return tokenizer, model


def belle_completion_to_prompt(completion):
    return "Human: " + completion + "\n\nAssistant:"


def chatglm_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModel.from_pretrained(name, trust_remote_code=True).half().cuda().eval()

    return tokenizer, model


def chatglm_completion_to_prompt(completion):
    return "[Round 0]\n问：" + completion + "\n答："


def chatglm2_completion_to_prompt(completion):
    return "[Round 1]\n\n问：" + completion +"\n\n答："


def chatglm3_completion_to_prompt(completion):
    return "<|user|>\n " + completion + "<|assistant|>"


def qwen_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name,
                                                 torch_dtype="auto",
                                                 device_map="auto").eval()

    return tokenizer, model


def qwen_completion_to_prompt(completion):
    tokenizer = load_tokenizer[0]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": completion}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


tokenizer_and_model_fn_dict = {
    "meta-llama/Llama-2-7b-chat-hf": partial(llama_model_and_tokenizer, auth_token=AUTH_TOKEN),
    "lmsys/vicuna-7b-v1.3": vicuna_model_and_tokenizer,
    "lmsys/vicuna-7b-v1.5": vicuna_model_and_tokenizer,
    "THUDM/chatglm-6b": chatglm_model_and_tokenizer,
    "THUDM/chatglm2-6b": chatglm_model_and_tokenizer,
    "THUDM/chatglm3-6b": chatglm_model_and_tokenizer,
    "BelleGroup/BELLE-7B-2M": belle_model_and_tokenizer,
    "Qwen/Qwen1.5-7B-Chat": qwen_model_and_tokenizer,
}

completion_to_prompt_dict = {
    "meta-llama/Llama-2-7b-chat-hf": llama_completion_to_prompt,
    "lmsys/vicuna-7b-v1.3": vicuna_completion_to_prompt,
    "lmsys/vicuna-7b-v1.5": vicuna_completion_to_prompt,
    "THUDM/chatglm-6b": chatglm_completion_to_prompt,
    "THUDM/chatglm2-6b": chatglm2_completion_to_prompt,
    "THUDM/chatglm3-6b": chatglm3_completion_to_prompt,
    "BelleGroup/BELLE-7B-2M": belle_completion_to_prompt,
    "Qwen/Qwen1.5-7B-Chat": qwen_completion_to_prompt,
}

llm_argument_dict = {
    "meta-llama/Llama-2-7b-chat-hf": {"context_window": 4096, "max_new_tokens": 64,
                          "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}},
    "lmsys/vicuna-7b-v1.3": {"context_window": 4096, "max_new_tokens": 64,
                          "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}},
    "lmsys/vicuna-7b-v1.5": {"context_window": 4096, "max_new_tokens": 64,
                          "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}},
    "THUDM/chatglm-6b": {"context_window": 4096, "max_new_tokens": 64,
                          "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}},
    "THUDM/chatglm2-6b": {"context_window": 4096, "max_new_tokens": 64,
                          "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}},
    "THUDM/chatglm3-6b": {"context_window": 4096, "max_new_tokens": 64,
                          "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95,
                                              "eos_token_id": [2, 64795, 64797]}},
    "BelleGroup/BELLE-7B-2M": {"context_window": 4096, "max_new_tokens": 64,
                          "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}},
    "Qwen/Qwen1.5-7B-Chat": {"context_window": 4096, "max_new_tokens": 64,
                          "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}},
}


def get_huggingfacellm(name):
    tokenizer, model = tokenizer_and_model_fn_dict[name](name)

    # Create a HF LLM using the llama index wrapper
    llm = HuggingFaceLLM(context_window=llm_argument_dict[name]["context_window"],
                         max_new_tokens=llm_argument_dict[name]["max_new_tokens"],
                         completion_to_prompt=completion_to_prompt_dict[name],
                         generate_kwargs=llm_argument_dict[name]["generate_kwargs"],
                         model=model,
                         tokenizer=tokenizer,
                         device_map="auto", )
    return llm