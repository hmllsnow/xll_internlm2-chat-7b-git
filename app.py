import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from openxlab.model import download

base_path = './xll_internlm2-chat-7b'
os.system(f'git clone https://code.openxlab.org.cn/hmllsnow/xll_internlm2-chat-7b.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()
# model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True,torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True)).cuda()

def chat(message, history=None):
    if history is None or len(history) == 0:
        history.append(("你好", "你好, 我是潘金莲, 我将回答你提出来的问题"))
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="xll_InternLM2-Chat-7B",
                description="""
我是小莲莲，一个美丽勤劳的女子.  
                 """,
                 ).queue(1).launch(show_api=True , quiet=True)
