import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm


tokenizer = AutoTokenizer.from_pretrained("facebook/esm-1b")

model = AutoModelForMaskedLM.from_pretrained("facebook/esm-1b")
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)
sequences_Example = ["A E T C Z A O","S K T Z P"]
sequences_Example = [re.sub(r"[UZOBX]", "<unk>", sequence) for sequence in sequences_Example]
embedding = fe(sequences_Example)
embedding = np.array(embedding)
print(embedding)
features = [] 
for seq_num in range(len(embedding)):
    seq_len = len(sequences_Example[seq_num].replace(" ", ""))
    padded_seq_len = len(embedding[seq_num])
    start_Idx = padded_seq_len-seq_len-2
    end_Idx = padded_seq_len-2
    seq_emd = embedding[seq_num][start_Idx:end_Idx]
    features.append(seq_emd)
print(features)