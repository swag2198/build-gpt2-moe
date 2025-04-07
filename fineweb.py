import os
import multiprocessing as mp
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

"""
    Fineweb-edu is a filtered dataset and the llama 372B model was
    the judge of which content is educational, so this is an already
    llm informed dataset.

    1. download the data
    2. preprocess and tokenize all the data
    3. save data shards to local_dir to load during training
"""

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT" # 10B tokens
shard_size = int(1e8) # 1e8 = 100 * 10**6 = 100M tokens in one shard


# create the local directory to store the data
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), # directory of this file
                              local_dir)                 # data shards folder name
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# print(f"__file__={__file__}\n{os.path.dirname(__file__)=}")

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
"""
>>> fw
Dataset({
    features: ['text', 'id', 'dump', 'url', 'file_path', 'language', 'language_score', 'token_count', 'score', 'int_score'],
    num_rows: 9672101
})
"""

# initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # 50256, need to append it at the start

def tokenize(doc: dict):
    """tokenizes a single document
    """
    assert 'text' in doc, "The key 'text' must be present in passed document!"
    tokens = [eot] # start with the `eot` token
    # tokens.extend(enc.encode(doc["text"])) # bug: with disallowed tokens or something!
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary size check"
    tokens_np_uint16 = tokens_np.astype(np.uint16) # tokens are now 16 bit unsigned int
    return tokens_np_uint16 # uint16 to save a little bit of space


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()-2) # in galvani a100: 64 cpus!
# print(f"cpus: {os.cpu_count()}, nprocs: {nprocs}")
with mp.Pool(nprocs) as pool:
    
    shard_index = 0
    all_tokens_np = np.empty((shard_size, ), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):

        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)

            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
