import os
import random
import numpy as np
import torch

import tiktoken

def load_tokens(filename: str):
    """
    load a shard from the given path and convert to pytorch tensor
    file_path: the path to the shard file (.npy file)
    return: a pytorch tensor
    """
    # expects .npy file, used inside dataloader
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DistributedDataLoader:

    def __init__(self, B, T, process_rank, num_processes, split, data_root, shuffle=True, master_process=True):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        self.shuffle = shuffle
        # get the shard filename
        data_root = data_root
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split: {split}"
        if self.shuffle:
            random.shuffle(self.shards)
        self.master_process = master_process
        if self.master_process:
            print(f"with data root: {data_root} found: {len(shards)} shards for: {split} split and num processes: {self.num_processes}")
        # state management
        self.reset()

    def reset(self):
        # useful in val_loader.reset()
        # state, initialize at shard 0 or first file
        if self.shuffle:
            random.shuffle(self.shards)
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank


    def next_batch(self):
        # prepare inputs and targets for a single *step* of the optimization
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T) # remove last token
        y = buf[1:].view(B, T)  # remove first token
        # advance to the next chunk of the array
        self.current_position += (B * T * self.num_processes)
        # check for next batch loading for all processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


class FineWebEduDataLoader(DistributedDataLoader):
    # initialize a single gpu dataloader as default
    def __init__(self, B, T, process_rank=0, num_processes=1, split='train', data_root='/weka/bethge/mwe102/brendel/build-gpt2/edu_fineweb10B', shuffle=True, master_process=True):
        super().__init__(B, T, process_rank, num_processes, split, data_root, shuffle, master_process)
    

if __name__ == "__main__":
    B = 32
    T = 1024

    # set the seed
    random.seed(42) # to make sure the dataloader is deterministic

    # initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # initialize the dataloader
    train_loader = FineWebEduDataLoader(B=B, T=T, process_rank=0, num_processes=1, split='train')
    val_loader = FineWebEduDataLoader(B=B, T=T, process_rank=0, num_processes=1, split='val')

    # test the dataloader
    print("-"*100)
    print("Testing train loader")
    print("-"*100)
    for i in range(2):
        x, y = train_loader.next_batch()
        print(x.shape, y.shape)
        # print the first 10 tokens of the first batch
        print(f"x: {tokenizer.decode(x[0, :100].tolist())}")
        print(f"y: {tokenizer.decode(y[0, :100].tolist())}")

    # test the val loader
    print("-"*100)
    print("Testing val loader")
    print("-"*100)
    for i in range(2):
        x, y = val_loader.next_batch()
        print(x.shape, y.shape)
        print(f"x: {tokenizer.decode(x[0, :100].tolist())}")
        print(f"y: {tokenizer.decode(y[0, :100].tolist())}")