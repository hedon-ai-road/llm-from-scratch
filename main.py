
import re
import urllib.request
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleTokenizerV2:
    def __init__(self, vocab) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove the spaces before specific punctuation marks.
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def download_file():
    url = ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

def split_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    result = [item.strip() for item in result if item.strip()]
    return result

def create_vocab(words):
    vocab = {token:integer for integer,token in enumerate(words)}
    return vocab

def slide_window(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(raw_text)
    enc_sample = enc_text[50:]

    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        # print(context, "---->", desired)
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader

def test_dataloader():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)

def embedding():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # torch.nn.Embedding 本质上是一个可训练的查找表。
    # 初始化时，它是一个随机的、无意义的表，与训练数据无关。
    # 在具体的训练任务中（例如预测下一个词），通过海量数据的训练和反向传播算法，一点一滴学习和塑造出来的。
    # 前向传播 -> 计算损失 -> 反向传播 -> 更新权重
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs: \n", inputs)
    print("\nInputs shape: \n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)
    print("\nToken Embeddings Shape: \n", token_embeddings.shape)

    # 查找表只有 4 行，分别对应着句子中的 4 个位置：位置 0、位置 1、位置 2 和 位置 3。
    # 每一行是一个 output_dim (256) 维的向量。
    # torch.arange(context_length) 会创建一个整数序列的张量，即 torch.tensor([0, 1, 2, 3])。这些整数就是位置索引 (position indices)。
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print("\nPos Embeddings Shape: \n", pos_embeddings.shape)

    # 通过 token_embeddings + pos_embeddings，模型得到的每个词的最终输入向量，
    # 就同时包含了 “What” 和 “Where” 的信息，为后续的计算做好了充分的准备。
    input_embeddings = token_embeddings + pos_embeddings
    print("\nInput Embeddings Shape: \n", input_embeddings.shape)


def main():
    words = split_text("the-verdict.txt")
    all_words = sorted(set(words))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab = create_vocab(all_words)

    tokenizer = tiktoken.get_encoding("gpt2")
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(ids)
    print(tokenizer.decode(ids))

    ids = tokenizer.encode("Akwirw ier", allowed_special={"<|endoftext|>"})
    print(ids)
    print(tokenizer.decode(ids))

    print("===== sliding window ======")
    slide_window("the-verdict.txt")

    print("===== dataloader =====")
    test_dataloader()

    print("===== embedding =====")
    embedding()


if __name__ == "__main__":
    main()
