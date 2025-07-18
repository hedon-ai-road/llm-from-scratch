
import re
import urllib.request
import tiktoken


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


if __name__ == "__main__":
    main()
