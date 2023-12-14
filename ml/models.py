from joblib import dump, load
from tqdm.auto import tqdm

tqdm.pandas()
import re

import pandas as pd
import torch
from datasets import Dataset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class Summarizer:
    def __init__(
            self, model: AutoModel, tokenizer: AutoTokenizer, device: str = "cuda"
    ):
        super().__init__()
        self.device = device

        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.model.to(self.device)

        self.WHITESPACE_HANDLER = lambda k: re.sub(
            "\s+", " ", re.sub("\n+", " ", k.strip())
        )

    def summarize(self, text):
        input_ids = self.tokenizer(
            [self.WHITESPACE_HANDLER(text)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )["input_ids"].to(self.device)

        output_ids = self.model.generate(
            input_ids=input_ids, max_length=256, no_repeat_ngram_size=2, num_beams=4
        )[0]

        summary = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return summary


class EmbeddingBuilder:
    def __init__(self,
                 model: AutoModel,
                 tokenizer: AutoTokenizer,
                 summarizer,
                 csv_path: str,
                 embeddings_save_path: str = "embeddings.npy",
                 kdtree_save_path: str = "kdtree.joblib",
                 batch_size: int = 1,
                 device: str = "cuda"):
        super().__init__()
        self.device = device

        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.model.to(self.device)

        self.summarizer = summarizer

        self.embeddings_save_path = embeddings_save_path
        self.kdtree_save_path = kdtree_save_path

        self.batch_size = batch_size

        self._load_dataset(csv_path)
        self._process_dataset()
        self.extract_embeddings()

    def _load_dataset(self, csv_path):
        df = pd.read_csv(csv_path)
        df["short_text"] = df["text"].progress_apply(self.summarizer.summarize)
        self.dataset = Dataset.from_pandas(df)

    def _process_dataset(self):
        self.dataset = self.dataset.map(
            lambda sample: self._preprocess_text(sample['short_text'])
        )

        self.dataset = self.dataset.remove_columns([
            'short_text'
        ])

        self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    def _preprocess_text(self, text):
        # print(text)
        out = self.tokenizer.encode_plus(text, max_length=512, truncation=True, padding="max_length")
        return out

    def extract_embeddings(self) -> None:
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size)

        embeddings = []

        for batch in tqdm(dataloader):
            input_ids, attention_masks = batch["input_ids"], batch["attention_mask"]
            input_ids, attention_masks = input_ids.to(self.device), attention_masks.to(self.device)

            with torch.no_grad():
                output = mean_pooling(self.model(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                ), attention_masks)

            embeddings.append(output.cpu())

        embeddings = torch.cat(embeddings, dim=0).numpy()

        kdtree = NearestNeighbors(n_neighbors=5,
                                  metric='cosine',
                                  algorithm='brute',
                                  n_jobs=-1)
        kdtree.fit(embeddings)

        dump(kdtree, self.kdtree_save_path)


class Retriever:
    def __init__(
            self,
            model: AutoModel,
            tokenizer: AutoTokenizer,
            k: int,
            csv_path: str,
            kdtree_load_path: str,
            device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        self.model = model
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = tokenizer

        self.df = pd.read_csv(csv_path)

        self.k = k

        self.kdtree = self._load_indexer(kdtree_load_path)

    @staticmethod
    def _load_indexer(path):
        kdtree = load(path)
        return kdtree

    def _preprocess_text(self, text):
        out = self.tokenizer.encode_plus(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return out

    def _get_embedding(self, sample):
        input_ids = sample["input_ids"]
        if len(input_ids.shape) < 2:
            input_ids.unsqueeze(dim=0)

        attention_mask = sample["attention_mask"]
        if len(attention_mask.shape) < 2:
            input_ids.unsqueeze(dim=0)

        with torch.no_grad():
            return (
                mean_pooling(
                    self.model(
                        input_ids.to(self.device), attention_mask.to(self.device)
                    ),
                    attention_mask.to(self.device),
                )
                .cpu()
                .numpy()
            )

    def search(self, text):
        sample = self._preprocess_text(text)
        embedding = self._get_embedding(sample)

        ind = self.kdtree.kneighbors(embedding, n_neighbors=5, return_distance=False)
        return self.df.iloc[ind[0]]


class Chat:
    def __init__(self, model, tokenizer, generation_config, retriever, k: int = 3, device: str = "cuda"):
        self.device = device

        self.model = model
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = tokenizer

        self.generation_config = generation_config
        self.retriever = retriever
        self.k = k

    @staticmethod
    def _remove_punct(text):
        text = "".join([char for char in text if char.isalpha() or char == " "])
        return text

    @staticmethod
    def _form_prompt(text, retrieved):
        texts = retrieved["text"].tolist()

        prompt = f"<SC6>Текст: {texts[0] + ' Ответь развернуто'}\nВопрос: {text}\nОтвет: <extra_id_0>"

        return prompt

    def generate(self, prompt):
        data = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **data,
            generation_config=self.generation_config,
        )[0]
        # print(self.tokenizer.decode(data["input_ids"][0].tolist()))
        out = self.tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)
        return out

    def answer(self, message):
        retrieved_samples = self.retriever.search(message)
        prompt = self._form_prompt(message, retrieved_samples)

        return (
            self.generate(prompt)[13:],
            f"На основе документа {retrieved_samples['url'].values[0]}",
        )
