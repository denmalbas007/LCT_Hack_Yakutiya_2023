import sys

from models import Chat, Retriever, EmbeddingBuilder, Summarizer

sys.path.insert(0, "/home/leffff/PycharmProjects/LCT_Hack_Yakutiya_2023/venv/lib/python3.10/site-packages")

import os
import random
from tqdm.auto import tqdm
tqdm.pandas()

import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig


def seed_everything(seed: int,
                    use_deterministic_algos: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(use_deterministic_algos)
    random.seed(seed)


if __name__ == "__main__":
    RANDOM_STATE = 42
    DEVICE = "cuda"

    seed_everything(RANDOM_STATE)

    summarizer_model_name = "csebuetnlp/mT5_multilingual_XLSum"
    summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)

    embedding_model_name = "ai-forever/sbert_large_nlu_ru"
    embedding_model = AutoModel.from_pretrained(embedding_model_name)
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

    generation_model_name = "Den4ikAI/FRED-T5-LARGE_text_qa"
    generation_config = GenerationConfig.from_pretrained(generation_model_name)
    generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
    generation_model = AutoModelForSeq2SeqLM.from_pretrained(generation_model_name)

    summarizer = Summarizer(
        model=summarizer_model,
        tokenizer=summarizer_tokenizer
    )

    builder = EmbeddingBuilder(
        model=embedding_model,
        tokenizer=embedding_tokenizer,
        summarizer=summarizer,
        csv_path="../data/parsed.csv",
        embeddings_save_path="embeddings.npy",
        kdtree_save_path="kdtree.joblib",
        batch_size=1,
        device="cuda"
    )

    retriever = Retriever(
        model=embedding_model,
        tokenizer=embedding_tokenizer,
        k=5,
        csv_path="../data/parsed.csv",
        kdtree_load_path="kdtree.joblib"
    )

    chat = Chat(
        model=generation_model,
        tokenizer=generation_tokenizer,
        generation_config=generation_config,
        retriever=retriever
    )

    print(chat.answer("Планируется ли открытие новых спорт комплексов?"))
