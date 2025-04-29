"""Embed the mental health data from mental-health related subreddits."""

from tqdm import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons

from utils import flatten_df

MH_DATAFRAMES = [
    "data/reddit/mental_health/depression_topall_topalltime.parquet",
    "data/reddit/mental_health/EatingDisorders_topall_topalltime.parquet",
    "data/reddit/mental_health/EDAnonymous_topall_topalltime.parquet",
]

POP_DATAFRAMES = [
    "data/reddit/pop_culture/Instagramreality_topall_topalltime.parquet",
    "data/reddit/pop_culture/celebritygossip_topall_topalltime.parquet",
    "data/reddit/pop_culture/fauxmoi_topall_topalltime.parquet",
    "data/reddit/pop_culture/popculturechat_topall_topalltime.parquet"
]

text_processor = TextPreProcessor(
    normalize=["url", "email", "percent", "money", "phone", "user",
               "time", "url", "date", "number"],
    fix_html=True,
    corrector="english",
    segmenter="twitter",
    dicts=[emoticons]
)
model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_pop_embeddings():
    documents = []
    embeddings = []
    batch_size = 32

    # Collect all of the documents from the dataframes by flattening and casting column as list
    for dataframe in POP_DATAFRAMES:
        df = pd.read_parquet(dataframe)
        df = flatten_df(df)
        df["content"].apply(lambda x: text_processor.pre_process_doc(x))
        documents += df["content"].tolist()

    for i in tqdm(range(0, len(documents), batch_size), desc="Encoding Documents..."):
        batch = documents[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    # Concatenate all embeddings into one array
    all_embeddings = np.vstack(embeddings)
    average_embedding = np.mean(all_embeddings, axis=0)

    # Save final average embedding
    np.save("src/embeddings/pop_embeddings.npy", all_embeddings)
    np.save("src/embeddings/pop_avg_embedding.npy", average_embedding)

def compute_mh_embeddings():
    documents = []
    embeddings = []
    batch_size = 32

    # Collect all of the documents from the dataframes by flattening and casting column as list
    for dataframe in MH_DATAFRAMES:
        df = pd.read_parquet(dataframe)
        df = flatten_df(df)
        df["content"].apply(lambda x: text_processor.pre_process_doc(x))
        documents += df["content"].tolist()

    for i in tqdm(range(0, len(documents), batch_size), desc="Encoding Documents..."):
        batch = documents[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    # Concatenate all embeddings into one array
    all_embeddings = np.vstack(embeddings)
    average_embedding = np.mean(all_embeddings, axis=0)

    # Save final average embedding
    np.save("src/embeddings/mh_embeddings.npy", all_embeddings)
    np.save("src/embeddings/mh_avg_embedding.npy", average_embedding)

if __name__ == "__main__":
    compute_mh_embeddings()


