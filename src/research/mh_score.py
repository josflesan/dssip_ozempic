import sys
import fasttext
import numpy as np
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons

from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")
text_processor = TextPreProcessor(
    normalize=["url", "email", "percent", "money", "phone", "user",
               "time", "url", "date", "number"],
    fix_html=True,
    corrector="english",
    segmenter="twitter",
    dicts=[emoticons]
)

# def compute_mh_score_sbert(post: str):
#     # Compute the sentence embedding
#     sentence_embed = model.encode(post)

#     # Get mental-health vector
#     mental_health_lexicon = [
#         "depressed", "depression", "hopeless", "anxious", "anxiety", "worthless", "tired", "alone", "suicidal",
#         "overwhelmed", "ugly", "panic", "fatigue", "fat", "miserable", "numb", "sad", "crying", "cry", "vomit",
#         "vomiting", "jealous", "angry", "mad", "disgusting", "revolting", "upset", "bloated", "disgust", "disgusted",        
#     ]
#     mh_embedding = model.encode(" ".join(mental_health_lexicon))

#     # Compute cosine similarity between post vector and MH vector
#     cosine_sim = np.dot(sentence_embed, mh_embedding) / (np.linalg.norm(sentence_embed) * np.linalg.norm(mh_embedding) + 1e-10)

#     return cosine_sim

if __name__ == "__main__":
    text = sys.argv[1]
    # print(compute_mh_score_sbert(text))

    print(text_processor.pre_process_doc(text))
