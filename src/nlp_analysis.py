"""Contains the logic to extend the original dataframes of collected posts with NLP analysis"""

import argparse
import logging
import os
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import spacy
from bertopic import BERTopic
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from swifter import set_defaults
from tqdm.auto import tqdm
from transformers import pipeline

from utils import convertLabelToSentiment, flatten_df

####################### LOGGING ###########################

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

###########################################################


####################### SETUP ###########################

tqdm.pandas()
nltk.download("stopwords")
stop_words = stopwords.words("english")
vectorizer_model = CountVectorizer(
    stop_words=stop_words,
    ngram_range=(1, 3)
)
topicModel = BERTopic(vectorizer_model=vectorizer_model)
embeddingModel = SentenceTransformer("all-MiniLM-L6-v2")

text_processor = TextPreProcessor(
    normalize=["url", "email", "percent", "money", "phone", "user",
               "time", "url", "date", "number"],
    fix_html=True,
    corrector="english",
    segmenter="twitter",
    dicts=[emoticons]
)

nlp = spacy.load("en_core_web_sm")
sentimentPipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
emotionPipeline = pipeline("text-classification", 
                            model="j-hartmann/emotion-english-distilroberta-base", 
                            return_all_scores=True)

#########################################################


############### UTILITIES ##################

def getArgmaxEmotion(text: str):
    return max(emotionPipeline(text[:512])[0], key=lambda x: x["score"])

############################################

####################### ANALYSIS ###########################

def topic_modelling(df: pd.DataFrame, out_filename: str, subreddit: str):
    # Compute the topics and probabilities from the documents
    logger.info("### PREPROCESSING DOCUMENTS ###")
    documents = df["content"].progress_apply(lambda x: text_processor.pre_process_doc(x[:512]))

    logger.info("### FITTING TOPIC MODEL ###")
    topics, probs = topicModel.fit_transform(documents)

    # Create new column for topics
    df["topics"] = topics
    df["topic_probs"] = probs

    # Write topic info into output files
    topic_info = topicModel.get_topic_info()
    mh_topics = topic_info[topic_info["Name"].str.contains("anxious|depress|panic|binge|fast", case=False)]
    
    topic_info.to_csv(f"logs/{subreddit}/" + out_filename + "_TOPICS.csv")
    mh_topics.to_csv(f"logs/{subreddit}/" + out_filename + "_MHTOPICS.csv")

    # Create visualizations and save them to results
    logger.info("### PLOTTING TOPIC MODELLING RESULTS ###")
    logger.info("1. PLOTTING TOPIC EMBEDDINGS")
    print(f"TOPIC EMBEDDINGS SHAPE: {topicModel.topic_embeddings_.shape}")
    if topicModel.topic_embeddings_.shape[0] >= 5:
        topicEmbeddings = topicModel.visualize_topics()  # Visualize the topics using embeddings
        topicEmbeddings.write_image(f"plots/{subreddit}/{out_filename}_TOPIC_EMBED.png", scale=2)
        topicEmbeddings.write_html(f"plots/{subreddit}/{out_filename}_TOPIC_EMBED.html")
    else:
        logger.warning("TOO FEW TOPICS TO VISUALIZE -- SKIPPING")

    logger.info("2. PLOTTING TOPIC HIERARCHY")
    topicsHierarchy = topicModel.visualize_hierarchy()
    topicsHierarchy.write_image(f"plots/{subreddit}/{out_filename}_TOPIC_HIER.png", scale=2)

    logger.info("3. PLOTTING TOPIC SCORES")
    topicsScores = topicModel.visualize_barchart()
    topicsScores.write_image(f"plots/{subreddit}/{out_filename}_TOPIC_SCORE.png", scale=3)

    logger.info("4. PLOTTING DOCUMENT EMBEDDINGS")
    documentEmbeddings = topicModel.visualize_documents(documents, topics=topicModel.topics_[:100], embeddings=None)
    # documentEmbeddings.write_image(f"plots/{out_filename}_TOPIC_DOCEMBED.png", scale=2)
    documentEmbeddings.write_html(f"plots/{subreddit}/{out_filename}_TOPIC_DOCEMBED.html")

    # Write representative documents for topics
    logger.info("### WRITING REPRESENTATIVE DOCUMENTS FOR TOP 10 TOPICS ###")
    with open(f"logs/{subreddit}/{out_filename}_REPRESENTATIVE_TOPICS.txt", "w") as f:
        for topic in range(0, 11):
            representative = topicModel.get_representative_docs(topic)

            # Break if no topics left
            if representative is None:
                break

            f.write(f"------------------- TOPIC {topic} -------------------\n\n")
            
            for i, doc in enumerate(representative[:5]):
                f.write(f"\n--- Document {i + 1} ---\n{doc}")
            
            f.write("\n\n")

def sentiment_analysis(df: pd.DataFrame):
    # Compute the result of the sentiment model
    logger.info("### COMPUTING SENTIMENT ###")
    df["sentiment_result"] = df["content"].progress_apply(lambda x: sentimentPipeline(text_processor.pre_process_doc(x[:512]))[0])

    # Extract into label and score and drop original sentiment_result column
    df[["sentiment", "sentiment_score"]] = df["sentiment_result"].apply(
        lambda x: pd.Series([convertLabelToSentiment(x["label"]), x["score"]])
    )
    df.drop(columns=["sentiment_result"], inplace=True)

    # Compute the result of the emotion model
    logger.info("### COMPUTING EMOTION ###")
    df["emotion_result"] = df["content"].progress_apply(getArgmaxEmotion)

    # Extract into label and score and drop original emotion_result column
    df[["emotion", "emotion_score"]] = df["emotion_result"].apply(
        lambda x: pd.Series([x["label"], x["score"]])
    )
    df.drop(columns=["emotion_result"], inplace=True)

def mental_health_score(df: pd.DataFrame):
    mh_embedding = np.load("src/embeddings/mh_avg_embedding.npy")

    def compute_mental_health_score(post: str):
        # Compute post embedding
        doc = text_processor.pre_process_doc(post)
        curr_embedding = embeddingModel.encode(doc, show_progress_bar=False)

        # Compute cosine similarity between post vector and MH vector
        cosine_sim = np.dot(curr_embedding, mh_embedding) / (np.linalg.norm(curr_embedding) * np.linalg.norm(mh_embedding) + 1e-10)

        return cosine_sim

    # Compute mental health column
    df["mh_score"] = df["content"].progress_apply(compute_mental_health_score)

def pronoun_analysis(df: pd.DataFrame):
    """
    Determines the person used in the text from pronoun analysis. Populates new column
    in the dataframe with 1 (first-person), 2 (third-person) or 0 (mixed/undefined)

    Args:
        df (pd.DataFrame): analysis dataframe
    """

    # Determine person from pronoun analysis
    def determine_pronoun(text: str):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
        
        doc = nlp(text)
        pronoun_counts = Counter()

        # Pronoun sets
        first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
        third_person = {"he", "him", "his", "she", "her", "they", "them", "their", "theirs", "it", "its"}
        person_threshold_ratio = 1

        # Count different first and third person pronouns
        for token in doc:
            if token.pos_ == "PRON":
                pronoun = token.text.lower()
                if pronoun in first_person:
                    pronoun_counts["first"] += 1
                elif pronoun in third_person:
                    pronoun_counts["third"] += 1

        if pronoun_counts["first"] > pronoun_counts["third"] * person_threshold_ratio:
            return 1
        elif pronoun_counts["third"] > pronoun_counts["first"] * person_threshold_ratio:
            return 2
        else:
            return 0
        
    df["person"] = df["content"].progress_apply(determine_pronoun)

def discourse_analysis(df: pd.DataFrame):
    """Analyze the structure of claims and evidence"""

    # Function to compute discourse markers
    def determine_discourse(text: str):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {}
    
        doc = nlp(text)

        # Identify claim and evidence structures
        claim_markers = ["i think", "believe", "claim", "argue", "suggest"]
        evidence_markers = ["because", "since", "research", "shows", "study", "evidence"]
        lemmas = [token.lemma_.lower() for token in doc]

        # Count different types of discourse elements
        totalClaims = sum(1 for lemma in lemmas if lemma in claim_markers)
        totalEvidence = sum(1 for lemma in lemmas if lemma in evidence_markers)
        evidenceRatio = 0

        # Calculate the ratio of claims to evidence
        if totalClaims > 0:
            evidenceRatio = totalEvidence / len([token for token in doc if not (token.is_space or token.is_punct)])

        return totalClaims, totalEvidence, evidenceRatio

    df[["claims", "evidence", "evidenceRatio"]] = df["content"].progress_apply(
        determine_discourse
    ).apply(pd.Series)

##################################################################################

if __name__ == "__main__":
    # Set up argparser
    parser = argparse.ArgumentParser(description="NLP Analysis Arguments")
    parser.add_argument("--file", type=str, help="Filepath of the parquet file to be analysed")
    args = parser.parse_args()

    # Extract filename and subreddit from args.file
    filename = args.file.split('/')[-1].strip('.parquet')
    subreddit = filename.split("_")[0].strip()

    # Create output directories if they don't exist
    os.makedirs(f"plots/{subreddit}", exist_ok=True)
    os.makedirs(f"logs/{subreddit}", exist_ok=True)

    # set_defaults(progress_bar=True, progress_bar_desc="test", allow_dask_on_strings=True)
    
    # Get the dataframe, flatten it and add year_month column
    logger.info("EXPANDING DATAFRAME WITH YEAR/MONTH...")
    df = pd.read_parquet(args.file)
    df = flatten_df(df)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["year"] = df["datetime"].dt.year
    df["year_month"] = df["datetime"].dt.to_period("M")

    # Compute sentiment
    logger.info("ANALYZING SENTIMENT...")
    sentiment_analysis(df)

    # Compute topics
    logger.info("ANALYZING TOPICS...")
    topic_modelling(df, out_filename=filename, subreddit=subreddit)

    # Compute mental health score
    logger.info("COMPUTING MENTAL HEALTH SCORE...")
    mental_health_score(df)
    # mental_health_bert(df)

    # Analyze pronoun structure
    logger.info("ANALYZING PRONOUNS...")
    pronoun_analysis(df)

    # Analyze discourse structure
    logger.info("ANALYZING DISCOURSE STRUCTURE...")
    discourse_analysis(df)
    
    # Save the enhanced dataframe
    logger.info("SAVING ANALYSED DATAFRAME")
    df.to_parquet(f"data/reddit/nlp_analysis/{args.file.split('/')[-1].strip()}")
