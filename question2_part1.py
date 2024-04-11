#
# Juliane Bruck #8297746
# Assignment1 Question 2
#


import subprocess
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.metrics import mean_squared_error
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel, T5Model
import torch
import tensorflow_hub as hub
import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

# Download necessary NLTK data (do this once)
nltk.download('punkt')


# SBERT model
m_modelSBERT = SentenceTransformer('all-MiniLM-L6-v2')

# Load the RoBERTa model
m_modelROBERTA = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

# Load a pre-trained CLIP model
m_modelCLIP = SentenceTransformer('clip-ViT-B-32')

# Load a pre-trained Distill ROBERTA model
m_modelDISTILROBERTA = SentenceTransformer(
    'paraphrase-distilroberta-base-v1')

# Load a pre-trained MPNET model
m_modelMPNET = SentenceTransformer('all-mpnet-base-v2')

# Load USE model
m_modelUSE = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder/4")

#
# Scale the cosine similarity from range [-1, 1] to [0, 5]
#


def map_to_range(input_number):
    output_number = ((input_number + 1) / 2) * 5
    return round(output_number)


#
# Model for SBERT
#
def calculate_similarity_SBERT(sentence1, sentence2):

    embeddings = m_modelSBERT.encode([sentence1, sentence2])
    # Ensure embeddings are normalized
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Step 3: Compute cosine similarity
    similarity_score = cosine_similarity(
        [embeddings[0]], [embeddings[1]])[0][0]

    similarity_score = map_to_range(similarity_score)
    # print(f"{sentence1} >>>> {sentence2} == {similarity_score}")
    return similarity_score


#
# Model for ROBERTA
#
def calculate_similarity_ROBERTA(sentence1, sentence2):
    embeddings = m_modelROBERTA.encode([sentence1, sentence2])

    similarity_score = cosine_similarity(
        [embeddings[0]], [embeddings[1]])[0][0]

    return map_to_range(similarity_score)


#
# Model for USE
#
def calculate_similarity_USE(sentence1, sentence2):

    embeddings = m_modelUSE([sentence1, sentence2])
    similarity_score = cosine_similarity(
        [embeddings[0]], [embeddings[1]])[0][0]
    similarity_score = map_to_range(similarity_score)
    return similarity_score


#
# Model for CLIP
#
def calculate_similarity_CLIP(sentence1, sentence2):
    try:
        # Compute embeddings for both sentences
        embeddings1 = m_modelCLIP.encode(sentence1)
        embeddings2 = m_modelCLIP.encode(sentence2)

        # Compute cosine similarity
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

        # Convert to a numpy number for easy reading
        similarity_score = cosine_similarity.numpy()[0][0]
        return map_to_range(similarity_score)

    except Exception as e:
        # Retry using a max length since the model is restrictive.
        sentence1 = sentence1[:77]
        sentence2 = sentence2[:77]

        embeddings1 = m_modelCLIP.encode(sentence1)
        embeddings2 = m_modelCLIP.encode(sentence2)

        # Compute cosine similarity
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

        # Convert to a numpy number for easy reading
        similarity_score = cosine_similarity.numpy()[0][0]
        return map_to_range(similarity_score)


#
# Model for DISTILLROBERTA
#
def calculate_similarity_DISTILROBERTA(sentence1, sentence2):

    # Compute embeddings for both sentences
    embeddings1 = m_modelDISTILROBERTA.encode(sentence1)
    embeddings2 = m_modelDISTILROBERTA.encode(sentence2)
    # Compute cosine similarity
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Convert to a numpy number for easy reading
    similarity_score = cosine_similarity.numpy()[0][0]
    return map_to_range(similarity_score)


#
# Model for MPNET
#
def calculate_similarity_MPNET(sentence1, sentence2):
    # Compute embeddings for both sentences
    embeddings1 = m_modelMPNET.encode(sentence1)
    embeddings2 = m_modelMPNET.encode(sentence2)
    # Compute cosine similarity
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Convert to a numpy number for easy reading
    similarity_score = cosine_similarity.numpy()[0][0]
    return map_to_range(similarity_score)


#
# Write output with one result in range 0,5 per line
#
def output_score(output_file_path, scores):
    try:
        with open(output_file_path, 'w') as file:
            for score in scores:
                file.write(str(score) + '\n')
        print(f"Scores successfully written to {output_file_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")


#
# Given an input file containing 2 sentences per line, produce an output file
# with one score per line
#
def process_and_score(input_file_path, output_file_path, evaluationFunction):
    line_count = 0
    scores = []
    # some of the lines are delimited only by tabs or colons
    pattern = r'\t|:'

    # extract sentences from each line in the text
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        first_two_sentences_per_line = []
        for line in lines:
            line_count += 1
            # Tokenize the line into sentences
            sentences = nltk.regexp_tokenize(line, pattern, gaps=True)

            # Extract the first two sentences if available
            if len(sentences) >= 2:
                first_two_sentences_per_line = sentences[:2]

                similarity_score = evaluationFunction(
                    first_two_sentences_per_line[0], first_two_sentences_per_line[1])

                scores.append(similarity_score)
            else:
                # Some issue in parsing, enter a 0 value so the number of lines in the output match the GS
                scores.append(0)
                print(f"{input_file_path} #{line_count} --- invalid line ---")

    # output scores to the file
    output_score(output_file_path, scores)


# Load the STS dataset
dataset_path_answer = ".\\sts2016-english-with-gs-v1.0\\STS2016.input.answer-answer.txt"
dataset_path_headlines = ".\\sts2016-english-with-gs-v1.0\\STS2016.input.headlines.txt"
dataset_path_plaigiarism = ".\\sts2016-english-with-gs-v1.0\\STS2016.input.plagiarism.txt"
dataset_path_postediting = ".\\sts2016-english-with-gs-v1.0\\STS2016.input.postediting.txt"
dataset_path_question = ".\\sts2016-english-with-gs-v1.0\\STS2016.input.question-question.txt"


# SBERT processing
SBERT_output_path_answer = ".\\output\\SBERT\\STS2016.jub.output.answer-answer.txt"
SBERT_output_path_headlines = ".\\output\\SBERT\\STS2016.jub.output.headlines.txt"
SBERT_output_path_plaigiarism = ".\\output\\SBERT\\STS2016.jub.output.plagiarism.txt"
SBERT_output_path_postediting = ".\\output\\SBERT\\STS2016.jub.output.postediting.txt"
SBERT_output_path_question = ".\\output\\SBERT\\STS2016.jub.output.question-question.txt"

print(f"SBERT processing {dataset_path_answer}")
process_and_score(dataset_path_answer, SBERT_output_path_answer,
                  calculate_similarity_SBERT)
print(f"SBERT processing {dataset_path_headlines}")
process_and_score(dataset_path_headlines, SBERT_output_path_headlines,
                  calculate_similarity_SBERT)
print(f"SBERT processing {dataset_path_plaigiarism}")
process_and_score(dataset_path_plaigiarism, SBERT_output_path_plaigiarism,
                  calculate_similarity_SBERT)
print(f"SBERT processing {dataset_path_postediting}")
process_and_score(dataset_path_postediting, SBERT_output_path_postediting,
                  calculate_similarity_SBERT)
print(f"SBERT processing {dataset_path_question}")
process_and_score(dataset_path_question, SBERT_output_path_question,
                  calculate_similarity_SBERT)


# USE model processing
USE_output_path_answer = ".\\output\\USE\\STS2016.jub.output.answer-answer.txt"
USE_output_path_headlines = ".\\output\\USE\\STS2016.jub.output.headlines.txt"
USE_output_path_plaigiarism = ".\\output\\USE\\STS2016.jub.output.plagiarism.txt"
USE_output_path_postediting = ".\\output\\USE\\STS2016.jub.output.postediting.txt"
USE_output_path_question = ".\\output\\USE\\STS2016.jub.output.question-question.txt"

print(f"USE processing {dataset_path_answer}")
process_and_score(dataset_path_answer, USE_output_path_answer,
                  calculate_similarity_USE)
print(f"USE processing {dataset_path_headlines}")
process_and_score(dataset_path_headlines, USE_output_path_headlines,
                  calculate_similarity_USE)
print(f"USE processing {dataset_path_plaigiarism}")
process_and_score(dataset_path_plaigiarism, USE_output_path_plaigiarism,
                  calculate_similarity_USE)
print(f"USE processing {dataset_path_postediting}")
process_and_score(dataset_path_postediting, USE_output_path_postediting,
                  calculate_similarity_USE)
print(f"USE processing {dataset_path_question}")
process_and_score(dataset_path_question, USE_output_path_question,
                  calculate_similarity_USE)


# ROBERTA model processing
ROBERTA_output_path_answer = ".\\output\\ROBERTA\\STS2016.jub.output.answer-answer.txt"
ROBERTA_output_path_headlines = ".\\output\\ROBERTA\\STS2016.jub.output.headlines.txt"
ROBERTA_output_path_plaigiarism = ".\\output\\ROBERTA\\STS2016.jub.output.plagiarism.txt"
ROBERTA_output_path_postediting = ".\\output\\ROBERTA\\STS2016.jub.output.postediting.txt"
ROBERTA_output_path_question = ".\\output\\ROBERTA\\STS2016.jub.output.question-question.txt"

print(f"ROBERTA processing {dataset_path_answer}")
process_and_score(dataset_path_answer, ROBERTA_output_path_answer,
                  calculate_similarity_ROBERTA)
print(f"ROBERTA processing {dataset_path_headlines}")
process_and_score(dataset_path_headlines, ROBERTA_output_path_headlines,
                  calculate_similarity_ROBERTA)
print(f"ROBERTA processing {dataset_path_plaigiarism}")
process_and_score(dataset_path_plaigiarism, ROBERTA_output_path_plaigiarism,
                  calculate_similarity_ROBERTA)
print(f"ROBERTA processing {dataset_path_postediting}")
process_and_score(dataset_path_postediting, ROBERTA_output_path_postediting,
                  calculate_similarity_ROBERTA)
print(f"ROBERTA processing {dataset_path_question}")
process_and_score(dataset_path_question, ROBERTA_output_path_question,
                  calculate_similarity_ROBERTA)


# MPNET model processing
MPNET_output_path_answer = ".\\output\\MPNET\\STS2016.jub.output.answer-answer.txt"
MPNET_output_path_headlines = ".\\output\\MPNET\\STS2016.jub.output.headlines.txt"
MPNET_output_path_plaigiarism = ".\\output\\MPNET\\STS2016.jub.output.plagiarism.txt"
MPNET_output_path_postediting = ".\\output\\MPNET\\STS2016.jub.output.postediting.txt"
MPNET_output_path_question = ".\\output\\MPNET\\STS2016.jub.output.question-question.txt"

print(f"MPNET processing {dataset_path_answer}")
process_and_score(dataset_path_answer, MPNET_output_path_answer,
                  calculate_similarity_MPNET)
print(f"MPNET processing {dataset_path_headlines}")
process_and_score(dataset_path_headlines, MPNET_output_path_headlines,
                  calculate_similarity_MPNET)
print(f"MPNET processing {dataset_path_plaigiarism}")
process_and_score(dataset_path_plaigiarism, MPNET_output_path_plaigiarism,
                  calculate_similarity_MPNET)
print(f"MPNET processing {dataset_path_postediting}")
process_and_score(dataset_path_postediting, MPNET_output_path_postediting,
                  calculate_similarity_MPNET)
print(f"MPNET processing {dataset_path_question}")
process_and_score(dataset_path_question, MPNET_output_path_question,
                  calculate_similarity_MPNET)


# DISTILROBERTA model processing
DISTILROBERTA_output_path_answer = ".\\output\\DISTILROBERTA\\STS2016.jub.output.answer-answer.txt"
DISTILROBERTA_output_path_headlines = ".\\output\\DISTILROBERTA\\STS2016.jub.output.headlines.txt"
DISTILROBERTA_output_path_plaigiarism = ".\\output\\DISTILROBERTA\\STS2016.jub.output.plagiarism.txt"
DISTILROBERTA_output_path_postediting = ".\\output\\DISTILROBERTA\\STS2016.jub.output.postediting.txt"
DISTILROBERTA_output_path_question = ".\\output\\DISTILROBERTA\\STS2016.jub.output.question-question.txt"

print(f"DISTILROBERTA processing {dataset_path_answer}")
process_and_score(dataset_path_answer, DISTILROBERTA_output_path_answer,
                  calculate_similarity_DISTILROBERTA)
print(f"DISTILROBERTA processing {dataset_path_headlines}")
process_and_score(dataset_path_headlines, DISTILROBERTA_output_path_headlines,
                  calculate_similarity_DISTILROBERTA)
print(f"DISTILROBERTA processing {dataset_path_plaigiarism}")
process_and_score(dataset_path_plaigiarism, DISTILROBERTA_output_path_plaigiarism,
                  calculate_similarity_DISTILROBERTA)
print(f"DISTILROBERTA processing {dataset_path_postediting}")
process_and_score(dataset_path_postediting, DISTILROBERTA_output_path_postediting,
                  calculate_similarity_DISTILROBERTA)
print(f"DISTILROBERTA processing {dataset_path_question}")
process_and_score(dataset_path_question, DISTILROBERTA_output_path_question,
                  calculate_similarity_DISTILROBERTA)


# CLIP model processing
CLIP_output_path_answer = ".\\output\\CLIP\\STS2016.jub.output.answer-answer.txt"
CLIP_output_path_headlines = ".\\output\\CLIP\\STS2016.jub.output.headlines.txt"
CLIP_output_path_plaigiarism = ".\\output\\CLIP\\STS2016.jub.output.plagiarism.txt"
CLIP_output_path_postediting = ".\\output\\CLIP\\STS2016.jub.output.postediting.txt"
CLIP_output_path_question = ".\\output\\CLIP\\STS2016.jub.output.question-question.txt"


print(f"CLIP processing {dataset_path_answer}")
process_and_score(dataset_path_answer, CLIP_output_path_answer,
                  calculate_similarity_CLIP)
print(f"CLIP processing {dataset_path_headlines}")
process_and_score(dataset_path_headlines, CLIP_output_path_headlines,
                  calculate_similarity_CLIP)
print(f"CLIP processing {dataset_path_plaigiarism}")
process_and_score(dataset_path_plaigiarism, CLIP_output_path_plaigiarism,
                  calculate_similarity_CLIP)
print(f"CLIP processing {dataset_path_postediting}")
process_and_score(dataset_path_postediting, CLIP_output_path_postediting,
                  calculate_similarity_CLIP)
print(f"CLIP processing {dataset_path_question}")
process_and_score(dataset_path_question, CLIP_output_path_question,
                  calculate_similarity_CLIP)


print(f"\n\n ====== FINISHED processing ========")
