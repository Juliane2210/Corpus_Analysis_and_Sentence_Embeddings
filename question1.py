import os
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from prettytable import PrettyTable

# Download NLTK stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Create a table instance
table = PrettyTable()

# Path to the Atticus dataset
dataset_path = ".\\CUAD_v1\\full_contract_txt"

# Output file paths
output_file_path = "output.txt"
tokens_file_path = "tokens.txt"

# Function to concatenate text files in a directory


def concatenate_files(directory):
    corpus = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):

            print(f"Processing file: {filename}")

            file = open(os.path.join(directory, filename),
                        'r', encoding='utf-8')
            corpus += file.read() + " "
            file.close()

    return corpus


# Function to tokenize and preprocess text
def tokenize_and_preprocess(text):
    tokens = word_tokenize(text)
    # Remove punctuation and symbols
    tokens = [re.sub(r'\W', '', token) for token in tokens]
    # Remove empty tokens
    tokens = [token for token in tokens if token]
    return tokens


# Concatenate text files to form the corpus
corpus = concatenate_files(dataset_path)

# Tokenize and preprocess the corpus
tokens = tokenize_and_preprocess(corpus)


# Write the tokenizer's output to a file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write("\n".join(tokens))

# Count the occurrences of each token
token_counts = Counter(tokens)

# Write token frequencies to a file
with open(tokens_file_path, 'w', encoding='utf-8') as tokens_file:
    for token, count in token_counts.most_common():
        tokens_file.write(f"{token}\t{count}\n")

# Calculate requested statistics
num_tokens = len(tokens)
num_types = len(token_counts)
type_token_ratio = num_types / num_tokens
single_occurrence_tokens = sum(
    1 for count in token_counts.values() if count == 1)

# Exclude punctuation and calculate word-related statistics
words = [token for token in tokens if token.isalpha()]
num_words = len(words)
type_token_ratio_words = len(set(words)) / num_words
word_counts = Counter(words)

# Exclude stopwords and calculate statistics
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
type_token_ratio_filtered = len(set(filtered_words)) / len(filtered_words)
filtered_word_counts = Counter(filtered_words)

# Compute bigrams
bigrams = list(zip(tokens[:-1], tokens[1:]))
filtered_bigrams = [bigram for bigram in bigrams if all(
    word.isalpha() and word.lower() not in stop_words for word in bigram)]
bigram_counts = Counter(filtered_bigrams)

# Display results


table.field_names = ["Question", "Result"]

# Add data to the table
table.add_row(["# of tokens (b):", num_tokens])
table.add_row(["# of types (b):", num_types])


table.add_row(["type/token ratio (b):", type_token_ratio])
table.add_row(["tokens appeared only once (d):", single_occurrence_tokens])
table.add_row(["# of words (excluding punctuation) (e):", num_words])
table.add_row(
    ["type/token ratio (excluding punctuation) (e):", type_token_ratio_words])


# Display top 3 most frequent words and their frequencies
top_words = word_counts.most_common(3)
# print("Top 3 most frequent words:")

three_frequent_words = ""
for word, frequency in top_words:
    three_frequent_words += f"{word}: {frequency} \n"

table.add_row(
    ["Top 3 most frequent words:", three_frequent_words])

table.add_row(
    ["type/token ratio (excluding punctuation and stopwords) (f):", type_token_ratio_filtered])


# Display top 3 most frequent words and their frequencies (excluding stopwords)
top_filtered_words = filtered_word_counts.most_common(3)
top_three_filtered_words = ""
for word, frequency in top_filtered_words:
    top_three_filtered_words += f"{word}: {frequency} \n"

table.add_row(
    ["Top 3 most frequent words (excluding stopwords):", top_three_filtered_words])


# Display top 3 most frequent bigrams and their frequencies
top_bigrams = bigram_counts.most_common(3)

three_frequent_bigram = ""
for bigram, frequency in top_bigrams:
    three_frequent_bigram += f"{bigram}: {frequency} \n"

table.add_row(
    ["Top 3 most frequent bigrams:", three_frequent_bigram])

# Print the table
print(table)
