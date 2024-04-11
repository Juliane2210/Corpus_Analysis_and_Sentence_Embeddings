#
# Juliane Bruck , 8297746
#
# This script will invoke the script to compare the gold standard to data obtained from our models
#
# to execute perl script on Windows, we installed Straberry perl : https://www.perl.org/get.html
#

# Part #1 Compare the output to the gold standard

from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import subprocess

# The script being invoked to compute correlations is the a perl script
# CUAD_v1/sts2016-english-with-gs-v1.0/correlation-noconfidence.pl
perl_script_path = '.\\sts2016-english-with-gs-v1.0\\correlation-noconfidence.pl'

goldstandard_path_answer = ".\\sts2016-english-with-gs-v1.0\\STS2016.gs.answer-answer.txt"
goldstandard_path_headlines = ".\\sts2016-english-with-gs-v1.0\\STS2016.gs.headlines.txt"
goldstandard_path_plaigiarism = ".\\sts2016-english-with-gs-v1.0\\STS2016.gs.plagiarism.txt"
goldstandard_path_postediting = ".\\sts2016-english-with-gs-v1.0\\STS2016.gs.postediting.txt"
goldstandard_path_question = ".\\sts2016-english-with-gs-v1.0\\STS2016.gs.question-question.txt"


# SBERT processing
SBERT_output_path_answer = ".\\output\\SBERT\\STS2016.jub.output.answer-answer.txt"
SBERT_output_path_headlines = ".\\output\\SBERT\\STS2016.jub.output.headlines.txt"
SBERT_output_path_plaigiarism = ".\\output\\SBERT\\STS2016.jub.output.plagiarism.txt"
SBERT_output_path_postediting = ".\\output\\SBERT\\STS2016.jub.output.postediting.txt"
SBERT_output_path_question = ".\\output\\SBERT\\STS2016.jub.output.question-question.txt"

perl_command_SBERT_answer = ['perl', perl_script_path,
                             goldstandard_path_answer, SBERT_output_path_answer]

perl_command_SBERT_headlines = ['perl', perl_script_path,
                                goldstandard_path_headlines, SBERT_output_path_headlines]

perl_command_SBERT_plaigiarism = ['perl', perl_script_path,
                                  goldstandard_path_plaigiarism, SBERT_output_path_plaigiarism]

perl_command_SBERT_postediting = ['perl', perl_script_path,
                                  goldstandard_path_postediting, SBERT_output_path_postediting]

perl_command_SBERT_question = ['perl', perl_script_path,
                               goldstandard_path_question, SBERT_output_path_question]

try:
    # Run the Perl script scripts on SBERT output for various files
    print(f"Executing perl script on {SBERT_output_path_answer}")
    result = subprocess.run(perl_command_SBERT_answer, check=True)
    print(f"Executing perl script on {SBERT_output_path_headlines}")
    result = subprocess.run(perl_command_SBERT_headlines, check=True)
    print(f"Executing perl script on {SBERT_output_path_plaigiarism}")
    result = subprocess.run(perl_command_SBERT_plaigiarism, check=True)
    print(f"Executing perl script on {SBERT_output_path_postediting}")
    result = subprocess.run(perl_command_SBERT_postediting, check=True)
    print(f"Executing perl script on {SBERT_output_path_question}")
    result = subprocess.run(perl_command_SBERT_question, check=True)

except subprocess.CalledProcessError as e:
    print(f"Error executing Perl script: {e}")


# USE model processing
USE_output_path_answer = ".\\output\\USE\\STS2016.jub.output.answer-answer.txt"
USE_output_path_headlines = ".\\output\\USE\\STS2016.jub.output.headlines.txt"
USE_output_path_plaigiarism = ".\\output\\USE\\STS2016.jub.output.plagiarism.txt"
USE_output_path_postediting = ".\\output\\USE\\STS2016.jub.output.postediting.txt"
USE_output_path_question = ".\\output\\USE\\STS2016.jub.output.question-question.txt"

perl_command_USE_answer = ['perl', perl_script_path,
                           goldstandard_path_answer, USE_output_path_answer]

perl_command_USE_headlines = ['perl', perl_script_path,
                              goldstandard_path_headlines, USE_output_path_headlines]

perl_command_USE_plaigiarism = ['perl', perl_script_path,
                                goldstandard_path_plaigiarism, USE_output_path_plaigiarism]

perl_command_USE_postediting = ['perl', perl_script_path,
                                goldstandard_path_postediting, USE_output_path_postediting]

perl_command_USE_question = ['perl', perl_script_path,
                             goldstandard_path_question, USE_output_path_question]

try:
    # Run the Perl script scripts on USE output for various files
    print(f"Executing perl script on {USE_output_path_answer}")
    result = subprocess.run(perl_command_USE_answer, check=True)
    print(f"Executing perl script on {USE_output_path_headlines}")
    result = subprocess.run(perl_command_USE_headlines, check=True)
    print(f"Executing perl script on {USE_output_path_plaigiarism}")
    result = subprocess.run(perl_command_USE_plaigiarism, check=True)
    print(f"Executing perl script on {USE_output_path_postediting}")
    result = subprocess.run(perl_command_USE_postediting, check=True)
    print(f"Executing perl script on {USE_output_path_question}")
    result = subprocess.run(perl_command_USE_question, check=True)

except subprocess.CalledProcessError as e:
    print(f"Error executing Perl script: {e}")


# ROBERTA model processing
ROBERTA_output_path_answer = ".\\output\\ROBERTA\\STS2016.jub.output.answer-answer.txt"
ROBERTA_output_path_headlines = ".\\output\\ROBERTA\\STS2016.jub.output.headlines.txt"
ROBERTA_output_path_plaigiarism = ".\\output\\ROBERTA\\STS2016.jub.output.plagiarism.txt"
ROBERTA_output_path_postediting = ".\\output\\ROBERTA\\STS2016.jub.output.postediting.txt"
ROBERTA_output_path_question = ".\\output\\ROBERTA\\STS2016.jub.output.question-question.txt"

perl_command_ROBERTA_answer = ['perl', perl_script_path,
                               goldstandard_path_answer, ROBERTA_output_path_answer]

perl_command_ROBERTA_headlines = ['perl', perl_script_path,
                                  goldstandard_path_headlines, ROBERTA_output_path_headlines]

perl_command_ROBERTA_plaigiarism = ['perl', perl_script_path,
                                    goldstandard_path_plaigiarism, ROBERTA_output_path_plaigiarism]

perl_command_ROBERTA_postediting = ['perl', perl_script_path,
                                    goldstandard_path_postediting, ROBERTA_output_path_postediting]

perl_command_ROBERTA_question = ['perl', perl_script_path,
                                 goldstandard_path_question, ROBERTA_output_path_question]

try:
    # Run the Perl script scripts on ROBERTA output for various files
    print(f"Executing perl script on {ROBERTA_output_path_answer}")
    result = subprocess.run(perl_command_ROBERTA_answer, check=True)
    print(f"Executing perl script on {ROBERTA_output_path_headlines}")
    result = subprocess.run(perl_command_ROBERTA_headlines, check=True)
    print(f"Executing perl script on {ROBERTA_output_path_plaigiarism}")
    result = subprocess.run(perl_command_ROBERTA_plaigiarism, check=True)
    print(f"Executing perl script on {ROBERTA_output_path_postediting}")
    result = subprocess.run(perl_command_ROBERTA_postediting, check=True)
    print(f"Executing perl script on {ROBERTA_output_path_question}")
    result = subprocess.run(perl_command_ROBERTA_question, check=True)

except subprocess.CalledProcessError as e:
    print(f"Error executing Perl script: {e}")


# CLIP model processing
CLIP_output_path_answer = ".\\output\\CLIP\\STS2016.jub.output.answer-answer.txt"
CLIP_output_path_headlines = ".\\output\\CLIP\\STS2016.jub.output.headlines.txt"
CLIP_output_path_plaigiarism = ".\\output\\CLIP\\STS2016.jub.output.plagiarism.txt"
CLIP_output_path_postediting = ".\\output\\CLIP\\STS2016.jub.output.postediting.txt"
CLIP_output_path_question = ".\\output\\CLIP\\STS2016.jub.output.question-question.txt"

perl_command_CLIP_answer = ['perl', perl_script_path,
                            goldstandard_path_answer, CLIP_output_path_answer]

perl_command_CLIP_headlines = ['perl', perl_script_path,
                               goldstandard_path_headlines, CLIP_output_path_headlines]

perl_command_CLIP_plaigiarism = ['perl', perl_script_path,
                                 goldstandard_path_plaigiarism, CLIP_output_path_plaigiarism]

perl_command_CLIP_postediting = ['perl', perl_script_path,
                                 goldstandard_path_postediting, CLIP_output_path_postediting]

perl_command_CLIP_question = ['perl', perl_script_path,
                              goldstandard_path_question, CLIP_output_path_question]

try:
    # Run the Perl script scripts on CLIP output for various files
    print(f"Executing perl script on {CLIP_output_path_answer}")
    result = subprocess.run(perl_command_CLIP_answer, check=True)
    print(f"Executing perl script on {CLIP_output_path_headlines}")
    result = subprocess.run(perl_command_CLIP_headlines, check=True)
    print(f"Executing perl script on {CLIP_output_path_plaigiarism}")
    result = subprocess.run(perl_command_CLIP_plaigiarism, check=True)
    print(f"Executing perl script on {CLIP_output_path_postediting}")
    result = subprocess.run(perl_command_CLIP_postediting, check=True)
    print(f"Executing perl script on {CLIP_output_path_question}")
    result = subprocess.run(perl_command_CLIP_question, check=True)

except subprocess.CalledProcessError as e:
    print(f"Error executing Perl script: {e}")


# MPNET model processing
MPNET_output_path_answer = ".\\output\\MPNET\\STS2016.jub.output.answer-answer.txt"
MPNET_output_path_headlines = ".\\output\\MPNET\\STS2016.jub.output.headlines.txt"
MPNET_output_path_plaigiarism = ".\\output\\MPNET\\STS2016.jub.output.plagiarism.txt"
MPNET_output_path_postediting = ".\\output\\MPNET\\STS2016.jub.output.postediting.txt"
MPNET_output_path_question = ".\\output\\MPNET\\STS2016.jub.output.question-question.txt"

perl_command_MPNET_answer = ['perl', perl_script_path,
                             goldstandard_path_answer, MPNET_output_path_answer]

perl_command_MPNET_headlines = ['perl', perl_script_path,
                                goldstandard_path_headlines, MPNET_output_path_headlines]

perl_command_MPNET_plaigiarism = ['perl', perl_script_path,
                                  goldstandard_path_plaigiarism, MPNET_output_path_plaigiarism]

perl_command_MPNET_postediting = ['perl', perl_script_path,
                                  goldstandard_path_postediting, MPNET_output_path_postediting]

perl_command_MPNET_question = ['perl', perl_script_path,
                               goldstandard_path_question, MPNET_output_path_question]

try:
    # Run the Perl script scripts on MPNET output for various files
    print(f"Executing perl script on {MPNET_output_path_answer}")
    result = subprocess.run(perl_command_MPNET_answer, check=True)
    print(f"Executing perl script on {MPNET_output_path_headlines}")
    result = subprocess.run(perl_command_MPNET_headlines, check=True)
    print(f"Executing perl script on {MPNET_output_path_plaigiarism}")
    result = subprocess.run(perl_command_MPNET_plaigiarism, check=True)
    print(f"Executing perl script on {MPNET_output_path_postediting}")
    result = subprocess.run(perl_command_MPNET_postediting, check=True)
    print(f"Executing perl script on {MPNET_output_path_question}")
    result = subprocess.run(perl_command_MPNET_question, check=True)

except subprocess.CalledProcessError as e:
    print(f"Error executing Perl script: {e}")


# DISTILROBERTA model processing
DISTILROBERTA_output_path_answer = ".\\output\\DISTILROBERTA\\STS2016.jub.output.answer-answer.txt"
DISTILROBERTA_output_path_headlines = ".\\output\\DISTILROBERTA\\STS2016.jub.output.headlines.txt"
DISTILROBERTA_output_path_plaigiarism = ".\\output\\DISTILROBERTA\\STS2016.jub.output.plagiarism.txt"
DISTILROBERTA_output_path_postediting = ".\\output\\DISTILROBERTA\\STS2016.jub.output.postediting.txt"
DISTILROBERTA_output_path_question = ".\\output\\DISTILROBERTA\\STS2016.jub.output.question-question.txt"

perl_command_DISTILROBERTA_answer = ['perl', perl_script_path,
                                     goldstandard_path_answer, DISTILROBERTA_output_path_answer]

perl_command_DISTILROBERTA_headlines = ['perl', perl_script_path,
                                        goldstandard_path_headlines, DISTILROBERTA_output_path_headlines]

perl_command_DISTILROBERTA_plaigiarism = ['perl', perl_script_path,
                                          goldstandard_path_plaigiarism, DISTILROBERTA_output_path_plaigiarism]

perl_command_DISTILROBERTA_postediting = ['perl', perl_script_path,
                                          goldstandard_path_postediting, DISTILROBERTA_output_path_postediting]

perl_command_DISTILROBERTA_question = ['perl', perl_script_path,
                                       goldstandard_path_question, DISTILROBERTA_output_path_question]

try:
    # Run the Perl script scripts on DISTILROBERTA output for various files
    print(f"Executing perl script on {DISTILROBERTA_output_path_answer}")
    result = subprocess.run(perl_command_DISTILROBERTA_answer, check=True)
    print(f"Executing perl script on {DISTILROBERTA_output_path_headlines}")
    result = subprocess.run(perl_command_DISTILROBERTA_headlines, check=True)
    print(f"Executing perl script on {DISTILROBERTA_output_path_plaigiarism}")
    result = subprocess.run(perl_command_DISTILROBERTA_plaigiarism, check=True)
    print(f"Executing perl script on {DISTILROBERTA_output_path_postediting}")
    result = subprocess.run(perl_command_DISTILROBERTA_postediting, check=True)
    print(f"Executing perl script on {DISTILROBERTA_output_path_question}")
    result = subprocess.run(perl_command_DISTILROBERTA_question, check=True)

except subprocess.CalledProcessError as e:
    print(f"Error executing Perl script: {e}")
