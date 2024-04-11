#
# Juliane Bruck 8297746
# 

ASSIGNMENT #1 
==============  


Question #1
-------------

* question1.py - This Python script performs text analysis tasks on a corpus of text data.

   The script provides insights into the characteristics of the text corpus, including word frequencies, type/token ratios, and other linguistic features. 
   It uses NLTK for tokenization and stopwords, and PrettyTable for creating a readable table for the results.

   INVOKE in the following way:   python question1.py

   OUTPUT: output.txt  and tokens.txt  along with a table printed to the command line showing the statistics.

   Sample Output:
   +-------------------------------------------------------------+----------------------------------------+
|                           Question                          |                 Result                 |
+-------------------------------------------------------------+----------------------------------------+
|                       # of tokens (b):                      |                4000731                 |
|                       # of types (b):                       |                 53331                  |
|                    type/token ratio (b):                    |          0.013330313885137491          |
|                tokens appeared only once (d):               |                 20650                  |
|           # of words (excluding punctuation) (e):           |                3873313                 |
|        type/token ratio (excluding punctuation) (e):        |          0.011069851571509972          |
|                  Top 3 most frequent words:                 |              the: 239948               |
|                                                             |              of: 151492                |
|                                                             |              to: 127055                |
|                                                             |                                        |
| type/token ratio (excluding punctuation and stopwords) (f): |          0.019415272397837968          |
|       Top 3 most frequent words (excluding stopwords):      |             shall: 48424               |
|                                                             |           Agreement: 37026             |
|                                                             |             Party: 20524               |
|                                                             |                                        |
|                 Top 3 most frequent bigrams:                |        ('set', 'forth'): 6017          |
|                                                             |     ('Agreement', 'shall'): 3166       |
|                                                             | ('Confidential', 'Information'): 2867  |
|                                                             |                                        |
+-------------------------------------------------------------+----------------------------------------+
PS D:\juliane\assignment1>




Question #2.
-------------

The project contains two python files:

   * question2_part1.py  -  This script will take as input 5 text files with each line containing two sentences and a citation.  
      A cosine similarity is performed on the two sentences after being processed by one of six models.
      The output of the script is a file containing scores of each sentence comparison.   The scores are added 
      one per line and correspond sequentially to the lines of the input file.   This operation is performed for each model.

      The output files are grouped under the "output" folder and split up per model that was used to evaluate the scores.

      The script will take approximately 2 hours to run.

      INVOKE in the following way:   python question2_part1.py

      OUTPUT will be left in the "output" subfolder broken down per model.

   
   * question2_part2.py - This script will invoke a Perl script that takes as input the gold standard scores and performs a 
         Pearson comparison against the scores produced by part 1 mentioned above. 

         Pearl is a pre-requisite to install so the invokation works.   
         On Windows this requires Strawberry Perl: https://www.perl.org/get.html


         INVOKE in the following way:   python question2_part2.py

         Sample output from this final step can be viewed below: 

                     Executing perl script on .\output\SBERT\STS2016.jub.output.answer-answer.txt
                     Pearson: 0.62886
                     Executing perl script on .\output\SBERT\STS2016.jub.output.headlines.txt
                     Pearson: 0.72618
                     Executing perl script on .\output\SBERT\STS2016.jub.output.plagiarism.txt
                     Pearson: 0.72609
                     Executing perl script on .\output\SBERT\STS2016.jub.output.postediting.txt
                     Pearson: 0.73852
                     Executing perl script on .\output\SBERT\STS2016.jub.output.question-question.txt
                     Pearson: 0.67805

                     = (0.62886 + 0.72618 + 0.72609 + 0.73852 + 0.67805) / 5
                     = 0.69954


                     Executing perl script on .\output\USE\STS2016.jub.output.answer-answer.txt
                     Pearson: 0.51553
                     Executing perl script on .\output\USE\STS2016.jub.output.headlines.txt
                     Pearson: 0.65043
                     Executing perl script on .\output\USE\STS2016.jub.output.plagiarism.txt
                     Pearson: 0.65565
                     Executing perl script on .\output\USE\STS2016.jub.output.postediting.txt
                     Pearson: 0.70200
                     Executing perl script on .\output\USE\STS2016.jub.output.question-question.txt
                     Pearson: 0.57455

                     = (0.51553 + 0.65043 + 0.65565 + 0.70200 + 0.57455 ) / 5
                     = 0.619632


                     Executing perl script on .\output\ROBERTA\STS2016.jub.output.answer-answer.txt
                     Pearson: 0.67306
                     Executing perl script on .\output\ROBERTA\STS2016.jub.output.headlines.txt
                     Pearson: 0.88144
                     Executing perl script on .\output\ROBERTA\STS2016.jub.output.plagiarism.txt
                     Pearson: 0.71391
                     Executing perl script on .\output\ROBERTA\STS2016.jub.output.postediting.txt
                     Pearson: 0.71793
                     Executing perl script on .\output\ROBERTA\STS2016.jub.output.question-question.txt
                     Pearson: 0.60763

                     = (0.67306 + 0.88144 + 0.71391 + 0.71793 + 0.60763 ) / 5
                     = 0.718794


                     Executing perl script on .\output\CLIP\STS2016.jub.output.answer-answer.txt
                     Pearson: 0.03079
                     Executing perl script on .\output\CLIP\STS2016.jub.output.headlines.txt
                     Pearson: 0.56338
                     Executing perl script on .\output\CLIP\STS2016.jub.output.plagiarism.txt
                     Pearson: 0.21034
                     Executing perl script on .\output\CLIP\STS2016.jub.output.postediting.txt
                     Pearson: 0.12277
                     Executing perl script on .\output\CLIP\STS2016.jub.output.question-question.txt
                     Pearson: 0.49636

                     = (0.03079 + 0.56338 + 0.21034 + 0.12277 + 0.49636 ) / 5
                     = 0.284728


                     Executing perl script on .\output\MPNET\STS2016.jub.output.answer-answer.txt
                     Pearson: 0.67750
                     Executing perl script on .\output\MPNET\STS2016.jub.output.headlines.txt
                     Pearson: 0.76432
                     Executing perl script on .\output\MPNET\STS2016.jub.output.plagiarism.txt
                     Pearson: 0.71739
                     Executing perl script on .\output\MPNET\STS2016.jub.output.postediting.txt
                     Pearson: 0.75050
                     Executing perl script on .\output\MPNET\STS2016.jub.output.question-question.txt
                     Pearson: 0.73253

                     = (0.67750 + 0.76432 + 0.71739 + 0.75050 + 0.73253 ) /5
                     = 0.728448


                     Executing perl script on .\output\DISTILROBERTA\STS2016.jub.output.answer-answer.txt
                     Pearson: 0.55599
                     Executing perl script on .\output\DISTILROBERTA\STS2016.jub.output.headlines.txt
                     Pearson: 0.68416
                     Executing perl script on .\output\DISTILROBERTA\STS2016.jub.output.plagiarism.txt
                     Pearson: 0.70528
                     Executing perl script on .\output\DISTILROBERTA\STS2016.jub.output.postediting.txt
                     Pearson: 0.71410
                     Executing perl script on .\output\DISTILROBERTA\STS2016.jub.output.question-question.txt
                     Pearson: 0.64656

                     = (0.55599 + 0.68416 + 0.70528 + 0.71410 + 0.64656 ) / 5
                     = 0.661218