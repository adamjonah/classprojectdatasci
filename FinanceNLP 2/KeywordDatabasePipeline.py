import pandas as pd

import sys

from sentence_extraction import preprocess, get_first_n_sentence
from keyword_extraction import KeywordExtraction
from ProcessText import preprocess_text

# Print iterations progress 
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters/13685020
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def validate_dataframe(df):
    ## VALIDATE DATABASE FORMAT
    if len(df.columns) > 2:
        df.drop(df.columns.difference(['a','b']), 1, inplace=True) 

    if len(df.columns) != 2 or df.columns[0] != 'title' or df.columns[1] != 'description':
        print('Import CSV formatted incorrectly.')
        print('Please format as two columns: \'title\' and \'description\'.')
        exit()

    return df
        
def get_extraction_method(m:str, k:int, all_documents:list):

    if m in ['KEYBERT','BERT']:
        extraction_method = KeywordExtraction(all_documents, k).KeyBert
    elif m == 'YAKE':
        extraction_method = KeywordExtraction(all_documents, k).YAKE
    elif m == 'RAKE':
        extraction_method = KeywordExtraction(all_documents, k).RAKE
    elif m in ['KPMINER','KP_MINER']:
        extraction_method = KeywordExtraction(all_documents, k).KP_Miner
    elif m == 'TOPICRANK':
        extraction_method = KeywordExtraction(all_documents, k).TopicRank
    elif m == 'TEXTRANK':
        extraction_method = KeywordExtraction(all_documents, k).TextRank
    elif m in ['TFIDF', 'TF_IDF']:
        extraction_method = KeywordExtraction(all_documents, k).tf_idf
    elif m == 'IDF':
        extraction_method = KeywordExtraction(all_documents, k).idf
    else:
        print('Keyword extraction method not found.')
        print('Plese select from: \n - YAKE\n - RAKE\n - KP_Miner\n - KeyBert\n - TopicRank\n - TextRank\n - tf_idf\n - idf')
        exit()

    return extraction_method

def extraction_pipeline(df, extraction_method, n:int):
    l = df.shape[0]
    df_rows = []
    print("\n\nExtracting keywords in progress.")

    for i, row in df.iterrows():
        n_sentences = get_first_n_sentence(preprocess(row['description']), n)
        keywords = extraction_method(n_sentences)
        df_rows.append([row['title'], n_sentences, keywords])

        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)


    return pd.DataFrame(df_rows, columns =['title', 'n_sentences', 'keywords'])

if __name__ == '__main__':
    """
    run this file as:
    $ python3 KeywordDatabasePipeline.py [INPUT_DATABASE:csv] [EXTRACTION METHOD:str] [N:int] [K:int]
    """

    ## READ IN SOURCE DATABASE
    df = validate_dataframe(pd.read_csv(sys.argv[1], header=0))
    
    ## RUN PIPELINE
    text_sample = df.sample(frac=0.50)['description'].values.tolist()
    keyword_df = extraction_pipeline(df, get_extraction_method(sys.argv[2].upper(), int(sys.argv[4]), text_sample), int(sys.argv[3]))

    ## OUTPUT RESULT TO FILE
    keyword_df.to_csv(sys.argv[1][:-4]+'_keyword_database.csv')