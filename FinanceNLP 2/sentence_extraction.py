import random
import re
from string import punctuation
import unidecode
    
def preprocess(paragraph):

    # REMOVE SPECIAL CHARACTERS/ACCENTS
    paragraph = unidecode.unidecode(paragraph)

    # REMOVE NEW LINE CHARACTERS
    paragraph = paragraph.replace('\n', ' ')

    # REMOVE UNNECESSARY TOKENS
    nospace = ['"', "'", '\\']
    for char in nospace:
        paragraph = paragraph.replace(char, '')

    # REORDER END CHARS
    paragraph = paragraph.replace('.”', '”.')
    paragraph = paragraph.replace('!”', '”!')
    paragraph = paragraph.replace('?”', '”?')
    paragraph = paragraph.replace('.’', '’.')
    paragraph = paragraph.replace('.)', ').')

    # REMOVE ABBREVIATIONS
    paragraph = paragraph.replace('a.m.', 'AM')
    paragraph = paragraph.replace('p.m.', 'PM')
    paragraph = paragraph.replace('a.k.a.', 'AKA')
    paragraph = paragraph.replace('P.S.', 'PS')

    return paragraph


def get_first_n_sentence(paragraph, n):

    ## SPLIT PARAGRAPH ON APPROPRIATE PUNCTUATION
    r = re.compile(r"[^a-zA-Z0-9-',’”\—\"\'\$\(\)\s]+".format(re.escape(punctuation)))
    description = r.split(paragraph)[:-1]

    description = [d for d in description if len(d) > 15]

    ## VERIFY THAT THE DESCRIPTION CONTAINS >= N SENTENCES
    if n > len(description):
        return description

    ## GET RANDOM START INDEX FOR SENTENCES
    first_n_sentences = []
    if len(description) > n + 1:
        start_index =  random.randint(0,len(description)-n-1)
    else:
        start_index = 0
    
    for i in range(start_index, start_index+n):
        first_n_sentences.append(description[i].strip())

    return first_n_sentences