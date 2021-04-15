import numpy as np
import pandas as pd
import nltk, scipy
import math, os, sys

from collections import Counter
from string import punctuation, ascii_lowercase
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
punct = punctuation + "“”’"
english_words = set()
try:
    with open('data/english.dic') as f:
        for line in f:
            english_words.add(line.lower().strip('\n').strip('\r\n'))
except IOError as e:
    print(e)
    sys.exit(1)

# headers = "id" "title" "publication" "author" "date" "year" "month" "url" "content"


def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    '''Returns the cosine similarity between V1 and V2. This is computed as
    V1 . V2 / |V1| * |V2|.
    
    Args:
        v1 (np.ndarray): A compatible vector.
        v2 (np.ndarray): A compatible vector.

    Returns:
        (float): Returns a number in the range [0, 1].
    '''
    return np.dot(v1, v2) / (v1.size * v2.size)


def jacc_coef(v1: np.ndarray, v2: np.ndarray) -> float:
    '''Returns the Jaccard Coefficient between V1 and V2. This is computed as
    V1 AND V2 / V1 OR V2.
    
    Args:
        v1 (np.ndarray): A compatible vector.
        v2 (np.ndarray): A compatible vector.

    Returns:
        (float): Returns a number in the range [0, 1].
    '''
    return np.intersect1d(v1, v2) / np.union1d(v1, v2)


def cos_dist(v1: np.ndarray, v2: np.ndarray):
    cross_sum = sum([v1[i] * v2[i] for i in range(v1.size)])
    mult = v1.sum() * v2.sum()
    if mult == 0:
        return math.nan
    return 1 - (cross_sum / mult)


def jacc_dist(v1: np.ndarray, v2: np.ndarray):
    min_sum = 0
    max_sum = 0
    for i in range(v1.size):
        min_sum += min(v1[i], v2[i])
        max_sum += max(v1[i], v2[i])
    if max_sum == 0:
        return math.nan
    return 1 - (min_sum / max_sum)


def js_divergence(v1: np.ndarray, v2: np.ndarray):
    ent_v1 = scipy.stats.entropy(v1)
    ent_v2 = scipy.stats.entropy(v2)
    ent_v3 = scipy.stats.entropy((v1 + v2) / 2)
    return ((ent_v1 + ent_v2) / 2) - ent_v3


def clean_word(w):
    q = w.strip(punct).strip(punct)
    if not q: return None
    if q in punct: return None
    if q in stop_words: return None
    if q not in english_words: return None
    return q


def local_word_count(content: str):
    joined = ''
    for sent in nltk.sent_tokenize(content.lower()):
        joined += ' '.join([clean_word(w) for w in sent.split() if clean_word(w)]) + ' '
    counts = Counter(joined.split())
    return counts


def global_word_count(filepath: str) -> None:
    '''Writes the global word count for every article in the specified FILEPATH to file.
    '''
    base_file = filepath.split(os.path.sep)[-1].strip('.csv')
    # word file is the word counts file itself
    word_file = os.path.abspath(os.path.join('data', base_file + '_word_count.txt'))
    input_frame = pd.read_csv(filepath, encoding='utf-8')
    counts = dict()
    for content in input_frame['content']:
        for w, c in local_word_count(content).items():
            if w not in counts: counts[w] = c
            else: counts[w] = counts[w] + c
    with open(word_file, 'w', encoding='utf-8') as outfile:
        for w, c in counts.items():
            outfile.write(f'{w}:{c}\n')


def parse_mfd(filepath='data/mft_dict.dic', encoding='utf-8'):
    '''Parses the MFT dictionary found in FILEPATH.

    Returns:
        (dict): A Python dict mapping MFT categories to sets of words that fall into
        that category.
    '''
    try:
        categories = dict()
        mfd = dict()
        with open(filepath, encoding=encoding) as f:
            read_categories = False
            for line in f:
                line = line.strip('\r\n').strip('\n')
                if line == '%':
                    read_categories = not read_categories
                    continue
                p = line.split()
                if read_categories:
                    if 'Virtue' in p[1]:
                        category = (p[1][:p[1].index('Virtue')], 'Virtue')
                    elif 'Vice' in p[1]:
                        category = (p[1][:p[1].index('Vice')], 'Vice')
                    elif 'General' in p[1]:
                        category = (p[1][:p[1].index('General')], 'General')
                    categories[p[0]] = category
                else:
                    category = categories[p[1]]
                    if category not in mfd: mfd[category] = set()
                    mfd[category].add(p[0].strip('*'))
        return mfd
    except IOError as e:
        print(e)
        sys.exit(1)

    
def extract_1d_headers(dictionary, prefix, join_char='-'):
    headers = []
    for key in dictionary.keys():
        headers.append(
            f'{join_char}'.join(
                [
                    prefix, 
                    *(key.split(join_char))
                ]
            ).lower()
        )
    return headers


def extract_2d_headers(dictionary, prefix, join_char='-'):
    headers = []
    for p in dictionary.keys():
        for q in dictionary[p].keys():
            headers.append(
                f'{join_char}'.join(
                    [
                        prefix, 
                        *(p.split(join_char)), 
                        *(q.split(join_char))
                    ]
                ).lower()
            )
    return headers


def flatten_1d_dict(dictionary):
    flattened = []
    for feature in dictionary.keys():
        flattened.append(dictionary[feature])
    return flattened


def flatten_2d_dict(dictionary):
    flattened = []
    for cat in dictionary.keys():
        for feature in dictionary[cat].keys():
            flattened.append(dictionary[cat][feature])
    return flattened