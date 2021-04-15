import math, os, sys
from collections import Counter
import heapq
from operator import itemgetter
from multiprocessing import Process

import nltk
import numpy as np
import pandas as pd
import pymongo
import scipy
import torch

from model import load_kv_model
from tools import (
    clean_word, english_words, punct, 
    stop_words, parse_mfd,
    cos_sim, local_word_count, cos_dist,
    jacc_dist, js_divergence,
    extract_1d_headers, extract_2d_headers,
    flatten_1d_dict, flatten_2d_dict
)
from gensim.models import Word2Vec
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def load_semantic_axes(infile='data/mft_axes.npz'):
    return np.load(infile)


def _generate_bias_pmfs(database, collection, key):
    '''Returns the probability mass functions for each term in each document w.r.t.
    each bias class.
    '''
    client = pymongo.MongoClient()
    db = client[database]
    coll = db[collection]
    bias_classes = coll.distinct(key)
    term_bias_pmfs = {k:dict() for k in bias_classes}
    total_counts = {k:0 for k in bias_classes}
    for doc in coll.find():
        try:
            bias = doc['bias']
        except KeyError:
            print(doc)
            sys.exit(1)
        word_data = doc['word-data']
        for word, attribs in word_data.items():
            count = attribs['count']
            if word not in term_bias_pmfs[bias]:
                term_bias_pmfs[bias][word] = count
            else:
                term_bias_pmfs[bias][word] += count
            total_counts[bias] += count
    for bias, pmf in term_bias_pmfs.items():
        for word in pmf.keys():
            if total_counts[bias] == 0:
                term_bias_pmfs[bias][word] = 0
            else:
                term_bias_pmfs[bias][word] /= total_counts[bias]
    return term_bias_pmfs


def _generate_doc_pmfs(doc):
    n = sum([doc['word-data'][w]['count'] for w in doc['word-data'].keys()])
    doc_pmfs = {w:doc['word-data'][w]['count']/n for w in doc['word-data'].keys()}
    return doc_pmfs


def _generate_global_pmfs(database, collection):
    client = pymongo.MongoClient()
    db = client[database]
    coll = db[collection]
    pmfs = dict()
    n = 0
    for doc in coll.find():
        for word, attribs in doc['word-data'].items():
            count = attribs['count']
            if word not in pmfs: pmfs[word] = count
            else: pmfs[word] += count
            n += count
    for word in pmfs.keys():
        if n == 0:
            pmfs[word] = 0
        else:
            pmfs[word] /= n
    return pmfs


def generate_semantic_axes(infile='data/antonyms.txt', outfile='data/mft_axes.npz', shape=100):
    model = load_kv_model()
    mft_antonyms = dict()
    try:
        with open(infile) as f:
            category = ''
            for line in f:
                if line.startswith('%'):
                    category = line.strip('%').strip('\r\n').strip('\n')
                    mft_antonyms[category] = set()
                else:
                    vice, virtue = line.strip('\r\n').strip('\n').split(':')
                    mft_antonyms[category].add((vice, virtue))
    except IOError as e:
        print(e)
        sys.exit(1)
    for category, pair_set in mft_antonyms.items():
        nvectors = 0
        avg_vector = np.zeros(shape)
        for pair in pair_set:
            vice, virtue = pair[0], pair[1]
            try:
                vice_vector = model.get_vector(vice)
            except KeyError:
                vice_vector = np.zeros(shape)
            try:
                virtue_vector = model.get_vector(virtue)
            except KeyError:
                vice_vector = np.zeros(shape)
            avg_vector += virtue_vector - vice_vector
            nvectors += 1
        if nvectors == 0:
            avg_vector = np.zeros(shape=avg_vector.shape)
        else:
            avg_vector /= nvectors
        mft_antonyms[category] = avg_vector
    np.savez(
        outfile, 
        Harm=mft_antonyms['Harm'], 
        Fairness=mft_antonyms['Fairness'], 
        Ingroup=mft_antonyms['Ingroup'], 
        Authority=mft_antonyms['Authority'], 
        Purity=mft_antonyms['Purity'], 
        General=mft_antonyms['General']
    )


def _generate_mft_features(model, mft_axes, word_data):
    mft_features = {k:{'framing-bias':0, 'intensity':0} for k, _ in mft_axes.items()}
    total_count = 0
    for category in mft_axes.keys():
        for word, attribs in word_data.items():
            if word not in model.wv: continue
            count = attribs['count']
            mft_features[category]['framing-bias'] += (
                count * cos_sim(mft_axes[category], model.wv.get_vector(word))
            )
            total_count += count
        if total_count == 0:
            mft_features[category]['framing-bias'] = 0
        else:
            mft_features[category]['framing-bias'] /= total_count
        for word, attribs in word_data.items():
            if word not in model.wv: continue
            count = attribs['count']
            mft_features[category]['intensity'] += (
                count * ((
                    cos_sim(mft_axes[category], model.wv.get_vector(word)) - 
                    mft_features[category]['framing-bias']
                ) ** 2)
            )
        if total_count == 0:
            mft_features[category]['intensity'] = 0
        else:
            mft_features[category]['intensity'] /= total_count
    return mft_features


def _generate_it_features(doc, global_pmfs, term_bias_pmfs, word_data, topn):
    bias_entropy = dict()
    for word in word_data.keys():
        bias_list = []
        for bias in term_bias_pmfs.keys():
            bias_list.append(0 if word not in term_bias_pmfs[bias] else term_bias_pmfs[bias][word])
        pk = np.array(bias_list)
        bias_entropy[word] = scipy.stats.entropy(pk)
    # extract the top 10 smallest entropy words
    ref_vocab = dict(heapq.nsmallest(topn, bias_entropy.items(), key=itemgetter(1)))
    doc_pmfs = _generate_doc_pmfs(doc)
    v1 = np.array([p for w, p in global_pmfs.items() if w in ref_vocab])
    v2 = np.array([p for w, p in doc_pmfs.items() if w in ref_vocab])
    it_features = {
        'cosine-distance': cos_dist(v1, v2),
        'jaccard-distance': jacc_dist(v1, v2),
        'jensen-shannon-divergence': js_divergence(v1, v2)
    }
    return it_features


def _generate_sa_features(doc):
    analyzer = SentimentIntensityAnalyzer()
    sentence_scores = []
    for sent in nltk.sent_tokenize(doc['content'].lower()):
        sentence_scores.append(analyzer.polarity_scores(sent))
    sentence_arrays = dict()
    for key in ['compound', 'neg', 'neu', 'pos']:
        sentence_arrays[key] = []
        for sentence_dict in sentence_scores:
            sentence_arrays[key].append(sentence_dict[key])
        sentence_arrays[key] = np.array([sentence_arrays[key]])
    sa_features = {
        'compound-sentence-valence': dict(),
        'negative-sentence-valence': dict(),
        'neutral-sentence-valence': dict(),
        'positive-sentence-valence': dict()
    }
    for tup in [
        ('compound-sentence-valence', 'compound'), 
        ('negative-sentence-valence', 'neg'), 
        ('neutral-sentence-valence', 'neu'), 
        ('positive-sentence-valence', 'pos')
    ]:
        key, cat = tup[0], tup[1]
        sa_features[key]['min'] = sentence_arrays[cat].min() if sentence_arrays[cat].size>0 else math.nan
        sa_features[key]['max'] = sentence_arrays[cat].max() if sentence_arrays[cat].size>0 else math.nan
        sa_features[key]['median'] = np.median(sentence_arrays[cat]) if sentence_arrays[cat].size>0 else math.nan
        sa_features[key]['mean'] = sentence_arrays[cat].mean() if sentence_arrays[cat].size>0 else math.nan
        sa_features[key]['std'] = sentence_arrays[cat].std() if sentence_arrays[cat].size>0 else math.nan
        sa_features[key]['var'] = pow(sentence_arrays[cat].std(), 2)  if sentence_arrays[cat].size>0 else math.nan
    return sa_features


def do_work(database, collection, start, ndocs, batch_size=100, topn=25):
    model = Word2Vec.load('data/embeddings.kv')
    with open('data/keys.txt', mode='w') as f:
        for word in model.wv.key_to_index.keys():
            f.write(f'{word}\n')
    mft_axes = load_semantic_axes()
    term_bias_pmfs = _generate_bias_pmfs(database, collection, 'bias')
    global_pmfs = _generate_global_pmfs(database, collection)
    batch = []
    client = pymongo.MongoClient()
    db = client[database]
    coll = db[collection]
    for i in range(ndocs):
        rowid = start + i
        doc = coll.find_one({'rowid': rowid})
        if not doc: continue
        articleid = doc['articleid']
        word_data = doc['word-data']
        mft_features = _generate_mft_features(model, mft_axes, word_data)
        it_features = _generate_it_features(doc, global_pmfs, term_bias_pmfs, word_data, topn)
        sa_features = _generate_sa_features(doc)
        feature_doc = {
            'articleid': articleid,
            'mft_features': mft_features,
            'it_features': it_features,
            'sa_features': sa_features,
            'bias_label': doc['bias']
        }
        batch.append(feature_doc)
        if len(batch) == batch_size:
            db['features'].insert_many(batch)
            batch.clear()
    if batch:
        db['features'].insert_many(batch)


class FeatureGenerator:
    '''Generates features for content from the dataset.'''

    def __init__(self, database='news-bias-detector', collection='article-frames', 
            topn=25, batch_size=25):
        self.database = database
        self.collection = collection
        self.topn = topn
        self.batch_size = batch_size


    def generate(self, threads=10):
        client = pymongo.MongoClient()
        db = client[self.database]
        coll = db[self.collection]
        doc_count = coll.count()
        step_size = doc_count // threads
        for j in range(threads):
            starting_id = j * step_size
            p = Process(target=do_work, args=[self.database, self.collection, starting_id, step_size, self.topn])
            p.start()


class Annotator:
    '''Annotates the dataset for use with feature generation.'''

    def init_word_data(self, content: str):
        counts = local_word_count(content)
        word_data = {x: {'count': count} for x, count in counts.items() if x}
        return word_data


    def annotate(self, fp_content, fp_sources='data/sources.txt'):
        # read in the bias labels
        sources = dict()
        with open(fp_sources, encoding='utf-8') as f:
            for line in f:
                source, bias = line.split(':')
                sources[source] = bias.strip('\n').strip('\r\n')
        self.bias_range = generate_range(list(dict.fromkeys(list(sources.values()))))
        # generate the bias tendencies for each word in the corpus
        word_frame = pd.concat((
            pd.read_csv(
                fp, 
                encoding='utf-8', 
                usecols=['articleid', 'author', 'content', 'publication'],
                dtype={'articleid': np.int32, 'content': 'string', 'author': 'string', 'publication': 'string'}
            ) for fp in fp_content
        ))
        publishers = list(sources.keys())
        word_frame['author'] = word_frame['author'].apply(lambda x: ['missing'] if pd.isnull(x) else [y.strip() for y in x.split('and')])
        word_frame['word_data'] = word_frame['content'].apply(self.init_word_data)
        client = pymongo.MongoClient()
        db = client['news-bias-detector']
        i = 0
        for row in word_frame.itertuples():
            document = dict()
            document['rowid'] = i
            document['articleid'] = int(row.articleid)
            document['authors'] = row.author
            document['publication'] = row.publication
            document['bias'] = sources[row.publication]
            document['content'] = row.content
            document['word-data'] = row.word_data
            document['total-word-count'] = sum([v['count'] for v in document['word-data'].values()])
            db['article-frames'].insert_one(document)
            i += 1

        
class FeatureIO:

    def __init__(self, database='news-bias-detector', collection='features', outfile='data/features.csv'):
        self.database = database
        self.collection = collection


    def normalize(self, df, norm='min-max'):
        headers = []
        headers.extend(extract_2d_headers(df['mft_features'][0], 'mft'))
        headers.extend(extract_1d_headers(df['it_features'][0], 'it'))
        headers.extend(extract_2d_headers(df['sa_features'][0], 'sa'))
        values = []
        for _, row in df.iterrows():
            v = []
            v.extend(flatten_2d_dict(row['mft_features']))
            v.extend(flatten_1d_dict(row['it_features']))
            v.extend(flatten_2d_dict(row['sa_features']))
            values.append(v)
        frame = pd.DataFrame(values, columns=headers)
        if norm == 'min-max':
            frame = (frame-frame.min())/(frame.max()-frame.min())
        else:
            frame = (frame-frame.mean())/frame.std()
        frame['bias-label'] = df['bias_label']
        return frame



    def read_features(self):
        '''Reads features from the collection specfied during the creation of this
        object into a pandas frame and returns it.
        '''
        # connect to the local database
        client = pymongo.MongoClient()
        db = client[self.database]
        coll = db[self.collection]

        # get all of the documents containing the features
        cursor = coll.find()

        # create a dataframe from the documents
        df = pd.DataFrame(list(cursor))

        # remove the id fields
        del df['_id']
        del df['articleid']

        return df


    def write_features(self, df, path='data/features.csv', *args, **kwargs):
        '''Writes DF to the indicated file.
        '''
        try:
            df.to_csv(path, *args, **kwargs)
        except IOError as e:
            print(e)
            sys.exit(1)
