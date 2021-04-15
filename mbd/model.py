import argparse

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from sklearn.metrics import (
    f1_score, precision_score, recall_score
) 

from tools import clean_word

from pymongo import MongoClient
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec, KeyedVectors
from torch.autograd import Variable


class EmbeddingModel:
    '''Implements a model for training word embeddings using the Word2Vec implemenetation
    from the gensim package.
    '''

    def __init__(self, database, collection):
        self.database = database
        self.collection = collection


    def __iter__(self):
        client = MongoClient()
        db = client[self.database]
        for doc in db[self.collection].find():
            for sent in nltk.sent_tokenize(doc['content'].lower()):
                yield [clean_word(w) for w in sent.split() if clean_word(w)]


    def save(self, filepath):
        '''Saves the learned model to the file indicated by FILEPATH.

        Args:
            filepath (str): The path to the file to save the model to.
        '''
        self.model.save(filepath)


    def train(self, vector_size=300, epochs=100):
        '''Trains a Word2Vec model on the MongoDB corpus pointed to by this object.

        Args:
            vector_size (int): The intended number of dimensions for each vector (default: 300).
            epochs (int): The number of training rounds (default: 100).
        '''
        self.model = Word2Vec(vector_size=vector_size)
        self.model.build_vocab(self)
        self.model.train(self, epochs=epochs, total_words=self.model.corpus_count)


def extract_glove_model(infile='data/glove.6B.100d.txt', outfile='data/glove.6B.100d.kv'):
    '''Intermediary step to convert GLoVE text file to format that gensim can work with.

    Args:
        infile (str): The relative or absolute path to the input glove txt file.
        outfile (str): The relative or absolute path to the output glove kv file.
    '''
    glove2word2vec(infile, outfile)


def load_kv_model(infile='data/glove.6B.100d.kv'):
    '''Loads a kv model in KeyedVector format. This is not a full model in that it is 
    lacking a binary tree as well as several other key features. The object returned 
    by this function is mostly used for comparison and distance function purposes as well 
    as fast lookups.

    Args:
        infile (str): The relative or absolute path to the kv model.

    Returns:
        (gensim.models.KeyedVectors): A model represented as a KeyedVector.
    '''
    return KeyedVectors.load_word2vec_format(infile)


def sample(df, beta=0.8, seed=0):
    train_rows = np.random.choice(df.index.values, int(len(df.index)*(beta)), replace=False)
    test_rows = np.random.choice(df.index.values, int(len(df.index)*(1-beta)), replace=False)
    train_set = df.loc[train_rows]
    test_set = df.loc[test_rows]
    train_x, train_y = train_set.loc[:, train_set.columns!='bias-label'], train_set.iloc[:,-1:]
    test_x, test_y = test_set.loc[:, test_set.columns!='bias-label'], test_set.iloc[:,-1:]
    return train_x, train_y, test_x, test_y


class EnsembleClassificationModel:

    def __init__(self, infile='data/features.csv'):
        self.infile = infile

    
    def run_classification(self, ab_params={}, gb_params={}, rf_params={}):
        from sklearn.ensemble import (
            AdaBoostClassifier,
            GradientBoostingClassifier,
            ExtraTreesClassifier
        )

        df = pd.read_csv(self.infile)
        df = df.dropna()

        df.reset_index(drop=True)

        train_x, train_y, test_x, test_y = sample(df)

        classifiers = {
            'AdaBoost': AdaBoostClassifier(**ab_params), 
            'Gradient Boosting': GradientBoostingClassifier(**gb_params), 
            'Extremely Random Trees': ExtraTreesClassifier(**rf_params)
        }
        
        test_y_ravel = np.ravel(test_y,order='C')

        for k, v in classifiers.items():
            print(f'Running {k} classifier...')
            v.fit(train_x, np.ravel(train_y,order='C'))
            y_out = v.predict(test_x)
            score = v.score(test_x, test_y_ravel)
            f1 = f1_score(test_y_ravel, y_out, average='micro')
            precision = precision_score(test_y_ravel, y_out, average='micro')
            recall = recall_score(test_y_ravel, y_out, average='micro')
            print(f'Classifier: {k}')
            print(f'Accuracy: {score}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}')
            print()


class NNClassificationModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super(NNClassificationModel, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(in_features=39, out_features=300)
        self.hidden = nn.Linear(in_features=300, out_features=6)
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.hidden.weight, gain=1.0)

    def forward(self, x):
        '''Performs a forward pass through the model using the input tensors, 
        X, Y, and Z.

        Returns:
            (torch.Tensor): A 1d tensor containing class probabilities for the classes 
            defined by the model.
        '''
        x1 = self.linear(x)
        x2 = F.relu(x1)
        e = self.hidden(x2)
        return e


# adapted from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class FeatureDataset(torch.utils.data.Dataset):

    def __init__(self, features, transform=None):
        self.features = features
        self.transform = transform
        # get the features for each categories from the frame
        self.x_features = [c for c in self.features.columns if c.startswith('mft-') or c.startswith('it-') or c.startswith('sa-')]
        # convert categorical to numeric for tensorflow
        self.features['bias-label'] = self.features["bias-label"].astype("category")
        self.features['bias-label'] = self.features['bias-label'].cat.codes


    def __len__(self):
        return len(self.features)


    def __getitem__(self, idx):
        # get the row specifed by idx
        row = self.features.iloc[idx]

        feature_list = [row[f] for f in self.x_features]

        # create the tensors for the x features
        x = nn.Parameter(torch.Tensor(feature_list))

        # create the tensor for the label data
        label = torch.Tensor([row['bias-label']])

        return x, label

    
def train_model(use_cuda, model, epochs, loader, criterion=nn.CrossEntropyLoss(), clip_value=1):
    if use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    
    if use_cuda:
        model.cuda()
    else:
        model.cpu()

    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    
    model.train()
    for epoch in range(epochs):
        correct = 0
        running_loss = 0
        running_total = 0
        print("Epoch {}/{}".format(epoch+1, epochs))
        print("-" * 10)

        for x, labels in loader:
            x = x.to(device)
            if use_cuda:
                labels = torch.flatten(labels).type(torch.cuda.LongTensor)
            else:
                labels = torch.flatten(labels).type(torch.LongTensor)
            labels = labels.to(device)

            outputs = model(x)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            reverted = torch.argmax(outputs, dim=1)

            running_loss += loss
            correct += (reverted == labels).float().sum()
            running_total += torch.numel(reverted)

        accuracy = 100 * (correct / running_total)
        print('Loss:', running_loss.item())
        print('Accuracy %:', accuracy.item())


def test_model(use_cuda, model, loader, criterion=nn.CrossEntropyLoss()):
    if use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    correct = 0
    running_scores = 0
    running_loss = 0
    running_total = 0
    running_f1 = 0
    running_precision = 0
    running_recall = 0

    with torch.set_grad_enabled(False):
        for x, labels in loader:
            x = x.to(device)
            if use_cuda:
                labels = torch.flatten(labels).type(torch.cuda.LongTensor)
            else:
                labels = torch.flatten(labels).type(torch.LongTensor)

            outputs = model(x)

            loss = criterion(outputs, labels)
            reverted = torch.argmax(outputs, dim=1)

            running_loss += loss
            correct += (reverted == labels).float().sum()
            running_total += torch.numel(reverted)
            running_scores += 1
            nlabels = labels.cpu().numpy()
            nreverted = reverted.cpu().numpy()
            running_f1 += f1_score(nlabels, nreverted, average='micro')
            running_precision += precision_score(nlabels, nreverted, average='micro')
            running_recall += recall_score(nlabels, nreverted, average='micro')

        accuracy = 100 * (correct / running_total)
        print('Loss:', running_loss.item())
        print('Accuracy %:', accuracy.item())
        print('Average F1 Score:', running_f1 / running_scores)
        print('Average Precision:', running_precision / running_scores)
        print('Average Recall:', running_recall / running_scores)


def save_model(model, path='data/pytorch_model.mod'):
    try:
        torch.save(model.state_dict(), path)
    except IOError as e:
        print(e)
        sys.exit(1)


def ensemble_runner(*args, **kwargs):
    model = EnsembleClassificationModel(kwargs['filepath'])
    model.run_classification(ab_params=kwargs['ab_params'], gb_params=kwargs['gb_params'], rf_params=kwargs['rf_params'])


def dnn_runner(**kwargs):
    try:
        df = pd.read_csv(kwargs['filepath'])
    except IOError as e:
        print(e)
        sys.exit(1)
    df.dropna(inplace=True)
    train_x, train_y, test_x, test_y = sample(df)
    
    train_set = train_x
    train_set['bias-label'] = train_y['bias-label']
    train_set.reset_index(drop=True, inplace=True)
    train_weights = train_set.value_counts() / len(train_set)
    train_loader = torch.utils.data.DataLoader(
        dataset=FeatureDataset(train_set),
        batch_size=kwargs['batch_size'],
        sampler=torch.utils.data.WeightedRandomSampler(
            weights=train_weights.tolist(),
            num_samples=len(train_set)
        )
    )
    test_set = test_x
    test_set['bias-label'] = test_y['bias-label']
    test_set.reset_index(drop=True, inplace=True)
    
    test_weights = test_set.value_counts() / len(test_set)
    test_loader = torch.utils.data.DataLoader(
        dataset=FeatureDataset(test_set),
        batch_size=args.batch_size,
        sampler=torch.utils.data.WeightedRandomSampler(
            weights=test_weights.tolist(),
            num_samples=len(test_set)
        )
    )
    model = NNClassificationModel()
    print('Now Training...')
    train_model(kwargs['use_cuda'], model, kwargs['epochs'], train_loader)
    print('Now Testing...')
    test_model(kwargs['use_cuda'], model, test_loader)
    save_model(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runner', type=str, default='ensemble', help='One of {\'ensemble\', \'nn\'} (default:\'ensemble\')')
    parser.add_argument('--filepath', type=str, default='data/features.csv', help='The path to the input features file (default: data/features.csv).')
    parser.add_argument('--use-cuda', dest='use_cuda', type=bool, default=False,
        help='Whether to use CUDA acceleration. You must have an appropriate version of PyTorch installed as well as the correct version of CUDA to match it. You must also have a CUDA enabled (Nvidia) GPU.'
    )
    parser.add_argument('--epochs', dest='epochs', type=int, default=100,
        help='The number of training iterations.'
    )
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
        help='The number of instances to train/validate with in a single batch.'
    )
    parser.add_argument('--ab-n-estimators', dest='ab_n_estimators', type=int, default=50, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html'
    )
    parser.add_argument('--ab-learning-rate', dest='ab_learning_rate', type=float, default=1, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html'
    )
    parser.add_argument('--ab-algorithm', dest='ab_algorithm', type=str, default='SAMME.R', 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html'
    )
    parser.add_argument('--ab-random-state', dest='ab_random_state', type=int, default=None, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html'
    )
    parser.add_argument('--gb-loss', dest='gb_loss', type=str, default='deviance', 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-learning-rate', dest='gb_learning_rate', type=float, default=0.1, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-n-estimators', dest='gb_n_estimators', type=int, default=100, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-subsample', dest='gb_subsample', type=float, default=1.0, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-criterion', dest='gb_criterion', type=str, default='friedman_mse', 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-min-samples-split', dest='gb_min_samples_split', type=int, default=2, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-min-samples-leaf', dest='gb_min_samples_leaf', type=int, default=1, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-min-weight-fraction-leaf', dest='gb_min_weight_fraction_leaf', type=float, default=0.0, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-max-depth', dest='gb_max_depth', type=int, default=3,
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-min-impurity-decrease', dest='gb_min_impurity_decrease', type=float, default=0.0, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-random-state', dest='gb_random_state', type=int, default=None, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-max-leaf-nodes', dest='gb_max_leaf_nodes', type=int, default=None, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-warm-start', dest='gb_warm_start', type=bool, default=False, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-validation-fraction', dest='gb_validation_fraction', type=float, default=0.1, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-n-iter-no-change', dest='gb_n_iter_no_change', type=int, default=None, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-tolerance', dest='gb_tolerance', type=float, default=1e-4, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--gb-ccp-alpha', dest='gb_ccp_alpha', type=float, default=0.0, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
    )
    parser.add_argument('--ert-n-estimators', dest='ert_n_estimators', type=int, default=100, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-criterion', dest='ert_criterion', type=str, default='gini', 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-max-depth', dest='ert_max_depth', type=int, default=None, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-min-samples-split', dest='ert_min_samples_split', type=int, default=2, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-min-samples-leaf', dest='ert_min_samples_leaf', type=int, default=1, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-min-weight-fraction-leaf', dest='ert_min_weight_fraction_leaf', type=float, default=0.0, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-max-leaf-nodes', dest='ert_max_leaf_nodes', type=int, default=None, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-min-impurity-decrease', dest='ert_min_impurity_decrease', type=float, default=0.0, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-bootstrap', dest='ert_bootstrap', type=bool, default=False, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-oob-score', dest='ert_oob_score', type=bool, default=False, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-n-jobs', dest='ert_n_jobs', type=int, default=None, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-random-state', dest='ert_random_state', type=int, default=None, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-warm-start', dest='ert_warm_start', type=bool, default=False, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-ccp-alpha', dest='ert_ccp_alpha', type=float, default=0.0, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    parser.add_argument('--ert-max-samples', dest='ert_max_samples', type=float, default=None, 
        help='Please see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
    )
    args = parser.parse_args()

    if args.runner == 'ensemble':
        ab_params = {
            'n_estimators': args.ab_n_estimators,
            'learning_rate': args.ab_learning_rate,
            'algorithm': args.ab_algorithm,
            'random_state': args.ab_random_state
        }
        gb_params = {
            'loss': args.gb_loss,
            'learning_rate': args.gb_learning_rate,
            'n_estimators': args.gb_n_estimators,
            'subsample': args.gb_subsample,
            'criterion': args.gb_criterion,
            'min_samples_split': args.gb_min_samples_split,
            'min_samples_leaf': args.gb_min_samples_leaf,
            'min_weight_fraction_leaf': args.gb_min_weight_fraction_leaf,
            'max_depth': args.gb_max_depth,
            'min_impurity_decrease': args.gb_min_impurity_decrease,
            'random_state': args.gb_random_state,
            'max_leaf_nodes': args.gb_max_leaf_nodes,
            'warm_start': args.gb_warm_start,
            'validation_fraction': args.gb_validation_fraction,
            'n_iter_no_change': args.gb_n_iter_no_change,
            'tol': args.gb_tolerance,
            'ccp_alpha': args.gb_ccp_alpha
        }
        rf_params = {
            'n_estimators': args.ert_n_estimators,
            'criterion': args.ert_criterion,
            'max_depth': args.ert_max_depth,
            'min_samples_split': args.ert_min_samples_split,
            'min_samples_leaf': args.ert_min_samples_leaf,
            'min_weight_fraction_leaf': args.ert_min_weight_fraction_leaf,
            'max_leaf_nodes': args.ert_max_leaf_nodes,
            'min_impurity_decrease': args.ert_min_impurity_decrease,
            'bootstrap': args.ert_bootstrap,
            'oob_score': args.ert_oob_score,
            'n_jobs': args.ert_n_jobs,
            'random_state': args.ert_random_state,
            'warm_start': args.ert_warm_start,
            'ccp_alpha': args.ert_ccp_alpha,
            'max_samples': args.ert_max_samples
        }
        kwargs={'filepath': args.filepath, 'ab_params':ab_params, 'gb_params':gb_params, 'rf_params':rf_params}
        ensemble_runner(**kwargs)
    elif args.runner == 'nn':
        kwargs = {
            'filepath': args.filepath,
            'use_cuda': args.use_cuda,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        }
        dnn_runner(**kwargs)