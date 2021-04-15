import argparse
import os, sys

from annotator import (
    Annotator, FeatureGenerator, generate_semantic_axes,
    FeatureIO
)
from model import EmbeddingModel, FeatureDataset, extract_glove_model


def extract_embeddings():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('database', type=str)
    parser.add_argument('collection', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('dimensions', type=int, default=100)
    parser.add_argument('epochs', type=int, default=10)
    args = parser.parse_args()
    model = EmbeddingModel(args.database, args.collection)
    model.train(args.dimensions, args.epochs)
    model.save(f'data/{args.outfile}')


def run_annotator():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source-files', dest='source_files', type=str, nargs='+', help='')
    args = parser.parse_args()
    anno = Annotator()
    anno.annotate(args.source_files)


if __name__ == '__main__':
    dataset = FeatureDataset()
    #for x, y, z, label in dataset:
    #    print('X:', x)
    #    print('Y:', y)
    #    print('Z:', z)
    #    print('Label', label)