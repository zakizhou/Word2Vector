from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf


def build_corpus(corpus_name):
    pass


def convert_to_records(corpus, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for word, context in corpus:
        example = tf.train.Example(features=tf.train.Features(feature={
            "word": tf.train.Feature(int64_list=tf.train.Int64List(value=[word])),
            "context": tf.train.Feature(int64_list=tf.train.Int64List(value=[word]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_and_encode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(serialized=serialized,
                                       features={
                                           "word": tf.FixedLenFeature([], tf.int64),
                                           "context": tf.FixedLenFeature([], tf.int64)
                                       })
    word = features['word']
    context = features['context']
    return word, context


def input_producer(records_name):
    filename_queue = tf.train.string_input_producer([records_name])
    word, context = read_and_encode(filename_queue=filename_queue)
    words, contexts = tf.train.shuffle_batch([word, context],
                                             batch_size=128,
                                             capacity=50000,
                                             min_after_dequeue=60000)
    return words, contexts


class Config(object):
    def __init__(self):
        self.embedding_size = 256
        self.learning_rate = 4e-4


class Inputs(object):
    def __init__(self):
        self.examples = None
        self.labels = None
        self.vocab_size = 50000
        self.num_sampled = 10
        self.num_true = 1

