from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import tensorflow as tf


class Word2Vec(object):
    def __init__(self, config, inputs):
        embedding_size = config.embedding_size
        vocab_size = inputs.vocab_size
        num_sampled = inputs.num_sampled
        num_true = inputs.num_true
        learning_rate = config.learning_rate
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable("embedding",
                                        shape=[vocab_size, embedding_size])
            embed = tf.nn.embedding_lookup(embedding, inputs.examples)
        with tf.variable_scope("nce_loss"):
            nce_weights = tf.get_variable("nce_weights",
                                          shape=[vocab_size, embedding_size])
            nce_bias = tf.get_variable("nce_bias",
                                       shape=[vocab_size])
            nce_loss_per_example = tf.nn.nce_loss(weights=nce_weights,
                                                  biases=nce_bias,
                                                  inputs=embed,
                                                  labels=inputs.labels,
                                                  num_sampled=num_sampled,
                                                  num_classes=vocab_size,
                                                  num_true=num_true)
            self.__loss = tf.reduce_mean(nce_loss_per_example)
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.__train_op = optimizer.minimize(self.__loss)

    @property
    def loss(self):
        return self.__loss

    @property
    def train_op(self):
        return self.__train_op


class Word2Vector(object):
    def __init__(self, config, inputs):
        vocab_size = inputs.vocab_size
        embedding_size = config.embedding_size
        learning_rate = config.learning_rate
        unigrams = inputs.unigrams
        num_sampled = config.num_sampled
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable(name="embedding",
                                        shape=[vocab_size, embedding_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.05),
                                        dtype=tf.float32)
            # embed: [batch_size, embedding_size]
            embed = tf.nn.embedding_lookup(embedding, inputs.inputs)

        with tf.variable_scope("output"):
            """
            I use fixed_unigram_candidate_sampler because not all the classes follow the zipfian distribution
            and the whole part code is equivalent of
            ```
            self.__loss = tf.nn.nce_loss(projection,
                                         bias,
                                         inputs.inputs,
                                         inputs.labels
                                         sampled_values=sampled)
            ```
            """
            projection = tf.get_variable(name="project",
                                         shape=[vocab_size, embedding_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.05),
                                         dtype=tf.float32)
            bias = tf.get_variable(name="bias",
                                   shape=[vocab_size],
                                   initializer=tf.constant_initializer(value=0.),
                                   dtype=tf.float32)
            # projection_labels: [batch_size, embedding_size]
            projection_labels = tf.nn.embedding_lookup(projection, inputs.labels)
            projection_bias_labels = tf.nn.embedding_lookup(bias, inputs.labels)
            # logits: [batch_size]
            logits = tf.reduce_sum(tf.mul(embed, projection_labels), 1) + projection_bias_labels
            # sampled: [num_sampled]
            sampled = tf.nn.fixed_unigram_candidate_sampler(true_classes=inputs.labels,
                                                            num_true=1,
                                                            num_sampled=num_sampled,
                                                            unique=True,
                                                            range_max=vocab_size,
                                                            distortion=0.75,
                                                            unigrams=unigrams)
            # projection_sampled: [num_sampled, embedding_size]
            projection_sampled = tf.nn.embedding_lookup(projection, sampled[0])
            # project_bias_sampled: [num_sampled]
            projection_bias_sampled = tf.nn.embedding_lookup(bias, sampled[0])
            # [batch_size, num_sampled] = [batch_size, embedding_size] * [num_sampled, embedding_size]^T + [num_sampled]
            sampled_logits = tf.matmul(embed, projection_sampled, transpose_b=True) + projection_bias_sampled

        with tf.name_scope("nce_loss"):
            original_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                    targets=tf.ones_like(logits))
            sampled_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=sampled_logits,
                                                                   targets=tf.zeros_like(sampled_logits))
            self.__loss = tf.reduce_mean(tf.reduce_sum(original_loss) + tf.reduce_sum(sampled_loss))

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.__train_op = optimizer.minimize(self.__loss)

    @property
    def loss(self):
        return self.__loss

    @property
    def train_op(self):
        return self.train_op


