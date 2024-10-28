"""
Utilities for scoring negative samples

For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

"""

import tensorflow as tf


def calculate_negative_logits(seq, neg_emb, args):
    """
    Calculate scores for negative samples
    :param seq: Tensor of causal, sequence embeddings
    :param neg_emb: Tensor of embeddings of negative items
    :param args: CLI arguments for model
    :return: tf.Tensor of scores of negative items
    """
    if args.neg_sampler_type == "balanced":
        # There's a set of negatives for each positive
        seq_emb = tf.expand_dims(seq, axis=1)
        neg_logits = tf.reduce_sum(tf.multiply(neg_emb, seq_emb), axis=-1)
        return neg_logits
    elif args.neg_sampler_type == "adaptive":
        # Expand dims to [batch_size, 1, max_seq_len, hidden_dims]
        seq_emb = tf.expand_dims(seq, axis=1)
        # Expand dims to [batch_size, max_neg_samples, 1, hidden_dims]
        neg_emb = tf.expand_dims(neg_emb, axis=2)
        # Multiply to get [batch_size, max_neg_samples, max_seq_len]
        # Each row is score of each negative, for each sequence item
        neg_logits = tf.reduce_sum(tf.multiply(neg_emb, seq_emb), axis=-1)
        # Keep only the top k values of logits from each row
        neg_logits = adaptive_top_k(
            tf.transpose(neg_logits, perm=[0, 2, 1]), k=args.num_adaptive_samples
        )
        neg_logits = tf.transpose(neg_logits, perm=[0, 2, 1])
        return neg_logits
    else:
        neg_logits = tf.matmul(
            neg_emb, tf.transpose(seq, perm=[0, 2, 1]), transpose_b=False
        )
        return neg_logits


def adaptive_top_k(logits, k=10):
    """
    Getting the top_k scores from logits tensor
    :param logits: Scores for negative items
    :param k: Top k scores to be selected
    :return: tf.Tensor of shape `logits` with top k scores
    """
    values, indices = tf.nn.top_k(logits, k, sorted=False)
    logits_shape = tf.shape(logits)
    dims = [
        tf.range(logits_shape[i]) for i in range(logits_shape.shape.num_elements() - 1)
    ]
    grid = tf.meshgrid(*dims, tf.range(k), indexing="ij")
    scatter_idx = tf.stack(grid[:-1] + [indices], axis=-1)
    return tf.scatter_nd(scatter_idx, values, logits_shape)


def create_sampling_placeholders(args):
    """
    Create placeholder for negative samples, based on method
    :param args: CLI parameters for
    :return:
    """
    if args.neg_sampler_type in ["mixed", "adaptive"]:
        # Adding space for additional negatives
        return tf.compat.v1.placeholder(
            tf.int32, shape=(None, args.num_neg_samples + args.num_mixed_samples)
            , name='neg_sample_placeholder')
    elif args.neg_sampler_type == "balanced":
        # Adding space for negatives for each positive
        return tf.compat.v1.placeholder(
            tf.int32, shape=(None, None, args.num_neg_samples)
            , name='neg_sample_placeholder')
    else:
        return tf.compat.v1.placeholder(tf.int32, shape=(None, args.num_neg_samples), name='neg_sample_placeholder')