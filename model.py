"""
Model structure, including modified negative sampling loss

For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

"""
from modules import *
from util_neg_loss import *
import tensorflow_ranking as tfr


class Model:
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=(), name='is_training_placeholder')
        self.u = tf.compat.v1.placeholder(tf.int32, shape=(None), name='user_placeholder')
        self.input_seq = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen), name='input_seq_placeholder')
        self.pos = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen), name='positive_item_placeholder')

        # Creating placeholders for negative samples, based on method
        self.neg = create_sampling_placeholders(args)

        self.loss_fn = tfr.keras.losses.SigmoidCrossEntropyLoss(
            reduction=tf.losses.Reduction.AUTO
        )

        pos = self.pos
        neg = self.neg
        seq_mask = tf.expand_dims(
            tf.cast(tf.not_equal(self.input_seq, 0), dtype=tf.float32), -1
        )

        # Get causal, sequential embeddings for all items in user sequence
        with tf.compat.v1.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(
                self.input_seq,
                vocab_size=itemnum + 1,
                num_units=args.hidden_units,
                zero_pad=True,
                scale=True,
                l2_reg=args.l2_emb,
                scope="input_embeddings",
                with_t=True,
                reuse=reuse,
            )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(
                    tf.expand_dims(tf.range(tf.shape(input=self.input_seq)[1]), 0),
                    [tf.shape(input=self.input_seq)[0], 1],
                ),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True,
            )
            self.seq += t

            # Dropout
            self.seq = tf.compat.v1.layers.dropout(
                self.seq,
                rate=args.dropout_rate,
                training=tf.convert_to_tensor(value=self.is_training),
            )
            self.seq *= seq_mask

            # Build blocks

            for i in range(args.num_blocks):
                with tf.compat.v1.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(
                        queries=normalize(self.seq),
                        keys=self.seq,
                        num_units=args.hidden_units,
                        num_heads=args.num_heads,
                        dropout_rate=args.dropout_rate,
                        is_training=self.is_training,
                        causality=True,
                        scope="self_attention",
                    )

                    # Feed forward
                    self.seq = feedforward(
                        normalize(self.seq),
                        num_units=[args.hidden_units, args.hidden_units],
                        dropout_rate=args.dropout_rate,
                        is_training=self.is_training,
                    )
                    self.seq *= seq_mask

            self.seq = normalize(self.seq)
        # Get the embeddings of the positive/negative items from embedding layer
        pos_emb = tf.nn.embedding_lookup(params=item_emb_table, ids=pos)
        neg_emb = tf.nn.embedding_lookup(params=item_emb_table, ids=neg)
        seq_emb = tf.reshape(
            self.seq,
            [tf.shape(input=self.input_seq)[0] * args.maxlen, args.hidden_units],
        )

        # Store embeddings for all items for evaluation at current epoch
        self.test_item = tf.compat.v1.placeholder(tf.int32, shape=(itemnum))
        self.test_item_emb = tf.nn.embedding_lookup(
            params=item_emb_table, ids=self.test_item
        )
        self.test_logits = tf.matmul(seq_emb, tf.transpose(a=self.test_item_emb))
        self.test_logits = tf.reshape(
            self.test_logits,
            [tf.shape(input=self.input_seq)[0], args.maxlen, itemnum],
        )
        self.test_logits = self.test_logits[:, -1, :]

        # Score for positive items
        self.pos_logits = tf.reduce_sum(
            tf.reshape(
                pos_emb,
                [tf.shape(input=self.input_seq)[0] * args.maxlen, args.hidden_units],
            )
            * seq_emb,
            axis=-1,
            )
        # Score for negative items
        self.neg_logits = calculate_negative_logits(self.seq, neg_emb, args)

        # Labels for positive items
        self.is_pos_target = tf.reshape(
            tf.cast(tf.not_equal(pos, 0), dtype=tf.float32),
            [tf.shape(input=self.input_seq)[0] * args.maxlen],
        )
        # Labels for negative items
        self.is_neg_target = tf.cast(tf.zeros_like(self.neg_logits), dtype=tf.float32)

        # Calculate BCE loss in two parts for positive and negative items
        self.loss = self.loss_fn(self.is_pos_target, self.pos_logits) + self.loss_fn(
            self.is_neg_target, self.neg_logits
        )

        # Add regularization losses
        reg_losses = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
        )
        self.loss += sum(reg_losses)

        if reuse is None:
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=args.lr, beta2=0.98
            )
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=self.global_step
            )

        self.merged = tf.compat.v1.summary.merge_all()

    def predict(self, sess, u, seq, item_idx):
        return sess.run(
            self.test_logits,
            {
                self.u: u,
                self.input_seq: seq,
                self.test_item: item_idx,
                self.is_training: False,
            },
        )