# -*- coding: utf-8 -*-
# class

from multiprocessing import Process, Queue
import tensorflow as tf
import numpy as np
import copy
import random


# import tensorflow_addons as tfa

class WarpSampler(object):
    def __init__(self, User, Beh, Beh_w, Behaviors, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        from Recommendation_Beibei_Method import sample_function
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      Beh,
                                                      Beh_w,
                                                      Behaviors,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


class Args:
    dataset = 'Yelp2'
    train_dir = True
    batch_size = 128
    lr = 0.0007
    hidden_units = 50
    maxlen = 170
    num_epochs = 1000
    num_heads = 1
    dropout_rate = 0.4
    l2_emb = 0.0
    l2_emb_class = 0.0
    use_res = True
    num_blocks = 1
    min_seq = 20
    device = 'cuda'


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        from Recommendation_Beibei_Method import embedding
        from Recommendation_Beibei_Method import Model_pro

        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())

        self.epoch = tf.compat.v1.placeholder(tf.int32, shape=())

        self.u = tf.compat.v1.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        self.buy_seq_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))
        self.cart_seq_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))
        self.fav_seq_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))
        self.click_seq_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))

        self.Aug_input_seq = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        self.Aug_seq_cxt = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen, 4))
        self.Aug_buy_seq_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))
        self.Aug_cart_seq_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))
        self.Aug_fav_seq_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))
        self.Aug_click_seq_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))

        self.pos = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        self.seq_cxt = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen, 4))
        self.pos_cxt = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen, 4))
        self.labels = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen, 4))

        self.pos_weight = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))
        self.neg_weight = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))
        self.recency = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))

        self.resweight_buy = tf.Variable(0.0, trainable=True)
        self.resweight_cart = tf.Variable(0.0, trainable=True)
        self.resweight_fav = tf.Variable(0.0, trainable=True)
        self.resweight_click = tf.Variable(0.0, trainable=True)

        pos = self.pos
        neg = self.neg

        mask = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.input_seq, 0)), -1)
        # sequence embedding, item embedding table
        in_seq, item_emb_table = embedding(self.input_seq,
                                           vocab_size=itemnum + 1,
                                           num_units=args.hidden_units,
                                           zero_pad=True,
                                           scale=True,
                                           l2_reg=args.l2_emb,
                                           scope="input_embeddings",
                                           with_t=True,
                                           reuse=reuse
                                           )

        # print('in seq shape....', tf.shape(in_seq))
        self.seq = Model_pro(in_seq, mask, self.seq_cxt, args, self.buy_seq_mask, self.cart_seq_mask, self.fav_seq_mask,
                             self.click_seq_mask, self.resweight_buy,
                             self.resweight_cart, self.resweight_fav, self.resweight_click, self.is_training,
                             reuse=None)
        # ===================== Augmentation ==========================

        Aug_mask = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.Aug_input_seq, 0)), -1)
        # sequence embedding, item embedding table
        Aug_in_seq = tf.nn.embedding_lookup(item_emb_table, self.Aug_input_seq)

        # print('in Aug_seq shape....', tf.shape(Aug_in_seq))
        # if self.is_training == True:
        self.Aug_seq = Model_pro(Aug_in_seq, Aug_mask, self.Aug_seq_cxt, args, self.Aug_buy_seq_mask,
                                 self.Aug_cart_seq_mask, self.Aug_fav_seq_mask, self.Aug_click_seq_mask,
                                 self.resweight_buy,
                                 self.resweight_cart, self.resweight_fav, self.resweight_click, self.is_training,
                                 reuse=True)

        # ==============================================================
        # self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_weight = tf.reshape(self.pos_weight, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg_weight = tf.reshape(self.neg_weight, [tf.shape(self.input_seq)[0] * args.maxlen])
        recency = tf.reshape(self.recency, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])

        trgt_cxt = tf.reshape(self.pos_cxt, [tf.shape(self.input_seq)[0] * args.maxlen, 4])
        trgt_cxt_emb = tf.compat.v1.layers.dense(inputs=trgt_cxt, units=args.hidden_units, activation=None, reuse=True,
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                 name="cxt_emb")

        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)

        pos_emb = tf.concat([pos_emb, trgt_cxt_emb], -1)
        neg_emb = tf.concat([neg_emb, trgt_cxt_emb], -1)
        # cxt
        pos_emb = tf.compat.v1.layers.dense(inputs=pos_emb, reuse=True, units=args.hidden_units, activation=None,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            name="feat_emb")
        neg_emb = tf.compat.v1.layers.dense(inputs=neg_emb, reuse=True, units=args.hidden_units, activation=None,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            name="feat_emb")

        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])
        Aug_seq_emb = tf.reshape(self.Aug_seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        self.test_item = tf.compat.v1.placeholder(tf.int32, shape=(100))
        self.test_item_cxt = tf.compat.v1.placeholder(tf.float32, shape=(100, 4))
        test_item_cxt_emb = tf.compat.v1.layers.dense(inputs=self.test_item_cxt, units=args.hidden_units,
                                                      activation=None, reuse=True,
                                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                      name="cxt_emb")

        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        test_item_emb = tf.concat([test_item_emb, test_item_cxt_emb], -1)
        test_item_emb = tf.compat.v1.layers.dense(inputs=test_item_emb, reuse=True, units=args.hidden_units,
                                                  activation=None,
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                  name="feat_emb")

        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 100])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        # if self.is_training == True:
        class_logits = tf.concat([pos_emb, seq_emb], -1)
        # *********************************===========    classification layer =============***************************************************
        classification_pred1 = tf.compat.v1.layers.dense(inputs=class_logits, units=args.hidden_units, activation=None,
                                                         reuse=False,
                                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                         kernel_regularizer=tf.keras.regularizers.l2(args.l2_emb_class),
                                                         name="classification1")
        # classification_pred2 = tf.compat.v1.layers.dense(inputs=classification_pred1 , units=args.hidden_units/2 ,activation=None, reuse=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="classification2")
        self.classification_pred = tf.compat.v1.layers.dense(inputs=classification_pred1, units=4, activation=None,
                                                             reuse=False,
                                                             kernel_initializer=tf.random_normal_initializer(
                                                                 stddev=0.01),
                                                             kernel_regularizer=tf.keras.regularizers.l2(
                                                                 args.l2_emb_class), name="classification")

        # print('shape of labels.....', tf.shape(self.labels))
        labels = tf.reshape(self.labels, [tf.shape(self.input_seq)[0] * args.maxlen, 4])
        # *********************************===========    classification layer ==============***************************************************
        # self.new_loss, self.con_mask = contrastive_loss(self.input_seq, seq_emb, Aug_seq_emb)

        istarget = tf.reshape(tf.compat.v1.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.compat.v1.log(tf.sigmoid(self.pos_logits) + 1e-24) * pos_weight * istarget -
            tf.compat.v1.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * neg_weight * istarget
        ) / tf.reduce_sum(istarget)

        labels_pred = tf.nn.softmax_cross_entropy_with_logits(labels, self.classification_pred)
        self.classification_loss = tf.reduce_sum(labels_pred * istarget) / tf.reduce_sum(istarget)

        # self.final_loss = self.loss #+ 0.1*self.new_loss

        self.final_loss = self.loss + 0.1 * self.classification_loss
        # self.final_loss = tf.cond(self.epoch < 1000, lambda: self.loss, lambda: (self.loss + 0.1 * self.classification_loss))

        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        self.final_loss += sum(reg_losses)

        tf.compat.v1.summary.scalar('loss', self.final_loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.compat.v1.summary.scalar('auc', self.auc)
            self.global_step = tf.compat.v1.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            # self.optimizer = tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=0.001)
            self.train_op = self.optimizer.minimize(self.final_loss, global_step=self.global_step)
            # self.train_op = self.optimizer.minimize(self.final_loss, var_list= self.optimizer.get_weights() ,tape=tf.GradientTape())
        else:
            tf.compat.v1.summary.scalar('test_auc', self.auc)

        self.merged = tf.compat.v1.summary.merge_all()

    def predict(self, sess, u, seq, item_idx, seq_cxt, test_item_cxt, buy_seq_mask, cart_seq_mask, fav_seq_mask,
                click_seq_mask):
        from Recommendation_Beibei_index import model
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False,
                         self.seq_cxt: seq_cxt, self.test_item_cxt: test_item_cxt, \
                         model.buy_seq_mask: buy_seq_mask, model.cart_seq_mask: cart_seq_mask,
                         model.fav_seq_mask: fav_seq_mask, model.click_seq_mask: click_seq_mask})


class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence, behaviour, weights):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        copied_behaviour = copy.deepcopy(behaviour)
        copied_weights = copy.deepcopy(weights)

        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length < 1:
            return [copied_sequence[start_index]], [copied_behaviour[start_index]], [copied_weights[start_index]]
        else:
            cropped_seq = copied_sequence[start_index:start_index + sub_seq_length]
            cropped_beh = copied_behaviour[start_index:start_index + sub_seq_length]
            cropped_w = copied_weights[start_index:start_index + sub_seq_length]
            return cropped_seq, cropped_beh, cropped_w


class Mask(object):
    """Randomly mask k items given a sequence"""

    def __init__(self, gamma=0.7):
        self.gamma = gamma

    def __call__(self, sequence, behaviour, weights):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        copied_behaviour = copy.deepcopy(behaviour)
        copied_weights = copy.deepcopy(weights)

        mask_nums = int(self.gamma * len(copied_sequence))
        mask = [0 for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
            copied_weights[idx] = copied_weights[idx] * mask_value
            copied_behaviour[idx] = [i * mask_value for i in copied_behaviour[idx]]
        return copied_sequence, copied_behaviour, copied_weights


class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""

    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence, behaviour, weights):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        copied_behaviour = copy.deepcopy(behaviour)
        copied_weights = copy.deepcopy(weights)

        sub_seq_length = int(self.beta * len(copied_sequence))

        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        sub_seq = copied_sequence[start_index:start_index + sub_seq_length]
        sub_beh = copied_behaviour[start_index:start_index + sub_seq_length]
        sub_weight = copied_weights[start_index:start_index + sub_seq_length]
        # random.shuffle(sub_seq)
        # Create a permutation of indices.
        indices = list(range(len(sub_seq)))  # Generate a list of indices.
        random.shuffle(indices)  # Shuffle the indices.

        # Apply the permutation to your lists.
        sub_seq = [sub_seq[i] for i in indices]
        sub_beh = [sub_beh[i] for i in indices]
        sub_weight = [sub_weight[i] for i in indices]

        # sub_seq, sub_beh, sub_weight = list(sub_seq), list(sub_beh), list(sub_weight)
        reordered_seq = copied_sequence[:start_index] + sub_seq + \
                        copied_sequence[start_index + sub_seq_length:]

        assert len(copied_sequence) == len(reordered_seq)

        reordered_beh = copied_behaviour[:start_index] + sub_beh + \
                        copied_behaviour[start_index + sub_seq_length:]
        assert len(copied_behaviour) == len(reordered_beh)

        reordered_w = copied_weights[:start_index] + sub_weight + \
                      copied_weights[start_index + sub_seq_length:]
        assert len(copied_weights) == len(reordered_w)

        return reordered_seq, reordered_beh, reordered_w
