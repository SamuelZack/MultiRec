

class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
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
        from Recommendation_IJCAI_Method import embedding, Model_pro
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
       
