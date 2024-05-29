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
       
