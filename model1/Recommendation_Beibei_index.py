# -*- coding: utf-8 -*-

import numpy as np
import pickle
# import tensorflow_addons as tfa
import tensorflow as tf
import time
import timeit

from Recommendation_Beibei_Method import data_partition_beibei, evaluate_valid
from Recommendation_Beibei_Model import WarpSampler, Model, Args

args = Args()

tf.compat.v1.disable_eager_execution()
tf.random.set_seed(42)

tstInt = None
with open('./Datasets/tst_int', 'rb') as fs:
    tstInt = np.array(pickle.load(fs))
tstStat = (tstInt != None)
tstUsrs = np.reshape(np.argwhere(tstStat != False), [-1])
tstUsrs = tstUsrs + 1
print(len(tstUsrs))

print('maxlen...', args.maxlen, '   heads....', args.num_heads, '   dropout...', args.dropout_rate, '  lr,...', args.lr,
      ' emb size...', args.hidden_units, 'min len..', args.min_seq, ' L2 Reg..', args.l2_emb, 'Seed 42')
# print(' augmentation only crop 0.2 and mask 0.4 .... loss weight 0.02')
dataset = data_partition_beibei(args.dataset)
print('here')
[user_train, user_valid, user_test, Beh, user_valid_beh, Beh_w, Behaviors, usernum, itemnum] = dataset
print(usernum, '-', itemnum)
print('Nuber of interactions ...', len(user_train))
num_batch = len(user_train) / args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

# f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

sampler = WarpSampler(user_train, Beh, Beh_w, Behaviors, usernum, itemnum, batch_size=args.batch_size,
                      maxlen=args.maxlen, n_workers=3)
model = Model(usernum, itemnum, args)
sess.run(tf.compat.v1.initialize_all_variables())

T = 0.0
t0 = time.time()

for epoch in range(1, args.num_epochs + 1):
    total_loss = 0
    # for step in tqdm(range(int(num_batch)), total=int(num_batch), ncols=70, leave=False, unit='b'):

    for step in range(0, int(num_batch)):
        start = timeit.default_timer()
        u, seq, pos, neg, seq_cxt, pos_cxt, pos_weight, neg_weight, buy_seq_mask, cart_seq_mask, fav_seq_mask, click_seq_mask, recency, \
            Aug_seq, Aug_seq_cxt, Aug_buy_seq_mask, Aug_cart_seq_mask, Aug_fav_seq_mask, Aug_click_seq_mask, labels = sampler.next_batch()

        loss, _ = sess.run([model.final_loss, model.train_op],
                           {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                            model.buy_seq_mask: buy_seq_mask, model.cart_seq_mask: cart_seq_mask,
                            model.fav_seq_mask: fav_seq_mask, model.click_seq_mask: click_seq_mask,
                            model.is_training: True, model.seq_cxt: seq_cxt, model.pos_cxt: pos_cxt,
                            model.pos_weight: pos_weight,
                            model.neg_weight: neg_weight, model.recency: recency,
                            model.Aug_input_seq: Aug_seq, model.Aug_seq_cxt: Aug_seq_cxt,
                            model.Aug_buy_seq_mask: Aug_buy_seq_mask, model.Aug_cart_seq_mask: Aug_cart_seq_mask,
                            model.Aug_fav_seq_mask: Aug_fav_seq_mask, model.Aug_click_seq_mask: Aug_click_seq_mask,
                            model.labels: labels, model.epoch: epoch})
        # print('input sequence....', seq)
        # print('sim.......', con_mask)
        # print('loss.......', con_loss)
        # print(abc)
        total_loss = total_loss + loss
        stop = timeit.default_timer()
        # print('Time for batch in sec.....: ', stop - start)
    # print(abc)
    print('loss in epoch...', epoch, ' is  ', total_loss / int(num_batch))
    if epoch > 50 and epoch % 5 == 0:
        t1 = time.time() - t0
        T += t1
        print('Evaluating')
        t_valid = evaluate_valid(model, dataset, args, sess, Beh, epoch)

        print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f)' % (epoch, T, t_valid[0]))

sampler.close()
# f.close()

print("Done")
