import sys
from Recommendation_Beibei_Model import Crop, Mask, Args
import numpy as np
from collections import defaultdict
import copy
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.random.set_seed(42)


def sample_function(user_train, Beh, Beh_w, Behaviors, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        recency_alpha = 0.7
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        buy_seq_mask = np.zeros([maxlen], dtype=np.int32)
        cart_seq_mask = np.zeros([maxlen], dtype=np.int32)
        fav_seq_mask = np.zeros([maxlen], dtype=np.int32)
        click_seq_mask = np.zeros([maxlen], dtype=np.int32)

        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        recency = np.zeros([maxlen], dtype=np.float32)

        pos_weight = np.zeros([maxlen], dtype=np.float32)
        neg_weight = np.zeros([maxlen], dtype=np.float32)

        nxt = user_train[user][-1]
        idx = maxlen - 1
        seq_len = len(user_train[user]) - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):

            seq[idx] = i
            pos[idx] = nxt
            recency[idx] = recency_alpha ** (maxlen - idx)

            pos_weight[idx] = Beh_w[user][seq_len]
            neg_weight[idx] = 1.0

            if Behaviors[user][seq_len - 1] == 'pos':
                buy_seq_mask[idx] = 1
            elif Behaviors[user][seq_len - 1] == 'tip':
                cart_seq_mask[idx] = 1
            elif Behaviors[user][seq_len - 1] == 'neutral':
                fav_seq_mask[idx] = 1
            elif Behaviors[user][seq_len - 1] == 'neg':
                click_seq_mask[idx] = 1

            # print('recency[idx]...', recency[idx])
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            seq_len -= 1
            if idx == -1: break
        # print(abc)

        seq_cxt = list()
        pos_cxt = list()
        labels = list()
        # pos_weight = list()
        # neg_weight = list()
        seq_len = len(user_train[user]) - 1
        # print('sequence length is....', seq_len)
        # print('sequence ................', seq)
        # print('sequence behaviour is.... ', len(Beh[user]), '        '  , Beh[user])
        for i in seq:
            if i == 0:
                seq_cxt.append([0, 0, 0, 0])
            else:
                seq_cxt.append(Beh[user][seq_len - 1])
                seq_len -= 1

        pos_seq_len = len(user_train[user]) - 1
        for i in pos:
            # print('pos item.....', i, '    Beh[user][pos_seq_len]', Beh[user][pos_seq_len], ' Behaviors[user][pos_seq_len]', Behaviors[user][pos_seq_len])
            if i == 0:
                pos_cxt.append([0, 0, 0, 0])
                labels.append([0, 0, 0, 0])
            else:
                pos_cxt.append(Beh[user][pos_seq_len])
                # print('len: ',len(Behaviors[user]))
                # print('index: ',pos_seq_len)
                if Behaviors[user][pos_seq_len] == 'pos':
                    labels.append([1, 0, 0, 0])

                elif Behaviors[user][pos_seq_len] == 'tip':
                    labels.append([0, 1, 0, 0])

                elif Behaviors[user][pos_seq_len] == 'neutral':
                    labels.append([0, 0, 1, 0])

                elif Behaviors[user][pos_seq_len] == 'neg':
                    labels.append([0, 0, 0, 1])

                pos_seq_len -= 1

        # for i in pos :
        #    pos_weight.append(Beh_w[(user,i)])
        #    neg_weight.append(1.0)

        seq_cxt = np.asarray(seq_cxt)
        pos_cxt = np.asarray(pos_cxt)
        labels = np.asarray(labels)
        # print('length of labels: ', len(labels), '   len of items in labels: ', len(labels[0]))
        # pos_weight = np.asarray(pos_weight)
        Aug_seq, Aug_seq_cxt, Aug_buy_seq_mask, Aug_cart_seq_mask, Aug_fav_seq_mask, Aug_click_seq_mask = Augmented_seq(
            user_train[user][:-1], Beh[user][:-1], Behaviors, Beh_w[user][:-1], maxlen)

        return (
            user, seq, pos, neg, seq_cxt, pos_cxt, pos_weight, neg_weight, buy_seq_mask, cart_seq_mask, fav_seq_mask,
            click_seq_mask, recency, \
            Aug_seq, Aug_seq_cxt, Aug_buy_seq_mask, Aug_cart_seq_mask, Aug_fav_seq_mask, Aug_click_seq_mask, labels)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def Augmented_seq(seq, beh_seq, behavior, seq_w, maxlen):
    Aug_seq = np.zeros([maxlen], dtype=np.int32)
    Aug_buy_seq_mask = np.zeros([maxlen], dtype=np.int32)
    Aug_cart_seq_mask = np.zeros([maxlen], dtype=np.int32)
    Aug_fav_seq_mask = np.zeros([maxlen], dtype=np.int32)
    Aug_click_seq_mask = np.zeros([maxlen], dtype=np.int32)
    from Recommendation_Beibei_index import args
    if len(seq) < args.min_seq:
        new_seq, new_beh_seq, new_seq_w = seq, beh_seq, seq_w
    else:
        op = np.random.randint(1, 10)
        if op > 0 and op <= 5:
            crop = Crop(tao=0.2)  # already tried 0.4
            new_seq, new_beh_seq, new_seq_w = crop(seq, beh_seq, seq_w)
            # elif op >3 and op <= 6:
        #   reorder = Reorder(beta=0.2)
        #   new_seq, new_beh_seq, new_seq_w = reorder(seq, beh_seq, seq_w)
        elif op > 5 and op <= 10:
            mask = Mask(gamma=0.4)  # already tried 0.2 and 0.6
            new_seq, new_beh_seq, new_seq_w = mask(seq, beh_seq, seq_w)

            # nxt = new_seq[-1]
    idx = maxlen - 1
    seq_len = len(new_seq)

    for i in reversed(new_seq[:]):

        Aug_seq[idx] = i
        if behavior[seq_len - 1] == 'buy':
            Aug_buy_seq_mask[idx] = 1
        elif behavior[seq_len - 1] == 'cart':
            Aug_cart_seq_mask[idx] = 1
        elif behavior[seq_len - 1] == 'fav':
            Aug_fav_seq_mask[idx] = 1
        elif behavior[seq_len - 1] == 'pv':
            Aug_click_seq_mask[idx] = 1

        # print('recency[idx]...', recency[idx])
        # nxt = i
        idx -= 1
        seq_len -= 1
        if idx == -1: break
    # print(abc)

    Aug_seq_cxt = list()

    seq_len = len(new_seq)

    for i in Aug_seq:
        if i == 0:
            Aug_seq_cxt.append([0, 0, 0, 0])
        else:
            Aug_seq_cxt.append(new_beh_seq[seq_len - 1])
            seq_len -= 1

    Aug_seq_cxt = np.asarray(Aug_seq_cxt)

    return Aug_seq, Aug_seq_cxt, Aug_buy_seq_mask, Aug_cart_seq_mask, Aug_fav_seq_mask, Aug_click_seq_mask


def data_partition_Beibei(fname):
    usernum = 0
    itemnum = 0
    interactions = 0
    User = defaultdict(list)
    user_train = {}
    user_last_indx = {}
    user_valid = {}
    user_valid_beh = {}
    user_test = {}
    Beh = {}

    Beh_w = 0.0
    User_weights = defaultdict(list)
    seq_Beh = defaultdict(list)
    Behaviors = defaultdict(list)

    # assume user/item index starting from 1
    f = open('./Datasets/%s.txt' % fname, 'r')
    next(f)
    for line in f:
        interactions += 1
        u, i, b = line.rstrip().split(' ')
        # print( 'data type of user ....', u, '  ',i,'    ', b)
        # print(abc)
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        if b == 'buy':
            last_pos_idx = len(User[u])
            user_last_indx[u] = last_pos_idx
            Beh_w = 0.7
            if (u, i) in Beh:
                Beh[(u, i)][0] = Beh[(u, i)][0] + 1
                # Beh[(u,i)] = [1,0,0,0]
            else:
                Beh[(u, i)] = [1, 0, 0, 0]
            # if Beh[(u,i)][0] ==1:
            User_weights[u].append(Beh_w)
            # else:
            #   User_weights[u].append(Beh_w+(Beh[(u,i)][0]/100))

        elif b == 'cart':
            Beh_w = 0.1
            if (u, i) in Beh:
                Beh[(u, i)][1] = Beh[(u, i)][1] + 1
                # Beh[(u,i)] = [0,1,0,0]
            else:
                Beh[(u, i)] = [0, 1, 0, 0]
            # if Beh[(u,i)][1] ==1:
            User_weights[u].append(Beh_w)
            # else:
            #  User_weights[u].append(Beh_w+(Beh[(u,i)][1]/100))

        elif b == 'fav':
            Beh_w = 0.1
            if (u, i) in Beh:
                Beh[(u, i)][2] = Beh[(u, i)][2] + 1
                # Beh[(u,i)] = [0,0,1,0]
            else:
                Beh[(u, i)] = [0, 0, 1, 0]
                # if Beh[(u,i)][2] ==1:
            User_weights[u].append(Beh_w)
            # else:
            #  User_weights[u].append(Beh_w+(Beh[(u,i)][2]/100))

        elif b == 'pv':
            Beh_w = 0.1
            if (u, i) in Beh:
                # Beh[(u,i)] = [0,0,0,1]
                Beh[(u, i)][3] = Beh[(u, i)][3] + 1
            else:
                Beh[(u, i)] = [0, 0, 0, 1]

            # if Beh[(u,i)][3] ==1:
            User_weights[u].append(Beh_w)
            # else:
            #  User_weights[u].append(Beh_w+(Beh[(u,i)][3]/100))

        User[u].append(i)
        seq_Beh[u].append(Beh[(u, i)])
        Behaviors[u].append(b)
    print('Total Number of interactions is .....', interactions)
    for user in User:
        Beh[(user, 0)] = [0, 0, 0, 0]
        # Beh_w[(user,0)] = 0
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            last_item_indx = user_last_indx[user]
            last_item = User[user][last_item_indx]
            last_item_beh = seq_Beh[user][last_item_indx]

            items_list = User[user]
            del items_list[last_item_indx]

            items_beh_list = seq_Beh[user]
            del items_beh_list[last_item_indx]

            items_weight_list = User_weights[user]
            del items_weight_list[last_item_indx]

            behaviors_list = Behaviors[user]
            del behaviors_list[last_item_indx]

            # user_train[user] = items_list
            # user_train[user] = [value for value in items_list if value != last_item]
            new_list = []
            new_beh = []
            new_weight = []
            new_behaviors = []
            truncated_item_list = items_list[:last_item_indx]
            truncated_beh_list = items_beh_list[:last_item_indx]
            truncated_weights_list = items_weight_list[:last_item_indx]
            truncated_behaviors_list = behaviors_list[:last_item_indx]
            '''
            for  counter in range (0, len(truncated_item_list)) :
              if truncated_item_list[counter] != last_item:
                 new_list.append( truncated_item_list[counter])
                 new_beh.append(truncated_beh_list[counter])
                 new_weight.append(truncated_weights_list[counter])
                 new_behaviors.append(truncated_behaviors_list[counter])

            user_train[user] = new_list
            seq_Beh[user] = new_beh
            User_weights[user] = new_weight
            Behaviors[u]  = new_behaviors
            '''
            user_train[user] = truncated_item_list
            seq_Beh[user] = truncated_beh_list
            User_weights[user] = truncated_weights_list
            Behaviors[user] = truncated_behaviors_list
            # user_train[user] = [value for value in truncated_item_list if value != last_item]
            # user_train[user] = [value for value in truncated_item_list]

            # truncated_beh_list = items_beh_list[:last_item_indx]
            # seq_Beh[user] = [value for value in truncated_beh_list]

            # truncated_weights_list = items_weight_list[:last_item_indx]
            # User_weights[user] = [value for value in truncated_weights_list]

            # print('len of seq.....', len(user_train[user]))
            # print('len of seq beh.....', len(seq_Beh[user]))
            # print('len of seq weights.....', len(User_weights[user]))
            # print('=================================================')
            user_valid[user] = []
            user_valid[user].append(last_item)
            user_valid_beh[user] = []
            user_valid_beh[user].append(last_item_beh)
            user_test[user] = []
            user_test[user].append(last_item)
    return [user_train, user_valid, user_test, seq_Beh, user_valid_beh, User_weights, Behaviors, usernum, itemnum]


def data_partition_yelp(fname):
    usernum = 0
    itemnum = 0
    interactions = 0
    User = defaultdict(list)
    user_train = {}
    user_last_indx = {}
    user_valid = {}
    user_valid_beh = {}
    user_test = {}
    Beh = {}

    Beh_w = 0.0
    User_weights = defaultdict(list)
    seq_Beh = defaultdict(list)
    Behaviors = defaultdict(list)

    # assume user/item index starting from 1
    f = open('./Datasets/%s.txt' % fname, 'r')
    next(f)
    for line in f:
        interactions += 1
        u, i, b = line.rstrip().split(' ')

        # print(abc)
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        if b == 'pos':
            last_pos_idx = len(User[u])
            user_last_indx[u] = last_pos_idx
            Beh_w = 0.3
            if (u, i) in Beh:
                Beh[(u, i)][0] = Beh[(u, i)][0] + 1
                # Beh[(u,i)] = [1,0,0,0]
            else:
                Beh[(u, i)] = [1, 0, 0, 0]
            # if Beh[(u,i)][0] ==1:
            User_weights[u].append(Beh_w)
            # else:
            #   User_weights[u].append(Beh_w+(Beh[(u,i)][0]/100))

        elif b == 'tip':
            Beh_w = 0.3
            if (u, i) in Beh:
                Beh[(u, i)][1] = Beh[(u, i)][1] + 1
                # Beh[(u,i)] = [0,1,0,0]
            else:
                Beh[(u, i)] = [0, 1, 0, 0]
            # if Beh[(u,i)][1] ==1:
            User_weights[u].append(Beh_w)
            # else:
            #  User_weights[u].append(Beh_w+(Beh[(u,i)][1]/100))

        elif b == 'neutral':
            Beh_w = 0.2
            if (u, i) in Beh:
                Beh[(u, i)][2] = Beh[(u, i)][2] + 1
                # Beh[(u,i)] = [0,0,1,0]
            else:
                Beh[(u, i)] = [0, 0, 1, 0]
                # if Beh[(u,i)][2] ==1:
            User_weights[u].append(Beh_w)
            # else:
            #  User_weights[u].append(Beh_w+(Beh[(u,i)][2]/100))

        elif b == 'neg':
            Beh_w = 0.2
            if (u, i) in Beh:
                # Beh[(u,i)] = [0,0,0,1]
                Beh[(u, i)][3] = Beh[(u, i)][3] + 1
            else:
                Beh[(u, i)] = [0, 0, 0, 1]

            # if Beh[(u,i)][3] ==1:
            User_weights[u].append(Beh_w)
            # else:
            #  User_weights[u].append(Beh_w+(Beh[(u,i)][3]/100))

        User[u].append(i)
        seq_Beh[u].append(Beh[(u, i)])
        Behaviors[u].append(b)
        # if u == 19800:
        #       print('------------------------------------------------------- ', u)
        #       print( 'data type of user ....', User[u])
        #       print('BEhaviourss.....', Behaviors[u])
        #       print('len of seq.....', len(User[u]))
        #       print('len of seq beh.....', len(seq_Beh[u]))
        #       print('len of seq weights.....', len(User_weights[u]))
        #       print('len of seq Behaviors.....', len(Behaviors[u]))

    print('Total Number of interactions is .....', interactions)
    for user in User:
        Beh[(user, 0)] = [0, 0, 0, 0]
        # Beh_w[(user,0)] = 0
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            seq_Beh[user] = seq_Beh[user]
            User_weights[user] = User_weights[user]
            Behaviors[u] = Behaviors[user]

            user_valid[user] = []
            user_test[user] = []
        else:
            last_item_indx = user_last_indx[user]
            last_item = User[user][last_item_indx]
            last_item_beh = seq_Beh[user][last_item_indx]

            items_list = User[user]
            del items_list[last_item_indx]

            items_beh_list = seq_Beh[user]
            del items_beh_list[last_item_indx]

            items_weight_list = User_weights[user]
            del items_weight_list[last_item_indx]
            # if user == 19800:
            #   print('---------------------*********************---------------------------------- ', user)
            #   print( 'data type of user ....', User[user])
            #   print('BEhaviourss.....', Behaviors[user])
            #   print('len of seq.....', len(User[user]))
            #   print('len of seq beh.....', len(seq_Beh[user]))
            #   print('len of seq weights.....', len(User_weights[user]))
            #   print('len of seq Behaviors.....', len(Behaviors[user]))

            behaviors_list = Behaviors[user]
            del behaviors_list[last_item_indx]

            # user_train[user] = items_list
            # user_train[user] = [value for value in items_list if value != last_item]
            new_list = []
            new_beh = []
            new_weight = []
            new_behaviors = []
            truncated_item_list = items_list[:last_item_indx]
            truncated_beh_list = items_beh_list[:last_item_indx]
            truncated_weights_list = items_weight_list[:last_item_indx]
            truncated_behaviors_list = behaviors_list[:last_item_indx]

            user_train[user] = truncated_item_list
            seq_Beh[user] = truncated_beh_list
            User_weights[user] = truncated_weights_list
            Behaviors[user] = truncated_behaviors_list
            # user_train[user] = [value for value in truncated_item_list if value != last_item]
            # user_train[user] = [value for value in truncated_item_list]

            # truncated_beh_list = items_beh_list[:last_item_indx]
            # seq_Beh[user] = [value for value in truncated_beh_list]

            # truncated_weights_list = items_weight_list[:last_item_indx]
            # User_weights[user] = [value for value in truncated_weights_list]

            # print('len of seq.....', len(user_train[user]))
            # print('len of seq beh.....', len(seq_Beh[user]))
            # print('len of seq weights.....', len(User_weights[user]))
            # print('=================================================')
            user_valid[user] = []
            user_valid[user].append(last_item)
            user_valid_beh[user] = []
            user_valid_beh[user].append(last_item_beh)
            user_test[user] = []
            user_test[user].append(last_item)
    return [user_train, user_valid, user_test, seq_Beh, user_valid_beh, User_weights, Behaviors, usernum, itemnum]


def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        from Recommendation_Beibei_index import tstUsrs
        users = tstUsrs  # random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(99):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess, Beh, epoch):
    [train, valid, test, Beh, user_valid_beh, Beh_w, Behaviors, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        from Recommendation_Beibei_index import tstUsrs
        users = tstUsrs  # random.sample(range(1, usernum + 1), 10000)
        print(len(users))
    else:
        users = range(1, usernum + 1)
    for u in users:
        seq_cxt = list()

        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        buy_seq_mask = np.zeros([args.maxlen], dtype=np.int32)
        cart_seq_mask = np.zeros([args.maxlen], dtype=np.int32)
        fav_seq_mask = np.zeros([args.maxlen], dtype=np.int32)
        click_seq_mask = np.zeros([args.maxlen], dtype=np.int32)

        # seq_cxt = np.zeros([args.maxlen], dtype=np.float32)
        idx = args.maxlen - 1
        seq_len = len(train[u]) - 1
        for i in reversed(train[u]):
            seq[idx] = i

            if Behaviors[u][seq_len] == 'pos':
                buy_seq_mask[idx] = 1
            elif Behaviors[u][seq_len] == 'tip':
                cart_seq_mask[idx] = 1
            elif Behaviors[u][seq_len] == 'neutral':
                fav_seq_mask[idx] = 1
            elif Behaviors[u][seq_len] == 'neg':
                click_seq_mask[idx] = 1

            idx -= 1
            seq_len -= 1
            if idx == -1: break

        # for i in seq :
        #    seq_cxt.append(Beh[(u,i)])

        seq_len_cxt = len(train[u]) - 1
        for i in seq:  # add the seq behaviours
            if i == 0:
                seq_cxt.append([0, 0, 0, 0])
            else:
                # print('sequence length is....', seq_len_cxt)
                # print('sequence ................', seq)
                # print('sequence behaviour is.... ', len(Beh[u]))
                # print('============================================================')
                seq_cxt.append(Beh[u][seq_len_cxt])
                seq_len_cxt -= 1
        seq_cxt = np.asarray(seq_cxt)

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]

        testitemscxt = list()
        test_Behavior = user_valid_beh[u][0]

        # testitemscxt.append(user_valid_beh[u][0])
        # testitemscxt.append([1,0,0,0]) # always have the traget behaviour to be the buy only not to give the whole history while testing
        testitemscxt.append(test_Behavior)
        # print('behaviour of the tragte item..........', user_valid_beh[u][0])

        for _ in range(99):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            testitemscxt.append([1, 0, 0, 0])

        predictions = -model.predict(sess, [u], [seq], item_idx, [seq_cxt], testitemscxt, [buy_seq_mask],
                                     [cart_seq_mask], [fav_seq_mask], [click_seq_mask])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
            # if  epoch<200:
            # with open('/home/elsayed/MultiBehaviour/Tianchi_CUT_Results/Tianchi_CUT_Without_Behaviours_'+str(epoch)+'.txt', 'a') as f:
            # user  HR     NDCG
            # f.write(str(u)+';'+str(1)+';'+str(1/np.log2(rank + 2))+'\n')
        if valid_user % 100 == 0:
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]tf.compat.v1.layers.dense

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        lookup_table = tf.compat.v1.get_variable('lookup_table',
                                                 dtype=tf.float32,
                                                 shape=[vocab_size, num_units],
                                                 # initializer=tf.contrib.layers.xavier_initializer(),
                                                 regularizer=tf.keras.regularizers.l2(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        res=False,
                        with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.compat.v1.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        K = tf.compat.v1.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        V = tf.compat.v1.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.compat.v1.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        if res:
            outputs = (
                    outputs + queries)  # keys here will act as adding positional encoding after having the attention with the input sequence.

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs


def contrastive_loss(padded_seq,
                     batch_sample_one,
                     batch_sample_two,
                     temperature=1.0
                     ):
    # print('   size of first batch in contrastive loss.......',tf.shape(batch_sample_one))
    # print('   size of second batch in contrastive loss.......',tf.shape(batch_sample_two))
    sim11 = tf.matmul(batch_sample_one, tf.transpose(batch_sample_one)) / temperature
    sim22 = tf.matmul(batch_sample_two, tf.transpose(batch_sample_two)) / temperature
    sim12 = tf.matmul(batch_sample_one, tf.transpose(batch_sample_two)) / temperature
    d = tf.shape(sim12)[-1]
    # print('   size of d.......',d)
    # print('   size of sim11.......',tf.shape(sim11))
    sim11 = tf.linalg.set_diag(sim11, tf.fill([d], float('-inf')))
    sim22 = tf.linalg.set_diag(sim22, tf.fill([d], float('-inf')))
    raw_scores1 = tf.concat([sim12, sim11], axis=-1)
    # print('   size of raw_scores1.......',tf.shape(raw_scores1))
    raw_scores2 = tf.concat([sim22, tf.transpose(sim12, perm=[1, 0])], axis=-1)
    # print('   size of raw_scores2.......',tf.shape(raw_scores2))
    logits = tf.concat([raw_scores1, raw_scores2], axis=-2)
    # print('   size of logits.......',tf.shape(logits))
    labels = tf.range(2 * d, dtype=tf.int64)
    # print('   size of logits.......',tf.shape(labels))

    # istarget_con = tf.reshape(tf.compat.v1.to_float(tf.not_equal(padded_seq, 0)), [tf.shape(self.padded_seq)[0] * args.maxlen])
    # con_loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits= logits, axis=-1)
    scce3 = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)
    con_loss = tf.reduce_sum(scce3(labels, tf.nn.softmax(logits)))
    return con_loss, sim12


def Model_pro(in_seq, mask, seq_cxt, args, buy_seq_mask, cart_seq_mask, fav_seq_mask, click_seq_mask, resweight_buy,
              resweight_cart, resweight_fav, resweight_click, is_training, reuse=None):
    seq_cxt_emb = tf.compat.v1.layers.dense(inputs=seq_cxt, units=args.hidden_units, activation=None,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            name="cxt_emb", reuse=reuse,
                                            kernel_regularizer=tf.keras.regularizers.l2(args.l2_emb))

    seq = tf.concat([in_seq, seq_cxt_emb], -1)
    # cxt
    seq = tf.compat.v1.layers.dense(inputs=seq, units=args.hidden_units, activation=None,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="feat_emb",
                                    reuse=reuse,
                                    kernel_regularizer=tf.keras.regularizers.l2(args.l2_emb))

    # Positional Encoding
    '''
    self.projection, pos_emb_table = embedding(
        tf.tile(tf.expand_dims(tf.range(args.projection_size), 0), [tf.shape(self.input_seq)[0], 1]),
        vocab_size=args.maxlen,
        num_units=args.hidden_units,
        zero_pad=False,
        scale=False,
        l2_reg=args.l2_emb,
        scope="dec_pos",
        reuse=reuse,
        with_t=True
    )
    '''
    # Positional Encoding
    t, pos_emb_table = embedding(
        tf.tile(tf.expand_dims(tf.range(tf.shape(in_seq)[1]), 0), [tf.shape(in_seq)[0], 1]),
        vocab_size=args.maxlen,
        num_units=args.hidden_units,
        zero_pad=False,
        scale=False,
        l2_reg=args.l2_emb,
        scope="positional",
        reuse=reuse,
        with_t=True
    )
    seq += t

    # Dropout
    seq = tf.compat.v1.layers.dropout(seq,
                                      rate=args.dropout_rate,
                                      training=tf.convert_to_tensor(is_training))

    seq_res = seq * mask
    # ================== attention between all buy items ====================
    buy_seq_mask = tf.tile(tf.expand_dims(buy_seq_mask, -1),
                           [1, 1, args.hidden_units])  # change the mask shape to be (batch, maxlen, hidden emb)
    buy_seq_in = seq * buy_seq_mask
    buy_seq_in = normalize(buy_seq_in)
    for i in range(args.num_blocks):
        with tf.compat.v1.variable_scope("buy_num_blocks_%d" % i):
            buy_seq = multihead_attention(queries=buy_seq_in,
                                          keys=buy_seq_in,
                                          num_units=args.hidden_units,
                                          num_heads=args.num_heads,
                                          dropout_rate=args.dropout_rate,
                                          is_training=is_training,
                                          causality=True,
                                          reuse=reuse,
                                          res=False,
                                          scope="buy_self_attention")
            # Feed forward
            # buy_seq = buy_seq + buy_seq_in
            buy_seq = feedforward(buy_seq, num_units=[args.hidden_units, args.hidden_units],
                                  dropout_rate=args.dropout_rate, is_training=is_training, scope="buy_self_attention",
                                  reuse=reuse)

            buy_seq *= buy_seq_mask
            buy_seq = buy_seq + (buy_seq_in * resweight_buy)
            # buy_seq = buy_seq_in
            # buy_seq *= 0.7*(buy_seq_mask)

    # ================== attention between all buy items ====================
    cart_seq_mask = tf.tile(tf.expand_dims(cart_seq_mask, -1),
                            [1, 1, args.hidden_units])  # change the mask shape to be (batch, maxlen, hidden emb)
    cart_seq_in = seq * cart_seq_mask
    cart_seq_in = normalize(cart_seq_in)
    for i in range(args.num_blocks):
        with tf.compat.v1.variable_scope("cart_num_blocks_%d" % i):
            cart_seq = multihead_attention(queries=cart_seq_in,
                                           keys=cart_seq_in,
                                           num_units=args.hidden_units,
                                           num_heads=args.num_heads,
                                           dropout_rate=args.dropout_rate,
                                           is_training=is_training,
                                           causality=True,
                                           reuse=reuse,
                                           res=False,
                                           scope="cart_self_attention")
            # Feed forward
            # cart_seq = cart_seq + cart_seq_in
            cart_seq = feedforward(cart_seq, num_units=[args.hidden_units, args.hidden_units],
                                   dropout_rate=args.dropout_rate, is_training=is_training, scope="cart_self_attention",
                                   reuse=reuse)

            cart_seq *= cart_seq_mask
            cart_seq = cart_seq + (cart_seq_in * resweight_cart)
            # cart_seq =  cart_seq_in
            # cart_seq *= 0.1*(cart_seq_mask)

    # ================== attention between all buy items ====================
    fav_seq_mask = tf.tile(tf.expand_dims(fav_seq_mask, -1),
                           [1, 1, args.hidden_units])  # change the mask shape to be (batch, maxlen, hidden emb)
    fav_seq_in = seq * fav_seq_mask
    fav_seq_in = normalize(fav_seq_in)
    for i in range(args.num_blocks):
        with tf.compat.v1.variable_scope("fav_num_blocks_%d" % i):
            fav_seq = multihead_attention(queries=fav_seq_in,
                                          keys=fav_seq_in,
                                          num_units=args.hidden_units,
                                          num_heads=args.num_heads,
                                          dropout_rate=args.dropout_rate,
                                          is_training=is_training,
                                          causality=True,
                                          reuse=reuse,
                                          res=False,
                                          scope="fav_self_attention")
            # Feed forward
            # fav_seq = fav_seq + fav_seq_in
            fav_seq = feedforward(fav_seq, num_units=[args.hidden_units, args.hidden_units],
                                  dropout_rate=args.dropout_rate, is_training=is_training, scope="fav_self_attention",
                                  reuse=reuse)

            fav_seq *= fav_seq_mask
            fav_seq = fav_seq + (fav_seq_in * resweight_fav)
            # fav_seq =  fav_seq_in
            # fav_seq *= 0.1*(fav_seq_mask)

    # ================== attention between all buy items ====================
    click_seq_mask = tf.tile(tf.expand_dims(click_seq_mask, -1),
                             [1, 1, args.hidden_units])  # change the mask shape to be (batch, maxlen, hidden emb)
    click_seq_in = seq * click_seq_mask
    click_seq_in = normalize(click_seq_in)
    for i in range(args.num_blocks):
        with tf.compat.v1.variable_scope("click_num_blocks_%d" % i):
            click_seq = multihead_attention(queries=click_seq_in,
                                            keys=click_seq_in,
                                            num_units=args.hidden_units,
                                            num_heads=args.num_heads,
                                            dropout_rate=args.dropout_rate,
                                            is_training=is_training,
                                            causality=True,
                                            reuse=reuse,
                                            res=False,
                                            scope="click_self_attention")
            # Feed forward
            # click_seq = click_seq + click_seq_in
            click_seq = feedforward(click_seq, num_units=[args.hidden_units, args.hidden_units],
                                    dropout_rate=args.dropout_rate, is_training=is_training,
                                    scope="click_self_attention", reuse=reuse)

            click_seq *= click_seq_mask
            click_seq = click_seq + (click_seq_in * resweight_click)
            # click_seq =  click_seq_in
            # click_seq *= 0.1*(click_seq_mask)

    # seq = normalize(buy_seq) + normalize(cart_seq) + normalize(fav_seq) + normalize(click_seq) # +seq_cxt_emb # adding again the cxt
    # seq = buy_seq + cart_seq + fav_seq + click_seq +seq_cxt_emb
    # seq = normalize(seq) + t
    seq = seq_res
    # =========== attention between all items of the input sequence ==================

    # Build blocks
    seq *= mask
    for i in range(args.num_blocks):
        with tf.compat.v1.variable_scope("num_blocks_%d" % i):
            # Pack-attention
            seq = multihead_attention(queries=seq,
                                      keys=seq,
                                      num_units=args.hidden_units,
                                      num_heads=args.num_heads,
                                      dropout_rate=args.dropout_rate,
                                      is_training=is_training,
                                      causality=True,
                                      reuse=reuse,
                                      res=True,
                                      scope="self_attention")

            seq = seq + seq_res  # res with the original seq
            # Feed forward
            seq = feedforward(normalize(seq), num_units=[args.hidden_units, args.hidden_units],
                              dropout_rate=args.dropout_rate, is_training=is_training, scope="self_attention",
                              reuse=reuse)

            seq *= mask
            # seq = normalize(seq)
    return seq


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
