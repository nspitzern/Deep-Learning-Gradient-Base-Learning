from collections import Counter

STUDENT = {'name': 'Nadav Spitzer',
           'ID': '302228275',
           'name2': 'Lior Frenkel',
           'ID2': '204728315'
           }

F2I = dict()
L2I = dict()


def read_data(fname):
    data = []
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))
        return data


def get_common_features_number():
    if len(F2I) != 0:
        return len(F2I.keys())
    return 2  # for the xor problem


def get_labels_number():
    return len(L2I.keys())


def get_feature_index(feature):
    if len(F2I) != 0:
        return F2I.get(feature)
    return feature  # for XOR problem


def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]


def text_to_unigrams(text):
    return ["%s" % c1 for c1 in text]


def lang_to_index(dataset):
    for index in range(len(dataset)):
        lang = dataset[index][0]
        feats = dataset[index][1]

        dataset[index] = (L2I[lang], feats)

    return dataset


def index_to_lang(index):
    k = list(L2I.keys())
    v = list(L2I.values())

    l_idx = v.index(index)
    return k[l_idx]


def load_train_set(fname, features_type):
    data = read_data(fname)

    if features_type == 'bigrams':
        TRAIN = [(l,text_to_bigrams(t)) for l,t in data]
    else:
        TRAIN = [(l,text_to_unigrams(t)) for l,t in data]

    fc = Counter()
    for l, feats in TRAIN:
        fc.update(feats)

    # 700 most common bigrams in the training set.
    vocab = set([x for x, c in fc.most_common(700)])
    # vocab = set(fc.keys())

    return TRAIN, vocab


def load_dev_set(fname, features_type):
    data = read_data(fname)

    if features_type == 'bigrams':
        DEV = [(l,text_to_bigrams(t)) for l,t in data]
    else:
        DEV = [(l, text_to_unigrams(t)) for l, t in data]

    fc = Counter()
    for l, feats in DEV:
        fc.update(feats)

    # 600 most common bigrams in the training set.
    # vocab = set([x for x, c in fc.most_common(600)])

    vocab = set(fc.keys())

    return DEV, vocab


def load_test_set(fname, features_type):
    data = read_data(fname)

    if features_type == 'bigrams':
        TEST = [(l,text_to_bigrams(t)) for l,t in data]
    else:
        TEST = [(l, text_to_unigrams(t)) for l, t in data]

    return TEST




def load_data(train_fname, dev_fname, features_type):
    train_dataset, train_vocab = load_train_set(train_fname, features_type)
    dev_dataset, dev_vocab = load_dev_set(dev_fname, features_type)

    global L2I, F2I
    # label strings to IDs
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in train_dataset]))))}

    train_vocab.update(dev_vocab)

    # feature strings (bigrams) to IDs
    F2I = {f: i for i, f in enumerate(list(sorted(train_vocab)))}

    indexed_train = lang_to_index(train_dataset)

    indexed_dev = lang_to_index(dev_dataset)

    return indexed_train, indexed_dev






