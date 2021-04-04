import numpy as np

import mlpn as mlp
from utils import load_test_set, index_to_lang, get_common_features_number, get_feature_index

STUDENT = {'name': 'Nadav Spitzer',
           'ID': '302228275',
           'name2': 'Lior Frenkel',
           'ID2': '204728315'
           }


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.

    # get the number of features (bigrams)
    feat_vec = np.zeros(get_common_features_number())

    # create a vector that represents the distribution of the features
    for feature in features:
        # for each feature - add 1 to the vector
        index = get_feature_index(feature)

        if index is None:  # skip unknown features
            continue
        feat_vec[index] += 1
    return feat_vec


def predict_on_test(params):
    with open('test.pred', 'w') as f:
        test_set = load_test_set('test', 'bigrams')

        for _, features in test_set:
            x = feats_to_vec(features)
            pred = mlp.predict(x, params)
            print(index_to_lang(pred), file=f)