from sklearn.ensemble import VotingClassifier
from sklearn.model_selection._split import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import clone, Parallel, delayed, \
    _fit_and_predict
from itertools import product
import numpy as np


class PredefinedClassifier:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        return self.predictions

    def predict_proba(self, X):
        return self.predictions


class VotingClassifierCV(VotingClassifier):
    def __init__(self, estimators, voting='hard', weights=2, n_jobs=1,
                 flatten_transform=None,
                 cv=None, scoring='accuracy'):
        """
        cv and scoring - like in GridSearchCV
        weights - None, or int, or list of numbers (possible weights),
            or list of lists of numbers (weights combinations)
        The rest of parameters - like in VotingClassifier
        """
        super(VotingClassifierCV, self).__init__(estimators, voting=voting,
                                                 weights=weights,
                                                 n_jobs=n_jobs,
                                                 flatten_transform=flatten_transform)
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y, sample_weight=None):
        # fitting everything except weights
        original_weights = self.weights
        # have to set self.weights because VotingClassifier.fit checks them
        self.weights = np.ones(len(self.estimators))
        # fit to the full data
        super(VotingClassifierCV, self).fit(X, y, sample_weight=sample_weight)
        estimators = self.estimators_
        self.weights = original_weights

        # generate cross_validated predictions for each classifier
        cv = check_cv(self.cv)
        scoring = check_scoring(self, self.scoring)
        parallel = Parallel(n_jobs=self.n_jobs)
        method = 'predict_proba' if self.voting == 'soft' else 'predict'
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        verbose = False
        preds = []
        for name, est in self.estimators:
            prediction_blocks = parallel(
                delayed(_fit_and_predict)(
                    clone(est), X, y, train, test, verbose, fit_params, method)
                for train, test in cv.split(X, y))
            preds.append([pred for pred, _ in prediction_blocks])

        # prepare sample weights and targets for each fold
        test_targets = []
        test_weights = []
        for train, test in cv.split(y):
            test_targets.append(y[test])
            if sample_weight:
                test_weights.append(sample_weight[test])
            else:
                test_weights.append(None)

        # recreate list of possible weights
        weights_array = np.array(self.weights)
        if len(weights_array.shape) == 2:
            self.weigths_seq_ = self.weights
        else:
            if len(weights_array.shape) == 0:
                weight_wec = np.arange(self.weights)
            else:  # assume it is 1d
                weight_wec = self.weights
            clf_len = len(self.estimators)
            self.weigths_seq_ = [x for x in product(*[weight_wec] * clf_len)
                                 if sum(x) > 0]
        # score the classifier at different weights
        scores = []
        for weights_vector in self.weigths_seq_:
            self.weights_ = weights_vector
            cv_scores = []
            for fold, pred_vectors in enumerate(zip(*preds)):
                self.estimators_ = [PredefinedClassifier(pred_vector)
                                    for pred_vector in pred_vectors]
                test_y = test_targets[fold]
                test_x = None
                test_w = test_weights[fold]
                cv_scores.append(scoring(self, test_x, test_y, test_w))
            scores.append(cv_scores)
        self.scores_ = np.array(scores)

        # choose the best weight
        self.weights_ = self.weigths_seq_[
            np.argmax(np.mean(self.scores_, axis=1))]
        self.estimators_ = estimators

    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators.
        Note that it uses self.weigths_ instead of self.weights.
        """
        if self.weights_ is None:
            return None
        return [w for est, w in zip(self.estimators,
                                    self.weights_) if est[1] is not None]

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    print('Running the example of VotingClassifierCV')
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    clf_list = [('lr', LogisticRegression()), ('dt', DecisionTreeClassifier())]
    v = VotingClassifierCV(clf_list,
                           weights=[[0, 1], [1, 2], [1, 1], [2, 1], [1, 0]],
                           cv=6, voting='soft', scoring='accuracy')
    v.fit(X, y)
    print('mean scores:')
    print(v.scores_.mean(axis=1))
    print('corresponding weights:')
    print(v.weigths_seq_)
    print('chosen weights:')
    print(v.weights_)
