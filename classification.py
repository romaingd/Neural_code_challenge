import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer, accuracy_score
from sklearn.base import clone


def classify(est, x_tr, y_tr, groups_tr, x_te=None, test_index=None,
             perform_evaluation=True,
             perform_cross_validation=False, cv_params=None,
             compute_submission=False, submission_path='',
             random_state=None):
    '''
    Classification wrapper, handling cross-validation and fitting
    '''
    if compute_submission:
        assert((x_te is not None)
               & (test_index is not None)
               & (submission_path is not ''))

    if perform_cross_validation:
        assert(cv_params is not None)
        assert(not perform_evaluation)

    splitter = GroupShuffleSplit(n_splits=5, test_size=0.33,
                                 random_state=random_state)

    if perform_cross_validation:
        print('Cross-validating the following estimator:\n', est, '\n',
              'on the following parameters: %s' % list(cv_params.keys()), '\n')
        gscv = GridSearchCV(est, cv_params, verbose=1, n_jobs=4,
                            scoring=make_scorer(cohen_kappa_score),
                            cv=list(splitter.split(x_tr, y_tr, groups_tr)))
        gscv.fit(x_tr, y_tr)
        print('Best parameters: %r' % (gscv.best_params_))
        means = gscv.cv_results_['mean_test_score']
        stds = gscv.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print('\n')
        clf = gscv.best_estimator_
    
    if perform_evaluation:
        train_idx, test_idx = next(splitter.split(x_tr, y_tr, groups_tr))

        X_train = x_tr[train_idx]           # Overloading the notations is
        X_test = x_tr[test_idx]             # not ideal, and should be avoided

        y_train = y_tr[train_idx]
        y_test = y_tr[test_idx]

        clf = clone(est)
        print('Fitting the following classifier:\n', clf, '\n')
        clf.fit(X_train, y_train)
        evaluate_clf(clf, X_train, y_train, X_test, y_test)

    if compute_submission:
        clf = clone(est)
        clf.fit(x_tr, y_tr)
        y_te_pred = clf.predict(x_te)
        y_te_pred_df = pd.DataFrame(data=y_te_pred, columns=['TARGET'], index=test_index)
        y_te_pred_df.index.name = 'ID'
        y_te_pred_df.to_csv(submission_path, header=True, index=True)

    return(clf)


def evaluate_clf(clf, X_train, y_train, X_test, y_test):
    '''
    Evaluate a classifier on various metrics, on the training and test sets
    '''
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print('Training score:', cohen_kappa_score(y_train, y_train_pred))
    print('Test score:', cohen_kappa_score(y_test, y_test_pred))

    print('Training accuracy:', accuracy_score(y_train, y_train_pred))
    print('Test accuracy:', accuracy_score(y_test, y_test_pred))

    print('Mean training prediction:', np.mean(y_train_pred))
    print('Mean test prediction:', np.mean(y_test_pred))

    print('\n')


def find_best_threshold(y_proba, y_true):
    best_thr, best_score = 0, 0
    step = 0.01
    for thr in np.arange(0, 1+step, step):
        score = cohen_kappa_score((y_proba > thr) * 1, y_true)
        if score > best_score:
            best_score = score
            best_thr = thr
    print('Threshold: {:.3f} Score: {:.3f}'.format(best_thr, best_score))
    return(best_thr, best_score)