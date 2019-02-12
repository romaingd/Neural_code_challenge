import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer


def classify(x_tr, y_tr, groups_tr, est, est_params=None,
             perform_cross_validation=False, cv_params=None,
             random_state=None):
    '''
    Classification wrapper, handling cross-validation and fitting
    '''
    assert((est_params is not None)
           | (perform_cross_validation & (cv_params is not None)))

    splitter = GroupShuffleSplit(n_splits=5, test_size=0.33,
                                 random_state=random_state)

    if perform_cross_validation:
        print('Cross-validating the following estimator:\n', est, '\n',
              'on the following parameters: %s' % list(cv_params.keys()), '\n')
        gscv = GridSearchCV(est, cv_params, verbose=1,
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

    else:
        train_idx, test_idx = next(splitter.split(x_tr, y_tr, groups_tr))

        X_train = x_tr[train_idx]           # Overloading the notations is
        X_test = x_tr[test_idx]             # not ideal, and should be avoided

        y_train = y_tr[train_idx]
        y_test = y_tr[test_idx]

        clf = est.set_params(**est_params)
        print('Fitting the following classifier:\n', clf, '\n')

        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # ROC curve
        from sklearn.metrics import roc_curve, auc
        y_train_score = clf.predict_proba(X_train)[:, 1]
        y_test_score = clf.predict_proba(X_test)[:, 1]

        print('Training score:', cohen_kappa_score(y_train, y_train_pred))
        print('Test score:', cohen_kappa_score(y_test, y_test_pred))

        print('Mean training prediction:', np.mean(y_train_pred))
        print('Mean test prediction:', np.mean(y_test_pred))

        print('\n')

    return(clf)