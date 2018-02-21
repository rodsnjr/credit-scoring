import utils
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer, roc_auc_score
import sys

def build_submission_skl(classifier):
    submission_feature_set = utils.load_test_feature_set()
    submission_feature_set = submission_feature_set.drop(['SeriousDlqin2yrs'], axis=1)
    x = submission_feature_set.as_matrix()

    y_predicts = classifier.predict(x)
    y_probs = classifier.predict_proba(x)
    
    submission_set = utils.load_test_set()
    submission_set .SeriousDlqin2yrs = y_predicts
    submission_set ['Probability'] = y_probs[:,:1]
    
    return submission_set

if __name__ == "__main__":
    features_set = utils.load_features_set()
    y = features_set['SeriousDlqin2yrs'].as_matrix()
    x = features_set.drop(['SeriousDlqin2yrs'], axis=1).as_matrix()
    x_train_f, x_test_f, y_train_f, y_test_f = utils.split_dataset(x, y, portion=0.10)

    print sys.argv[1:][0]
    max_size = sys.argv[1:][0] if sys.argv[1:][0] < x_train_f.shape[0] else x_train_f.shape[0]

    clf = GradientBoostingClassifier()
    cv_sets = ShuffleSplit(max_size, n_iter = 10, 
                        test_size = 0.1, train_size=None, 
                        random_state = 0)

    parameters = {
        'min_samples_split': [2,3,4,5],
        'max_depth': [4, 5, 6],
        'subsample': [0.6, 0.8, 1.0],
        'n_estimators' : [100, 200, 300, 391]
    }

    scorer = make_scorer(roc_auc_score)
    grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=scorer, cv=cv_sets, verbose=10, n_jobs=4)
    grid_obj = grid_obj.fit(x_train_f[:max_size], y_train_f[:max_size])
    
    clf = grid_obj.best_estimator_
    
    y_targets = clf.predict(x_test_f)
    roc_auc_score(y_targets, y_train_f)
    print("%s: %0.2f - [%s]" % ('ROC_AUC_SCORE', score, clf.__class__.__name__))