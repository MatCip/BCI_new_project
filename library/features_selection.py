from utilities import *
from sklearn.feature_selection import *
from sklearn.svm import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV
import pandas as pd
import pickle
import time

# Load dataset 6
[X, y, order] = extract_dataset('data/DataDay3/')
n = 30
X_tr, X_te, y_tr, y_te = get_experimental_sequence(X, y, order, n)

# %%
svm = LinearSVC(penalty='l2')
svm_lin = SVC(kernel='linear')

extra_tree = ExtraTreesClassifier(bootstrap=True, oob_score=False, criterion="gini",
                                  max_features=0.9, min_samples_leaf=2, min_samples_split=9, n_estimators=1000,
                                  n_jobs=-1)

random_forest = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.9, min_samples_leaf=2,
                                       min_samples_split=9, n_estimators=10000, n_jobs=-1)

# Select features with Recursive Features Elimination
recursive_features_selector_svm_lin = RFECV(estimator=svm_lin, step=1, cv=10, n_jobs=-1, verbose=0)

# Select features from Model
model_features_selector_extra_2e5 = SelectFromModel(estimator=extra_tree, threshold='2.5*mean')
model_features_selector_forest = SelectFromModel(estimator=random_forest, threshold='2.5*mean')
model_features_selector_lasso = SelectFromModel(estimator=LassoCV(cv=10, normalize=True, n_jobs=-1))

pipe_extra_svm_2e5 = make_pipeline(model_features_selector_extra_2e5, StandardScaler(), svm)

pipe_forest_svm = make_pipeline(model_features_selector_forest, StandardScaler(), svm)

pipe_lasso_svm = make_pipeline(model_features_selector_lasso, StandardScaler(), svm)

pipe_recursive_forest = make_pipeline(StandardScaler(), recursive_features_selector_svm_lin, random_forest)

# %%


x_full = X
y_full = y

df = pd.DataFrame()
for n in range(20, x_full.shape[0] - 1):
    x_train, x_test, y_train, y_test = get_experimental_sequence(x_full, y_full, order, n)

    before = time.time()
    x_train_feat = model_features_selector_extra_2e5.fit_transform(x_train, y_train)
    x_test_feat = model_features_selector_extra_2e5.transform(x_test)
    feat_selected = x_train_feat.shape[1]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_feat)
    x_test_scaled = scaler.transform(x_test_feat)

    svm.fit(x_train_scaled, y_train)
    train_time = time.time() - before
    before = time.time()
    sc = svm.score(x_test_scaled, y_test)
    score_time = time.time() - before
    print({'train_size': n, 'test_size': x_full.shape[0] - n, 'features_selected': feat_selected, 'score': sc,
           'train_time': train_time, 'score_time': score_time})
    df = df.append({'train_size': n, 'test_size': x_full.shape[0] - n, 'features_selected': feat_selected, 'score': sc,
               'train_time': train_time, 'score_time': score_time}, ignore_index=True)
# %%
print(df)
with open('dataframe_extra_svm.pkl', 'wb') as f:
    pickle.dump(df, f)
