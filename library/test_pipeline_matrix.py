from library.utilities import *
from sklearn.preprocessing import *
from sklearn.svm import LinearSVC
from sklearn.ensemble import *
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
import pandas as pd


def block_performances(day, visit, patient, pipeline, x_full=None, y_full=None, df=None, block_size=20):
    if df is None:
        df = pd.DataFrame()  # Create a new dataframe, if not append to it
    if x_full is None:
        x_full, y_full = load_data(day=day, visit=visit, patient=patient, whole=True)
    for n_questions in range(block_size, x_full.shape[0] + 1 - block_size, block_size):
        x_train = x_full[:n_questions, :]
        x_test = x_full[n_questions:, :]
        y_train = y_full[:n_questions]
        y_test = y_full[n_questions:]
        pipeline.fit(x_train, y_train)
        df = df.append({'Day': day,
                        'Train size': n_questions,
                        'Test size': x_full.shape[0] - n_questions,
                        'Score on train': pipeline.score(x_train, y_train),
                        'Score on test': pipeline.score(x_test, y_test),
                        'visit': visit,
                        'patient': patient,
                        },
                       ignore_index=True)
    return df


def test_features(x_full, y_full, day='day label', visit='visit label', patient='patient label', train_step=10):
    # Models definition
    extra_tree = ExtraTreesClassifier(bootstrap=True, oob_score=False, criterion="gini",
                                      max_features=0.9, min_samples_leaf=2, min_samples_split=9, n_estimators=1000,
                                      n_jobs=4)
    extra_tree_grid = ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                           max_depth=100, max_features=0.1, max_leaf_nodes=None,
                                           min_impurity_decrease=0.0, min_impurity_split=None,
                                           min_samples_leaf=1, min_samples_split=2,
                                           min_weight_fraction_leaf=0.0, n_estimators=1500, n_jobs=1,
                                           oob_score=False, random_state=None, verbose=0, warm_start=False)
    random_forest = RandomForestClassifier(bootstrap=True, oob_score=False, criterion="gini",
                                           max_features=0.9, min_samples_leaf=2, min_samples_split=9, n_estimators=1000,
                                           n_jobs=4)

    svm_l2 = LinearSVC(penalty='l2', C=1.7, loss='hinge')
    svm_l2_sq = LinearSVC(penalty='l2', C=1.7, loss='squared_hinge')
    svm_l1 = LinearSVC(penalty='l1', dual=False)
    model_features_selector_extra = SelectFromModel(estimator=extra_tree, threshold='0.9*imp')

    pipe_norm_extra = make_pipeline(StandardScaler(), extra_tree)
    pipe_extra_svm = make_pipeline(model_features_selector_extra, StandardScaler(), svm_l2)
    pipe_svm_l2 = make_pipeline(StandardScaler(), svm_l2)
    pipe_svm_l2_sq = make_pipeline(RobustScaler(), svm_l2_sq)
    pipe_svm_l1 = make_pipeline(StandardScaler(), svm_l1)

    # %%
    # Computing Data
    df_extra_grid = pd.DataFrame()
    df_extra = pd.DataFrame()
    df_extra_svm = pd.DataFrame()
    df_forest = pd.DataFrame()
    df_svm_l2 = pd.DataFrame()
    df_svm_l2_sq = pd.DataFrame()
    df_svm_l1 = pd.DataFrame()

    print('start')
    df_extra_grid = block_performances(day=day, visit=visit, patient=patient, pipeline=extra_tree_grid,
                                       df=df_extra_grid, block_size=train_step, x_full=x_full, y_full=y_full)
    print('extra1')
    df_extra = block_performances(day=day, visit=visit, patient=patient, pipeline=extra_tree, block_size=train_step,
                                  df=df_extra, x_full=x_full, y_full=y_full)
    print('svm')

   # df_extra_svm = block_performances(day=day, visit=visit, patient=patient, pipeline=pipe_extra_svm,
                                    #  block_size=train_step, df=df_extra_svm, x_full=x_full, y_full=y_full)
    df_svm_l2 = block_performances(day=day, visit=visit, patient=patient, pipeline=pipe_svm_l2, block_size=train_step,
                                   df=df_svm_l2, x_full=x_full, y_full=y_full)
    print('svm_l2')
    df_svm_l2_sq = block_performances(day=day, visit=visit, patient=patient, pipeline=pipe_svm_l2_sq,
                                      block_size=train_step, df=df_svm_l2_sq, x_full=x_full, y_full=y_full)
    df_svm_l1 = block_performances(day=day, visit=visit, patient=patient, pipeline=pipe_svm_l1, block_size=train_step,
                                   df=df_svm_l1, x_full=x_full, y_full=y_full)
    df_forest = block_performances(day=day, visit=visit, patient=patient, pipeline=random_forest, block_size=train_step,
                                   df=df_forest, x_full=x_full, y_full=y_full)
    print('forest')

    # %% Plotting
    df_extra.loc[:, 'pipeline'] = 'ExtraTree'
   # df_extra_svm.loc[:, 'pipeline'] = 'Extra + SVM'
    df_svm_l2.loc[:, 'pipeline'] = 'SVM L2 Hinge'
    df_svm_l2_sq.loc[:, 'pipeline'] = 'SVM L2 Squared Hinge'
    df_svm_l1.loc[:, 'pipeline'] = 'SVM L1'
    df_forest.loc[:, 'pipeline'] = 'Random Forest'
    df_extra_grid.loc[:, 'pipeline'] = 'ExtraTree Grid'

    full_df_svm = pd.concat((
        df_svm_l1,
        df_svm_l2_sq,
        df_svm_l2
    ))

    full_df_tree = pd.concat((
        df_extra,
        df_forest,
        df_extra_grid
    ))

    plot_block_perf(full_df_tree, 'Tree Models')
    plot_block_perf(full_df_svm, 'Support Vector Machines Models')
#    plot_block_perf(df_extra_svm, 'Extra Tree + SVM')
