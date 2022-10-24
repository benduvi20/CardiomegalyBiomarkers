import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, callback
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score


def SplitData(df, splitFracs, class_col = 'class'):
    '''Split dataframe into smaller subsets which keep constant ratio of postive/negative samples. 
    Function retuns a list of DataFrames, where each new subset an element of list.  

    :param df: Dataframe to split
    :type df: pd.DataFrame
    :param splitFracs: list of floats which sum to 1 and relative size of subsets (splits) 
    :type splitFracs: list
    :param class_col: string to indicate which column defines binary classification, default - 'class'
    :type class_col: string

    :return: list of DataFrames of subsets
    :rtype: list of length len(splitFracs)
    '''

    # suffle df into random order
    df.sample(frac=1).reset_index(drop=True)

    # split df by class
    class1 = df.loc[df[class_col]==1]
    class0 = df.loc[df[class_col]==0]

    # initiate output list
    split_list = []

    # cycle through all split fractions and 
    for i in range(len(splitFracs)):
        
        # define split indexes for pos (class 1) samples
        start_class1 = round(sum(splitFracs[:i])*len(class1))
        end_class1 = round(sum(splitFracs[:(i+1)])*len(class1))

        # define split indexes for neg (class 0) samples
        start_class0 = round(sum(splitFracs[:i])*len(class0))
        end_class0 = round(sum(splitFracs[:(i+1)])*len(class0))        

        # collect pos and neg samples of split from class1 and class0 dataframes
        split_class1 = class1.iloc[start_class1:end_class1]
        split_class0 = class0.iloc[start_class0:end_class0]

        # concatenate pos and neg samples of split into new DataFrame
        split = pd.concat([split_class1, split_class0]).sample(frac=1).reset_index(drop=True)

        split_list.append(split)

    return split_list



def get_xgboost(feature_type, model_params):
    '''retreive model skeleton in form of xgboost.XGBClassifier object with model_param features 
    encoded. If using only biomarker features, few features are present reuireing to change max tree depth.

    :param features: string scribding types of features used to train model
    :type features: str
    :param model_params: dict of model parameters
    :type model_params: dict

    :return: model sekeleton with parameters encoded
    :rtype: xgboost.XGBClassifier object
    '''

    if feature_type == 'BMRK':
        # get model -> since using few features (2), use shallow max_depth
        model = XGBClassifier(eval_metric = model_params["eval_metric"], scale_pos_weight = model_params["scale_pos_weight"], 
                              colsample_bytree = model_params["colsample_bytree"], gamma = model_params["gamma"], subsample = model_params["subsample"], 
                              max_depth = model_params["max_depth_shallow"], learning_rate = model_params["lr"], 
                              callbacks=[callback.EvaluationMonitor(show_stdv=False), callback.EarlyStopping(model_params["early_stopping"])])

    else:
        # get model -> since using more features, use deeper max_depth
        model = XGBClassifier(eval_metric = model_params["eval_metric"], scale_pos_weight = model_params["scale_pos_weight"], 
                              colsample_bytree = model_params["colsample_bytree"], gamma = model_params["gamma"], subsample = model_params["subsample"], 
                              max_depth = model_params["max_depth_deep"], learning_rate = model_params["lr"], 
                              callbacks=[callback.EvaluationMonitor(show_stdv=False),callback.EarlyStopping(model_params["early_stopping"])])

    return model




def test_xgboost(model, test, features):
    '''Test xgboost model on 'test' and return preformance scores of accuracy, 
    AUC ROC, F1 score, and confusion matrix

    :param model: trained XGBoost model to be tested
    :type model: xgboost.XGBClassifier object
    :param test: DataFrame of test samples with features and ground truth classes
    :type test: pd.DataFrame
    :param features: list of features used in this model
    :type features: list

    :return accuracy: accuracy of predictions
    :rtype accuracy: float
    :return auc_roc: AUC ROC score of predictions
    :rtype auc_roc: float
    :return f1: F1 score of predictions
    :rtype f1: float
    :return cf: confusion matrix of predictions
    :rtype cf: ndarray of shape (2, 2)
    '''
    
    # make predictions on test set and make binary
    preds_raw = model.predict(test[features], ntree_limit=model.best_ntree_limit)
    preds = [round(value) for value in preds_raw]

    # evaluate preductions
    accuracy = accuracy_score(list(test['class']), preds)
    cf = confusion_matrix(list(test['class']), preds)
    auc_roc = roc_auc_score(list(test['class']), preds)
    f1 = f1_score(list(test['class']), preds)

    return accuracy, auc_roc, f1, cf




def train_test_xgboost(train_folds, val, valFoldNum, test, modalities_combinations, model_params, model_path, lossFigure, exportModels):
    '''Train xgboost models on train_folds with validaiotn set val, for differnt modality combinations. Then, if requested, 
    export loss figure and models to model_path folder. Subsequentally test model on test set and return DataFrame of summary
    performance scores.

    :param train_folds: list of k-1 DataFrame (each is one fold of data), in case where there are k folds which are combined into training set
    :type train_folds: list of pd.DataFrame's
    :param val: DataFrame of fold used for validation set
    :type val: pd.DataFrame
    :param valFoldNum: indicator of which fold is used for validation
    :type valFoldNum: int
    :param test: Dataframe of test set
    :type test: pd.DataFrame
    :param modalities_combinations: sublists of 2 elements containing: [list of features used in modality combination corresponding to second element, str of modality combiation]
    :type modalities_combinations: list of 2 element lists
    :param model_params: dictinoary descirbing xgboost model parameters
    :type model_params: dict
    :param model_path: file path to model and figure save folder
    :type model_path: str
    :param lossFigure: bool value to determine is training/validation loss figure is saved to model_path folder
    :type lossFigure: bool
    :param exportModels: bool value to determine if model is exported to model_path folder
    :type exportModels: bool

    :return results_df: DataFrame of preformance scores on test set for different modality combinations 
    :rtype results_df: pd.DataFrame 
    '''

    # combine training folds into single training set
    train = pd.concat(train_folds, axis=0).reset_index(drop=True)

    results = []
    for _, combination in enumerate(modalities_combinations):

        # match features to targetoutputs
        eval_set = [(train[combination[0]], train['class']), (val[combination[0]], val['class'])]

        # get model skeleton
        model = get_xgboost(combination[1], model_params)

        # train model using validation set and call back methods to get best model
        model.fit(train[combination[0]], train['class'], eval_set = eval_set)
        
        # if lossFigure, save figure with trianing and validation losses to model_path
        if lossFigure:
            # retrieve performance metrics
            trainval_error = model.evals_result()
            epochs = len(trainval_error['validation_0']['logloss'])
            x_axis = range(0, epochs)

            # plot log loss
            fig, ax = plt.subplots()
            ax.plot(x_axis, trainval_error['validation_0']['logloss'], label='Train Error')
            ax.plot(x_axis, trainval_error['validation_1']['logloss'], label='Val Error')
            ax.legend()
            plt.ylabel('Log Loss')
            plt.title('XGBoost Log Loss for ' + combination[1])
            plt.savefig(model_path + combination[1] + '_fold' + str(valFoldNum) + '_loss.jpg')

        # if exportModels, export models to model_path 
        if exportModels:
            model.save_model(model_path + combination[1] + '_fold' + str(valFoldNum) + '_model.json')


        # get predictions and evaluate
        [accuracy, auc_roc, f1, cf] = test_xgboost(model, test, combination[1])
        results.append([combination[1], accuracy, auc_roc, f1, cf])

    # return results of all modality combinations tested
    results_df = pd.DataFrame(results, columns=['Modalities', 'Accuracy', 'ROC AUC', 'F1 score','Confusion Matrix']).set_index('Modalities')

    return results_df