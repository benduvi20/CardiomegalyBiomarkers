import pandas as pd


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