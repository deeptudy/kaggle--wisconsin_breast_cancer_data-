import pandas as pd

def preprocess(df:pd.DataFrame):
    
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Drop unnecessary variable: unnamed: 32
    df_y = df['diagnosis']
    df_X = df.drop(columns=['Unnamed: 32', 'diagnosis', 'id'])

    return df_X, df_y



from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif(df:pd.DataFrame):
    """calculate_vif
    Args:
        df: pd.DataFrame (df_X)
    """
    variables = df.columns
    vif_data = pd.DataFrame()
    vif_data["Variable"] = variables
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data


from statsmodels.stats.outliers_influence import variance_inflation_factor
# code source: https://beckmw.wordpress.com/2013/02/05/collinearity-and-stepwise-vif-selection/

def X_filter_multicollinearity(X, thresh=10.0):
    """X_filter_multicollinearity
    Repeat folllowing process until every VIF < 10: 
        1) Remove a variable with the highest VIF
        2) Calculate VIF
    Args:
        X: pd.DataFrame(df_X)
        thresh: default = 10 (VIF thresh usually 5 or 10)
    """
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
            for ix in range(X.iloc[:, variables].shape[1])]
        maxloc = vif.index(max(vif))
    
        if max(vif) > thresh:
            print('==> [Dropped variable] : ' + X.iloc[:, variables].columns[maxloc])
            del variables[maxloc]
            
            if len(variables) > 1:
                dropped = True
    print('[Remaining variables] :')
    print(X.columns[variables])

    return variables