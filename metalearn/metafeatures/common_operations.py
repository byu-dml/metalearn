from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def replace_nominal_column(col):
    """
    Returns a One Hot Encoded ndarray of col
    """
    labelledCol = LabelEncoder().fit_transform(col)
    labelledCol = labelledCol.reshape(labelledCol.shape[0],1)
    return OneHotEncoder().fit_transform(labelledCol).toarray()

def get_column_of_class(data, columnIndex, label):
    """
    Returns all values of the column, for which the row matches the class label.
    """
    return data[:,columnIndex][data[:,-1] == label]
