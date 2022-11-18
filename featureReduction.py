from sklearn.decomposition import TruncatedSVD
import pandas as pd

def SVD(A, n_components=15):

    trun_svd =  TruncatedSVD(n_components)
    A_transformed = trun_svd.fit_transform(A)
    
    A_transformed = pd.DataFrame(A_transformed)
    
    return A_transformed

