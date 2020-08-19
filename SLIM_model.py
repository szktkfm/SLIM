import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from multiprocessing import Pool
import multiprocessing as multi
from joblib import Parallel, delayed

#from pyglmnet import GLM


class SLIM():
    
    
    def __init__(self, alpha, l1_ratio, user_num, item_num, lin_model='elastic',):
        if lin_model == 'lasso':
            self.reg = Lasso(alpha=alpha, positive=True)
        elif lin_model == 'elastic':
            self.reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True)
            #self.reg = GLM(distr='gaussian', alpha=l1_ratio, reg_lambda=alpha)
            
        self.user_num = user_num
        self.item_num = item_num

            
    def fit(self, user_item_train_df):
        # rating_mat
        self.row = np.array([r[0] for r in user_item_train_df.values], dtype=int)
        self.col = np.array([r[1] for r in user_item_train_df.values], dtype=int)
        self.data = np.ones(len(user_item_train_df), dtype=int)
        self.rating_mat = csr_matrix((self.data, (self.row, self.col)), shape = (self.user_num, self.item_num))
        
        # linear modelを解く
        sim_mat = []
        for i in range(self.item_num):
            X = self.del_col(i)
            y = self.rating_mat[:, i]
    
            self.reg.fit(X.toarray(), y.toarray())
            w = np.insert(self.reg.coef_, i, 0)[:,  np.newaxis]
            sim_mat.append(w)
    
            #if i > 1:
            #    break

        self.sim_mat = np.concatenate(sim_mat, axis=1)


    def fit_multi(self, user_item_train_df):
        # rating_mat
        self.row = np.array([r[0] for r in user_item_train_df.values], dtype=int)
        self.col = np.array([r[1] for r in user_item_train_df.values], dtype=int)
        self.data = np.ones(len(user_item_train_df), dtype=int)
        self.rating_mat = csr_matrix((self.data, (self.row, self.col)), shape = (self.user_num, self.item_num))
        
        # linear modelを解く
        sim_mat = Parallel(n_jobs=-1)([delayed(self.solve_lin_model)(n) for n in range(self.item_num)])
        self.sim_mat = np.concatenate(sim_mat, axis=1)

        
    def solve_lin_model(self, idx):
        X = self.del_col(idx)
        y = self.rating_mat[:, idx]
        self.reg.fit(X.toarray(), y.toarray())
        w = np.insert(self.reg.coef_, idx, 0)[:,  np.newaxis]
        return w
        

    def del_col(self, col_idx):
        row_new = self.row[self.col != col_idx]
        col_new = self.col[self.col != col_idx]
        col_new[col_new > col_idx] = col_new[col_new > col_idx] - 1
        data_new = self.data[self.col != col_idx]
    
        return csr_matrix((data_new, (row_new, col_new)), shape = (self.user_num, self.item_num-1))

    
    def load_sim_mat(self, path, user_item_train_df):
        # rating mat
        self.row = np.array([r[0] for r in user_item_train_df.values], dtype=int)
        self.col = np.array([r[1] for r in user_item_train_df.values], dtype=int)
        self.data = np.ones(len(user_item_train_df), dtype=int)
        self.rating_mat = csr_matrix((self.data, (self.row, self.col)), shape = (self.user_num, self.item_num))
        
        self.sim_mat = np.loadtxt(path)

    
    def save_sim_mat(self, path):
        np.savetxt(path, self.sim_mat)

        
    def predict(self):
        pred_mat = np.dot(self.rating_mat.toarray(), self.sim_mat)
        self.rec_mat = pred_mat - self.rating_mat

    def pred_ranking(self, user_id):
        # あるユーザの予測ランキングを返す
        #rec_mat = self.pred_mat - self.rating_mat
        row_user = self.rec_mat[user_id, :]
        #print(row_user)
        rec_item_idx = np.argsort(row_user)[::-1]

        return np.array(rec_item_idx)[0, :]