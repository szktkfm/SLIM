import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import ElasticNet, Ridge, Lasso


class Evaluater():
    
    
    def __init__(self, user_item_test_df, user_num):
        self.user_num = user_num
        self.user_items_dict = self.user_aggregate_item(user_item_test_df)
            
    def user_aggregate_item(self, df):
        user_items_dict = {}
        #for user in user_list:
        for i in range(self.user_num):
            items_df = df[df['reviewerID'] == i]
            user_items_dict[i] = list(items_df['asin'])
        return user_items_dict
    
    def topn_precision(self, sorted_idx, target_user_id, n=10):

        if len(self.user_items_dict[target_user_id]) == 0:
            return 2
        
        topn_idx = sorted_idx[:n]   
        #print(topn_idx)
        #print(user_items_test_dict[target_user_id])
        hit = len(set(topn_idx) & set(self.user_items_dict[target_user_id]))
    
        #precision = hit / len(self.user_items_dict[target_user_id])
        precision = hit / n
        # precision_sum += precision
                
        return precision

    def topn_map(self, sorted_idx, user):
        mean_avg_pre = 0
        if len(self.user_items_dict[user]) == 0:
            return 2

        precision_sum = 0
        for j in self.user_items_dict[user]:
            n = list(sorted_idx).index(j) + 1
            precision = self.topn_precision(sorted_idx, user, n)
            precision_sum += precision
        
        return precision_sum / len(self.user_items_dict[user])
    
    
    def topn_recall(n=10):
        return 0