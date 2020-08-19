import SLIM_model
import evaluate
import pandas as pd
import numpy as np
import pickle
import time

from importlib import reload
import optuna

# データロード
data_dir = './data/'
user_item_train_df = pd.read_csv(data_dir + 'user_item_train.csv')
user_item_test_df = pd.read_csv(data_dir + 'user_item_test.csv')
user_list = []
item_list = []
with open(data_dir + 'user_list.txt', 'r') as f:
    for l in f:
        user_list.append(l.replace('\n', ''))
        
with open(data_dir + 'item_list.txt', 'r') as f:
    for l in f:
        item_list.append(l.replace('\n', ''))



def time_since(runtime):
    mi = int(runtime / 60)
    sec = runtime - mi * 60
    return (mi, sec)


lin_model = 'elastic'
def objective(trial):
    start = time.time()
    # define model and fit
    alpha = trial.suggest_loguniform('alpha', 1e-6, 1)
    l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
    #lin_model = trial.suggest_categorical('lin_model', ['lasso', 'elastic'])
    
    model = SLIM_model.SLIM(alpha, l1_ratio, len(user_list), len(item_list), lin_model=lin_model)
    #model.fit(user_item_train_df)
    model.fit_multi(user_item_train_df)
    #model.load_sim_mat('./sim_mat.txt', user_item_train_df)

    # evaluate
    eval_model = evaluate.Evaluater(user_item_test_df, len(user_list))
    ## predict
    model.predict()

    score_sum = 0
    not_count = 0
    for i in range(len(user_list)):
        rec_item_idx = model.pred_ranking(i)
        #score = eval_model.topn_precision(rec_item_idx, i)
        score = eval_model.topn_map(rec_item_idx, i)
        if score > 1:
            not_count += 1
            continue
        score_sum += score

    mi, sec = time_since(time.time() - start)
    print('{}m{}sec'.format(mi, sec))

    return -1 * (score_sum / (len(user_list) - not_count))


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=20)

    df = study.trials_dataframe() # pandasのDataFrame形式
    df.to_csv('./result/hyparams.csv')
    # save best params 
    with open('./result/best_param.pickle', 'wb') as f:
        pickle.dump(study.best_params, f)