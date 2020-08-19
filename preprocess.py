import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from joblib import Parallel, delayed


def mk_user_item_id(i, user_item_values):
    row = user_item_values[i]
    user = user_list.index(row[0])
    item = item_list.index(row[1])
    return [user, item]


def user_item_id(user_item_values, user_list, item_list):
    # user_itemをID化
    count = 0
    user_item_list = []

    # ここを並列化
    #user_item_len = len(user_item_values)
    #user_item_list = Parallel(n_jobs=-1)([delayed(mk_user_item_id)(i, user_item_values) for i in range(500)])

    for row in user_item_values:
        user = user_list.index(row[0])
        item = item_list.index(row[1])
        user_item_list.append([user, item])

        #count += 1
        #if count > 1000:
        #    break

    df = pd.DataFrame(np.array(user_item_list),
                                columns = ['reviewerID', 'asin'])
    return df


def func():
    # データ読み込み
    user_item_df_no_id = pd.read_csv('./user_item.csv')
    item_list = list(set(list(user_item_df_no_id['asin'])))
    user_list = list(set(list(user_item_df_no_id['reviewerID'])))
    print('item size: {}'.format(len(item_list)))
    print('user size: {}'.format(len(user_list)))
    # 保存
    with open('./data/user_list.txt', 'w') as f:
        for user in user_list:
            f.write(user + '\n')
    #np.savetxt('user_list.txt', np.array(user_list))
    with open('./data/item_list.txt', 'w') as f:
        for item in item_list:
            f.write(item + '\n')
            

    # user-itemをID化
    user_item_df = user_item_id(user_item_df_no_id.values, user_list, item_list)

    # train-testスプリット
    user_item_df = user_item_df.take(np.random.permutation(len(user_item_df)))
    train_num = int(0.5 * len(user_item_df))
    user_item_train_df = user_item_df[0:train_num]
    user_item_test_df = user_item_df[train_num:]

    print('train {}'.format(train_num))
    print('test {}'.format(len(user_item_test_df)))
    # スプリットを保存
    user_item_train_df.to_csv('./data/user_item_train.csv', index=False)
    user_item_test_df.to_csv('./data/user_item_test.csv', index=False)



if __name__ == '__main__':
    s = time.time()
    func()
    runtime = time.time() - s
    print(runtime)