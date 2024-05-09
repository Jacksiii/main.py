
# For any questions, don't hesitate to contact me: Ding Zou (m202173662@hust.edu.cn)
#变动测试02
import os
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from time import time
from prettytable import PrettyTable
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from utils.parser import parse_args
from utils.data_loader import load_data
from modules.MCCLK import Recommender
from utils.evaluate import test
from utils.helper import early_stopping
import csv

import logging
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

# 增加一个函数来保存每个样本的损失到CSV文件
#def save_sample_losses(epoch, batch_idx, losses, labels, filename):
#    file_path = os.path.join(results_dir, filename)
#    file_exists = os.path.isfile(file_path)
#    with open(file_path, mode='a', newline='') as file:
#        writer = csv.writer(file)
#        if not file_exists:
#            writer.writerow(["Epoch", "Batch Index", "Sample Index", "Loss", "Label"])
#        for i, loss in enumerate(losses):
#            writer.writerow([epoch, batch_idx, i, loss.item(), labels[i].item()])


# 确保结果保存的目录存在
results_dir = "mcclkexperiment"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 增加一个函数来保存实验结果到CSV文件
def save_results_to_csv(results, filename):
    import csv
    file_path = os.path.join(results_dir, filename)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "Training Time", "Testing Time", "Loss", "Test AUC", "Test F1"])
        writer.writerow(results)

# 增加一个函数来保存图形
def save_plots(history, base_filename):
    plt.figure(figsize=(10, 8))
    plt.plot(history['epoch'], history['loss'], label='Loss')
    plt.plot(history['epoch'], history['auc'], label='Test AUC')
    plt.plot(history['epoch'], history['f1'], label='Test F1')
    plt.title('Model Performance Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{base_filename}.png"))
    plt.close()

def get_feed_dict(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict

def get_feed_dict_topk(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict
    # def negative_sampling(user_item, train_user_set):
    #     neg_items = []
    #     for user, _ in user_item.cpu().numpy():
    #         user = int(user)
    #         while True:
    #             neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
    #             if neg_item not in train_user_set[user]:
    #                 break
    #         neg_items.append(neg_item)
    #     return neg_items
    #
    # feed_dict = {}
    # entity_pairs = train_entity_pairs[start:end].to(device)
    # feed_dict['users'] = entity_pairs[:, 0]
    # feed_dict['pos_items'] = entity_pairs[:, 1]
    # feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
    #                                                             train_user_set)).to(device)

def _show_recall_info(recall_zip):
    res = ""
    for i, j in recall_zip:
        res += "K@%d:%.4f  "%(i,j)
    logging.info(res)

def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user, item, 0])
    return np.array(res)

def _get_user_record(data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def ctr_eval(model, data):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:

        batch = get_feed_dict(data, start, start + args.batch_size)
        labels = data[start:start + args.batch_size, 2]
        _, scores, _, _ = model(batch)
        scores = scores.detach().cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
        start += args.batch_size
    model.train()
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1

def topk_eval(model, train_data, data):
    # logging.info('calculating recall ...')
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    item_set = set(train_data[:, 1].tolist() + data[:, 1].tolist())
    train_record = _get_user_record(train_data, True)
    test_record = _get_user_record(data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        np.random.seed()
        user_list = np.random.choice(user_list, size=user_num, replace=False)

    model.eval()
    for user in user_list:
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size]
            input_data = _get_topk_feed_data(user, items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            _, scores, _, _ = model(batch)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            _, scores, _, _ = model(batch)
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
    model.train()

    recall = [np.mean(recall_list[k]) for k in k_list]
    return recall
    # _show_recall_info(zip(k_list, recall))

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    # train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_cf], np.int32))
    # eval_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in eval_cf], np.int32))
    # LongTensor
    # labels = torch.FloatTensor(np.array(cf[2] for cf in train_cf))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    # 数据和图形的存储路径
    date_str = datetime.now().strftime("%Y%m%d")
    base_filename = f"results_{date_str}"
    csv_filename = f"{base_filename}.csv"
    history = {'epoch': [], 'loss': [], 'auc': [], 'f1': []}

    print("start training ...")
    for epoch in range(args.epoch):
        """training CF"""
        #total_loss = 0  # 在这里初始化 total_loss
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf = train_cf[index]

        """training"""
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf, s, s + args.batch_size)
            batch_loss, _, _, _ = model(batch)
            #loss_mean, batch_loss, _, _, _ = model(batch)  # 获取均值损失和每个样本的损失
            # batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            #sample_losses = batch_loss.detach().cpu().numpy()  # 将Tensor转换为NumPy数组
            labels = batch['labels'].cpu().numpy()
            #save_sample_losses(epoch, s // args.batch_size, sample_losses, labels, 'sample_losses.csv')

            #total_loss += loss_mean.item() * args.batch_size
            loss += batch_loss
            s += args.batch_size

        train_e_t = time()
        # tsne_plot(model.all_embed, epoch)
        # if epoch % 10 == 9 or epoch == 1:
        if 1:
            """testing"""
            test_s_t = time()
            # ret = test(model, user_dict, n_params)
            test_auc, test_f1 = ctr_eval(model, test_cf_pairs)
            test_e_t = time()
            # ctr_info = 'epoch %.2d  test auc: %.4f f1: %.4f'
            # logging.info(ctr_info, epoch, test_auc, test_f1)
            # 记录数据
            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['auc'].append(test_auc)
            history['f1'].append(test_f1)

            # 保存到CSV
            save_results_to_csv([epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_auc, test_f1], csv_filename)
            train_res = PrettyTable()
            # train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision",
            #                          "hit_ratio", "auc"]
            # train_res.add_row(
            #     [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'],
            #      ret['precision'], ret['hit_ratio'], ret['auc']]
            # )
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "test auc", "test f1"]
            train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_auc, test_f1]
                )
            # train_res.field_names = ["Recall@5", "Recall@10", "Recall@20", "Recall@50", "Recall@100"]
            # train_res.add_row([recall[0], recall[1], recall[2], recall[3], recall[4]])
            print(train_res)

    # 保存图形
    save_plots(history, base_filename)
    print('early stopping at %d, test_auc:%.4f' % (epoch-30, cur_best_pre_0))
