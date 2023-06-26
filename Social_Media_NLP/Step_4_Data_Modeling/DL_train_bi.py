import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import AdamW, SGD
from torch.nn import DataParallel
import transformers
from transformers import BertModel
from transformers import BertTokenizer
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.flow as naf
import visdom
from torchviz import make_dot
from tqdm import tqdm
from focal_loss.focal_loss import FocalLoss

import numpy as np
import pandas as pd
import random
import re
import nltk
import datetime
from datetime import date
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, train_test_split, StratifiedShuffleSplit
from transformers import BertTokenizer, RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset

today = date.today()

# Check if a GPU is available and use it, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocessing(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # mention_pattern = re.compile(r'@[^\s]+')
    punctuation_pattern = re.compile(r'[^\w\s\']')
    text = url_pattern.sub('', text)
    # text = mention_pattern.sub('', text)
    text = punctuation_pattern.sub('', text)
    text = text.replace('\n', ' ')
    return text.lower()


def data_aug(skipwords, train_data, class_, k):
    aug1 = naw.SynonymAug(aug_src='wordnet', aug_p=0.4, stopwords=skipwords)
    aug2 = nac.KeyboardAug(aug_char_min=1, aug_char_max=2, aug_word_min=1,
                           aug_word_max=4, min_char=4, stopwords=skipwords)

    aug_pipeline = naf.Sequential([aug1, aug2])

    # Augment training data that was labeled as class_
    augmented_texts = []
    augmented_labels = []
    one_texts = train_data.loc[train_data['Label_basic'] == class_]['content'].tolist()
    one_labels = train_data.loc[train_data['Label_basic'] == class_]['Label_basic'].tolist()
    for text, label in zip(one_texts, one_labels):
        for _ in range(k):
            augmented_text = aug_pipeline.augment(text)
            if augmented_text:
                augmented_texts.append(augmented_text[0])
            else:
                augmented_texts.append('')
            augmented_labels.append(label)

    # concatenate the augmented data with the origin data
    train_data = pd.concat([train_data, pd.DataFrame({'content': augmented_texts,
                                                      'Label_basic': augmented_labels})],
                           ignore_index=True)
    # shuffle the data
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return train_data


class RoBERTa_Dataset(Dataset):
    def __init__(self, df):
        self.labels = df['Label_basic'].tolist()
        self.ori_texts = df['content'].tolist()
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in df['content']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_ori_texts(self, idx):
        return self.ori_texts[idx]

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_ori_texts = self.get_batch_ori_texts(idx)
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_ori_texts, batch_texts, batch_y


class RoBERTa_Classifier(nn.Module):
    def __init__(self, dropout=.3):
        super(RoBERTa_Classifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)  # binary

    def forward(self, input_ids, mask):
        _, pooled_output = self.roberta(input_ids=input_ids, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = nn.functional.softmax(linear_output, dim=1)
        return final_layer


def train(train_data, val_data, learning_rate, weight_decay, epochs, device, patience, batch_size, pp_terms_0):
    gamma = 3
    weights = [1 - 0.85, 1 - 0.15]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = FocalLoss(gamma=gamma, weights=class_weights)

    # initialize visdom
    vis = visdom.Visdom()
    val_loss_win = vis.line(X=[0], Y=[0], opts=dict(title='Val Loss'))
    val_acc_win = vis.line(X=[0], Y=[0], opts=dict(title='Val Accuracy'))

    accuracies = []
    losses = []

    # train = train_data
    train = data_aug(skipwords, train_data, 1, 1)  # data augmentation for 1 time

    # val = val_data
    val = data_aug(skipwords, val_data, 1, 1)  # data augmentation for 1 time

    train_dataset = RoBERTa_Dataset(train)
    val_dataset = RoBERTa_Dataset(val)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    model_f = RoBERTa_Classifier().cuda()
    model_f = DataParallel(model_f)
    criterion = criterion.cuda()

    optimizer = AdamW(model_f.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    best_val_loss = float('inf')
    now = datetime.datetime.now()
    hour = now.hour
    minute = now.minute
    model_name = f'dl-classification/model_{today}_{hour}_{minute}_bi.pth'
    patience_counter = 0

    print(f'Model name: {model_name}')
    print(f'Weights: {weights}')
    print(f'Gamma: {gamma}')
    print(f'Epoches: {epochs}')
    print(f'Patience: {patience}')
    print(f'Batch size: {batch_size}')
    print(f'Training size: {len(train_data)}')
    print(f'Validation size: {len(val_data)}')
    print(f'Dropout ratio: 0.35')
    print(f'Learning rate: {learning_rate}')
    print(f'Weight decay: {weight_decay}')

    for epoch_num in tqdm(range(epochs), desc='Epoch'):
        total_acc_train = 0
        total_loss_train = 0
        y_true_train = []
        y_pred_train = []
        torch.cuda.empty_cache()

        model_f.train()

        for train_ori_texts, train_input, train_label in tqdm(train_dataloader, desc='Training', leave=False):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            model_f.zero_grad()
            output = model_f(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            y_true_train += train_label.tolist()
            y_pred_train += output.argmax(dim=1).tolist()

            batch_loss.backward()
            optimizer.step()

        scheduler.step()
        model_f.eval()

        total_acc_val = 0
        total_loss_val = 0
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for val_ori_texts, val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model_f(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                # post-processing
                predictions_val = output.argmax(dim=1).clone()  # Ensure we're working with a separate tensor
                for i in range(len(predictions_val.tolist())):
                    text = val_ori_texts[i].lower()
                    if any(term in preprocessing(text) for term in pp_terms_0):
                        predictions_val[i] = 0
                        break

                acc = (predictions_val == val_label).sum().item()
                total_acc_val += acc
                y_true_val += val_label.tolist()
                y_pred_val += predictions_val.tolist()

        train_f1_score = f1_score(y_true_train, y_pred_train, average='weighted')  # 'macro'
        val_f1_score = f1_score(y_true_val, y_pred_val, average='weighted')  # 'macro'

        train_loss = total_loss_train / len(train)
        train_acc = total_acc_train / len(train)
        val_loss = total_loss_val / len(val)
        val_acc = total_acc_val / len(val)
        val_cf = confusion_matrix(y_true_val, y_pred_val)

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {train_loss: .3f} | Train Accuracy: {train_acc: .3f} | Train F1 Score: {train_f1_score: .3f} | Val Loss: {val_loss: .3f} | Val Accuracy: {val_acc: .3f} | Val F1 Score: {val_f1_score: .3f}')
        print('Validation Confusion Matrix:', '\n', val_cf)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Validation loss did not reduce for {patience} consecutive epochs, stopping early')
            break

        if train_acc >= 0.99:
            print(f'Training accuracy has reached the threshold 0.99, stopping early')
            break

        accuracies.append(val_acc)
        losses.append(val_loss)
        # plot loss and accuracy curves
        vis.line(X=[epoch_num + 1], Y=[val_loss], win=val_loss_win, update='append')
        vis.line(X=[epoch_num + 1], Y=[val_acc], win=val_acc_win, update='append')

        if epoch_num % 5 == 0:
            torch.save(model_f.state_dict(), model_name)

    torch.save(model_f.state_dict(), model_name)
    mean_acc = sum(accuracies) / len(accuracies)
    std_acc = np.std(accuracies)

    mean_loss = sum(losses) / len(losses)
    std_loss = np.std(losses)

    print(f'Mean Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}')
    print(f'Mean Loss: {mean_loss:.3f} +/- {std_loss:.3f}')


skipwords = ['bear', 'BEAR', 'Bear']

df_1 = pd.read_excel('dl-classification/CDFW_labeled_1.xlsx')
df_2 = pd.read_excel('dl-classification/CDFW_labeled_2.xlsx')
df_3 = pd.read_excel('dl-classification/Xin_labeled_3.xlsx')
df_4 = pd.read_excel('dl-classification/Xin_labeled_4.xlsx')
df_5 = pd.read_excel('dl-classification/Xin_labeled_5.xlsx')
df_6 = pd.read_csv('dl-classification/Xin_labeled_6.csv')

df_temp = pd.concat([df_1, df_2, df_3, df_4, df_5])
df_temp = df_temp.drop_duplicates(subset=['content', 'county'])
df_temp = df_temp.drop(['Label_detail', 'index', 'tokenized_tweets',
                        'tokenized', 'merged', 'Unnamed: 0'], axis=1) \
                 .reset_index(drop=True)
indices = df_temp[df_temp['content'].isin(df_6['content'])].index
df_temp = df_temp.drop(indices)
df = pd.concat([df_temp, df_6])

# df['Label_detail'] = df['Label_detail'].apply(lambda x: 'a' if x == 'A' else x)
df = df[df['Label_basic'].notna()].reset_index(drop=True)

mini = df[['content', 'Label_basic']].copy()
mini['content'] = mini['content'].apply(lambda text: preprocessing(text))
mini.loc[mini['Label_basic'] == 2, 'Label_basic'] = 0  # Set it as binary

# perform stratified sampling
X = mini.drop('Label_basic', axis=1)
y = mini['Label_basic']

n_splits = 1
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=42)

for train_index, test_index in sss.split(X, y):
    # select the data and labels for the training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    for train_index2, val_index in sss.split(X_train, y_train):
        X_train, X_val = X_train.iloc[train_index2], X_train.iloc[val_index]
        y_train, y_val = y_train.iloc[train_index2], y_train.iloc[val_index]

        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        test_df.to_csv('dl_results_bi/test_df.csv')

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
labels = {'Not related to real bears': 0,
          'Related to bears but not encounters': 1,
          'Related to bear encounters': 2}

EPOCHS = 30
learning_rate = 5e-6
weight_decay = 1e-4  # regularization
patience = 5
batch_size = 64
pp_terms_0 = ['big bear', 'in big bear', 'to big bear', 'big bear city', 'hungry bear', 'usgs', 'earthquake', 'quake',
              'bear river', 'bear valley', 'bear mountain', 'papa bear', 'brother bear', 'yoga', 'yogi', 'like a bear',
              'pooh bear', 'diner', 'smokey', 'golden bear', 'restaurant', 'boo bear', 'market', 'song', 'bar', 'toy',
              'bear crk', 'bear st', 'bear vly', 'crossfit', 'bear crawl', 'bear defense', 'koala', 'bear season',
              'nola bear', 'n67ff', 'bear fire', 'disney', 'hulu', 'bear basin', 'polar bear plunge', 'honey bear',
              'bearcreek', ]


# pp_terms_1 = ['see a bear', 'saw a bear', 'seen a bear', 'scene a bear', 'hit a bear', 'bear sighting', 'bear attack',
# 'chased by a bear', 'bear spotting']

def main():
    train(train_df, val_df, learning_rate, weight_decay, EPOCHS, device, patience, batch_size, pp_terms_0)


if __name__ == '__main__':
    main()
