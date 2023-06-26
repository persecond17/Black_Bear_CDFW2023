from torch.nn import DataParallel
import sys
from DL_train_bi import *


def evaluate(model, test_data, device, pp_terms_0):
    test = RoBERTa_Dataset(test_data)
    test_dataloader = DataLoader(test, batch_size=32, shuffle=False) # should not shuffle

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    predictions = []
    test_labels = []

    with torch.no_grad():

        for test_ori_texts, test_input, test_label in test_dataloader:
            test_labels += list(test_label)
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            # post-processing
            predictions_test = output.argmax(dim=1).clone()
            for i in range(len(predictions_test.tolist())):
                text = test_ori_texts[i].lower()
                for term in pp_terms_0:
                    preprocessed_text = preprocessing(text)
                    if term in preprocessed_text:
                        predictions_test[i] = 0
                        break

            acc = (predictions_test == test_label).sum().item()
            total_acc_test += acc
            predictions.extend(predictions_test.cpu().tolist())

    test_acc = total_acc_test / len(test)
    f1 = f1_score(test_labels, predictions, average='weighted') # 'macro'
    cf = confusion_matrix(test_labels, predictions)
    print('Test Confusion Matrix:', '\n', cf)
    return test_acc, f1, predictions


def main():
    model_name = sys.argv[1]
    model_t = RoBERTa_Classifier()
    state_dict = torch.load(f'dl-classification/{model_name}')
    try:
        model_t.load_state_dict(state_dict)
    except:
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('linear2', 'linear').split('module.')[1]
            new_state_dict[new_key] = state_dict[key]
        model_t.load_state_dict(new_state_dict)

    model_t = DataParallel(model_t)
    test_data = pd.read_csv('dl_results_bi/test_df.csv')
    test_data.loc[test_data['Label_basic'] == 2, 'Label_basic'] = 0  # Set it as binary
    # test = test_data
    test = data_aug(skipwords, test_data, 1, 1)

    test_acc, f1_score, predictions = evaluate(model_t, test, device, pp_terms_0)
    print(f'Model name: {model_name}')
    print(f'Test size: {len(test_data)}')
    print(f'Test accuracy: {test_acc:.3f}')
    print(f'Test F1 score: {f1_score:.3f}')


if __name__ == '__main__':
    main()
