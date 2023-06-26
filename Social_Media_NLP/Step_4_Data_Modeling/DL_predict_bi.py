import sys
from DL_train_bi import *
from DL_test_bi import *

l = int(sys.argv[2])
r = int(sys.argv[3])
save_path = sys.argv[4]

label = 1 # filter bear encounter related tweets
new = pd.read_csv('dl-classification/filtered.csv')
temp = new.iloc[l:r].copy()
temp['merged'] = temp['content'].apply(preprocessing)
temp['Label_basic'] = len(temp)*[-1]

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
acc, f1_score, output = evaluate(model_t, temp, device, pp_terms_0)
temp['Label_basic'] = output

print(f'Model name: {model_name}')
print(f'Data range: {l} - {r}')
print(f'Class - {label}')
print(temp['Label_basic'].value_counts())

final = temp[temp['Label_basic'] == label]
if save_path.endswith('.txt'):
    for index, row in final.iterrows():
        print(index, row['content'])
else:
    final.to_csv(f'dl_results_bi/{model_name}-{l}-{r}.csv')
    print('Done!')