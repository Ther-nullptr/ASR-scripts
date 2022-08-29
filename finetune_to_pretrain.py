'''
load weight from a finetuned model to a pretrained model
'''

import torch

finetune_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/data2vec-finetune/audio_base_ls_100h.pt'
pretrain_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/audio_base_ls.pt'
new_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/finetune_to_pretrain/data2vec_finetune_to_pretrain.pt'

# load all keys from w2v_encoder.w2v_model in finetuned model to pretrain model

finetune_model = torch.load(finetune_model_path)
pretrain_model = torch.load(pretrain_model_path)

str_length = len('w2v_encoder.w2v_model.')

for key in finetune_model['model'].keys():
    if 'w2v_encoder.w2v_model' in key:
        pretrain_model['model'][key[str_length:]] = finetune_model['model'][key]
        print(f'[maintain] convert {key} to {key[str_length:]}')
    else:
        print(f'[ignore] ignore {key}')

torch.save(pretrain_model, new_model_path)
print(f'save new model to {new_model_path}')