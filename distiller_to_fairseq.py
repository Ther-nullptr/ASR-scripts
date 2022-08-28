"""
convet a s3prl distiller model to fairseq model
"""

import sys
import torch
import s3prl.optimizers

distill_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/distill_data2vec_new.pt'
original_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/audio_base_ls.pt'
new_model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/distiller_data2vec.pt'
sys.modules["optimizers"] = s3prl.optimizers
distill_model = torch.load(distill_model_path)

print('distill model')
for key in distill_model['Distiller'].keys():
    print(key, distill_model['Distiller'][key].shape)
original_model = torch.load(original_model_path) 

print('hubert pretrain model')
for key in original_model['model']:
    if(hasattr(original_model['model'][key], 'shape')):
        print(key, original_model['model'][key].shape)
original_model['model'] = distill_model['Distiller']

original_model['cfg']['model']['encoder_layers'] = 2

torch.save(original_model, new_model_path)
print(f'save model to {new_model_path}')

