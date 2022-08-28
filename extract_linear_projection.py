"""
extract the linear projection from a finetuned model
"""

import torch

model_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/data2vec-finetune/audio_base_ls_100h.pt'
linear_projection_path = '/mnt/lustre/sjtu/home/xc915/superb/upstream_model/linear_projection/data2vec_linear_projection.pt'
model = torch.load(model_path)

class LinearProj(torch.nn.Module):
    #! now the size of linear projection is fixed: (768, 32)
    def __init__(self):
        self.linear = torch.nn.Linear(768, 32)

    def forward(self, x):
        x = self.linear(x)
        return x

linear_proj = LinearProj()
print(linear_proj.state_dict().keys())

linear_proj.linear.weight = model['model']['w2v_encoder.proj.weight']
linear_proj.linear.bias = model['model']['w2v_encoder.proj.bias']

torch.save(linear_proj, linear_projection_path)
print(f'save mdoel to {linear_projection_path}')