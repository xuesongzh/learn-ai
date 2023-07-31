import torch
import torchvision.models as models

'''
只保存模型 参数
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # we do not specify weights, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))

print(model)
print(list(model.state_dict().keys()))
# model.eval()
'''

#保存模型结构和参数
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model,'full_vgg16.pth')

model=torch.load('full_vgg16.pth')

for batch in dataloader:
    y=model(batch) # (64,1000,)
    