import torch

from DATA_PROCESS.data_loading import Image_Dataset
from torch.utils.data import DataLoader
from MODELS.model_params import Params
from training.trainer import Trainer
from DATA_PROCESS.data_transformation import transformations
from MODELS.VGG_Transfer_Model import Vgg19Classifier
from torchvision.models import vgg19

import warnings
warnings.filterwarnings('ignore')



def main():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    data_transforms = transformations()
    train_ds = Image_Dataset(datafile = 'C:/Users/ankit/Music/suryanamaskar/datasets/train_dataset.csv',transform=data_transforms['train'],target_transform=data_transforms['label_transforms'])
    train_dl = DataLoader(train_ds,batch_size=16,pin_memory=True)

    valid_ds = Image_Dataset(datafile = 'C:/Users/ankit/Music/suryanamaskar/datasets/valid_dataset.csv',transform=data_transforms['validation'],target_transform=data_transforms['label_transforms'])
    valid_dl = DataLoader(valid_ds,batch_size=1,pin_memory=True)

    model = Vgg19Classifier()    
    model.to(device)

    print(model)
    


    '''
    neu_net = CNN()
    neu_net.to(device)
    '''
    hyper_params = Params(model)
    optimizer = hyper_params.optimizer_()
    loss_func = hyper_params.loss_func()
    epochs = 30
    
    trainer = Trainer(epochs,train_dl,valid_dl,device,optimizer,model,loss_func,len(train_ds),len(valid_ds))
    trainer.train()
    



if __name__ == "__main__":
    main()
