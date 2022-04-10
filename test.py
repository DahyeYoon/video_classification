import timeit
from datetime import datetime
from tqdm import tqdm
from torch import nn, optim
import torch

def test(test_dataloader, model, device):
        model.eval()
        start_time = timeit.default_timer()
       
        with torch.no_grad():
                corrects = 0
                for data in tqdm(test_dataloader):
                        inputs, labels = data[0].to(device), data[-1].to(device)
                        
                        outputs = model(inputs)
                        _, preds = torch.max(outputs.data, 1)
                        corrects += (preds == labels).sum().item()


        # acc = 100. * corrects/ len(test_dataloader.dataset)

        print('Accuracy of the model on the test images: {} %'.format(100. * corrects/ len(test_dataloader.dataset)))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

def demo(data_loader, model,device):
        pass