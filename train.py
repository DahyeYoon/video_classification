import timeit
from datetime import datetime
import os
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable

def trainer(dataset, model, hyperparams, resume_epoch, use_tb, device):

    root_abspath = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    model_name = model.__get_model_name__()
    dataset_name=dataset[0].dataset.dataset.__get_dataset_name__()

    log_save_dir = os.path.join(root_abspath, 'runs', model_name+"_"+dataset_name)
    checkpoint_dir = os.path.join(root_abspath, 'checkpoints', model_name +'_'+dataset_name)
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    train_dataloader = dataset[0]
    val_dataloader = dataset[1]

    lr = hyperparams['lr']
    nEpochs = hyperparams['nEpochs']
    save_epoch = hyperparams['save_epoch']
    momentum = hyperparams['momentum']
    scheduler_type = hyperparams['scheduler']
    weight_decay=hyperparams['weight_decay']
    step_size=hyperparams['step_size']
    gamma=hyperparams['gamma']
    resume_epoch=hyperparams['resume_epoch']

    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if scheduler_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                            gamma=gamma)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    

    if resume_epoch == 0:
        print(">>> Train {} model from scratch on {} dataset".format(model_name, dataset_name))
    else:
        print(">>> Train from epoch {}".format(resume_epoch))
        ckpt_name=os.path.join(checkpoint_dir, 'epoch_'+ str(resume_epoch - 1)+ '.pth')
        checkpoint = torch.load(ckpt_name, map_location=lambda storage, loc: storage)   

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    model.to(device)
    criterion.to(device)

    if use_tb:

        writer = SummaryWriter(log_dir=os.path.join(log_save_dir, datetime.now().strftime('%b%d_%H-%M-%S')+'.local'))


    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        running_loss = 0.0
        corrects=0

        model.train()


        for i, data in enumerate(tqdm(train_dataloader)):
            inputs, labels = data[0].to(device), data[-1].to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            corrects += (preds == labels).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                        .format(epoch+1, nEpochs, i+1, len(train_dataloader), loss.item()))

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = 100. * corrects/ len(train_dataloader.dataset)

        if use_tb:
            writer.add_scalar('Loss/train_loss_per_epoch', epoch_loss, epoch)
            writer.add_scalar('Acc/train_acc_per_epoch', epoch_acc, epoch)
        


        print(">> Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, 'epoch_' + str(epoch) + '.pth'))
            print("Save checkpoint\n")
        val(val_dataloader, model, epoch, criterion, use_tb, writer, device)

    if use_tb:
        writer.close()




def val(val_dataloader, model, epoch, criterion, use_tb, writer, device):
 
    running_loss = 0.0
    corrects = 0
    model.eval()

    for i, data in enumerate(tqdm(val_dataloader)):

        inputs, labels = data[0], data[-1]
        inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device)

        with torch.no_grad():
            outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        corrects += (preds == labels).sum().item()


        running_loss += loss.item()
        corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_dataloader)
    epoch_acc = 100.0 * corrects / len(val_dataloader.dataset)

    if(use_tb):
        writer.add_scalar('Loss/val_loss_per_epoch', epoch_loss, epoch)
        writer.add_scalar('Acc/val_acc_per_epoch', epoch_acc, epoch)
        writer.close()