import torchvision.datasets.hmdb51
import transforms as T
import torch
from torch.utils.data import DataLoader
from data_loader import CUSTOM_DATASET
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_dataset(dataset_name, dataset_cfg, mode, device):
    
    video_path = dataset_cfg[dataset_name]['video_path']
    annotation_path = dataset_cfg[dataset_name]['annotation_path']
    num_classes = dataset_cfg[dataset_name]['num_classes']
    classInd_path = dataset_cfg[dataset_name]['classInd_path']

    num_frames=dataset_cfg['Hyperparams']['num_frames']
    clip_steps=dataset_cfg['Hyperparams']['clip_steps']
    num_workers=dataset_cfg['Hyperparams']['num_workers']
    batch_size=dataset_cfg['Hyperparams']['batch_size']
    shuffle= dataset_cfg['Hyperparams']['shuffle']

    if device=='cpu': pin_memory=False
    else: pin_memory=True

    if mode == 'train':
        train_transforms = torchvision.transforms.Compose([
                                                    T.ToFloatTensorInZeroOne(),
                                                    T.Resize((128, 128)),
                                                    T.CenterCrop((112, 112)),
                                                    T.RandomHorizontalFlip(),
                                                    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                    ])

        train_dataset =CUSTOM_DATASET(dataset_name, video_path, annotation_path, classInd_path,num_frames,step_between_clips=clip_steps, fold=1, train=True, transform=train_transforms, num_workers=num_workers)

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        trainset, valset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        return [train_loader, val_loader], num_classes

    elif mode == 'test':
        test_transforms = torchvision.transforms.Compose([T.ToFloatTensorInZeroOne(), T.Resize((112, 112)),
                                                    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        test_dataset = CUSTOM_DATASET(dataset_name,video_path, annotation_path, classInd_path,num_frames,step_between_clips=clip_steps, fold=1, train=False, transform=test_transforms, num_workers=num_workers)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return test_loader, num_classes
    