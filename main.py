import argparse
import torch
from dataset import get_dataset
from models.c3d import C3D
from train import trainer
from test import demo, test
import yaml
from torchinfo import summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for the video classification module')
    parser.add_argument('--model_name', type=str, default='C3D', help='model name to be used [3dcnn | lstm]')
    parser.add_argument('--run_mode', type=str, help='mode to run the module [train | test | demo]', default='train')
    parser.add_argument('--tb', action='store_true', help='whether to use tensorboard or not')
    parser.add_argument('--dataset_name', type=str, default='UCF101', help='dataset name to input')
    parser.add_argument('--task', type=str, required=False, help='optional task')
    parser.add_argument('--pretrained_model_path', type=str, required=False, help='path of pretrained model, if not provided, will train from scratch')
    parser.add_argument('--resume_epoch', type=int, default=int(0), help='resume from epoch')
    args = parser.parse_args()
    print(args)

    with open('./config.yaml', "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ----- DataLoader
    data_loader, num_classes = get_dataset(args.dataset_name, cfg['Datasets'], args.run_mode, device)
    # ----- Model
    # args.pretrained_model_path = './c3d-pretrained.pth'
    model =C3D(num_classes, pretrained=args.pretrained_model_path)
    summary(model, input_size=(cfg['Datasets']['Hyperparams']['batch_size'], 3, cfg['Datasets']['Hyperparams']['num_frames'], cfg['Datasets']['Hyperparams']['height'], cfg['Datasets']['Hyperparams']['width']))
    # ---- Train
    args.tb=True

    # ------- RUN
    if args.run_mode == 'train':
        trainer(data_loader, model, cfg['Hyperparams'], args.resume_epoch, args.tb, device)
    elif args.run_mode == 'test':
        test(data_loader,model,device)
    elif args.run_mode == 'demo':
        demo(data_loader, model,device)

    print('Done!')
