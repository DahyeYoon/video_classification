import argparse
import torch
from dataset import get_dataset
from models.c3d import C3D
from train import trainer
from test import demo, test
import yaml
from torchinfo import summary

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arguments for the video classification module')
    parser.add_argument('--model_name', type=str, default='C3D', help='model name to be used [3dcnn | lstm]')
    parser.add_argument('--dataset_name', type=str, default='UCF101', help='dataset name to input')
    parser.add_argument('--run_mode', type=str, help='mode to run the module [train | test | demo]', default='train')
    parser.add_argument('--resume_epoch', type=int, default=int(0), help='resume from epoch')
    parser.add_argument('--tb', action='store_true', help='whether to use tensorboard or not')
    parser.add_argument('--pretrained_model_path', type=str, required=False, help='path of pretrained model, if not provided, will train from scratch')
    parser.add_argument('--video_path', type=str, required=False, help='path of video for the demo')
    parser.add_argument('--model_path', type=str, required=False, help='path of model to inference')
    parser.add_argument('--task', type=str, required=False, help='optional task')
    args = parser.parse_args()
    print(args)

    cfg=load_config('./config.yaml')
    
    device= 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------- RUN
    if args.run_mode == 'train':
        # ----- DataLoader
        data_loader, num_classes = get_dataset(args.dataset_name, cfg['Datasets'], args.run_mode, device)
        # ----- Model
        if args.pretrained_model_path is not None:
            model =C3D(num_classes).from_pretrained(args.pretrained_model_path)
        else:
            model =C3D(num_classes, init_weight=True)
        summary(model, input_size=(cfg['Datasets']['Hyperparams']['batch_size'], 3, cfg['Datasets']['Hyperparams']['num_frames'], cfg['Datasets']['Hyperparams']['height'], cfg['Datasets']['Hyperparams']['width']))
        # ---- Train
        trainer(data_loader, model, cfg['Hyperparams'], args.resume_epoch, args.tb, device)

    elif args.run_mode == 'test':
        # ----- DataLoader
        data_loader, num_classes = get_dataset(args.dataset_name, cfg['Datasets'], args.run_mode, device)
        # ----- Model
        model =C3D(num_classes)
        model.load_state_dict(torch.load(args.model_path))
        summary(model, input_size=(cfg['Datasets']['Hyperparams']['batch_size'], 3, cfg['Datasets']['Hyperparams']['num_frames'], cfg['Datasets']['Hyperparams']['height'], cfg['Datasets']['Hyperparams']['width']))
        test(data_loader,model,device)

    elif args.run_mode == 'demo':
        class_to_idx=torch.load('class_to_idx.pth')
        model=C3D(num_classes=len(class_to_idx))
        model.load_state_dict(torch.load(args.model_path))
        demo(args.video_path, class_to_idx, model,device)


