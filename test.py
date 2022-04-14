import timeit
from tqdm import tqdm
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import transforms as T

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

        print('Accuracy of the model on the testset: {} %'.format(100. * corrects/ len(test_dataloader.dataset)))
        end_time = timeit.default_timer()
        print("Execution time: " + str(end_time - start_time) + "\n")

def demo(video_path,class_to_idx, model,device):
        
        transforms=T.DemoTransforms(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(112, 112))
        cap=cv2.VideoCapture(video_path)

        font =cv2.FONT_HERSHEY_DUPLEX
        color = (0, 255, 0)
        thickness = 1
        fontScale = 0.3

        w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')

        delay = int(1000 / fps) # 1s = 1000ms

        out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
        frames=[]
        frame_queue=[]
        while(cap.isOpened()):
                ret, frame=cap.read()
                if ret:
                        # convert bgr to rgb
                        frame_queue.append(frame)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        if(len(frames)==16):
                                trans_frames=transforms(torch.tensor(np.array(frames),dtype=torch.uint8))
                                model.eval()
                                with torch.no_grad():
                                        inputs = trans_frames.to(device)
                                        outputs = model(inputs)
                                        _, pred = torch.max(outputs.data, 1)
                                        prob=torch.max(F.softmax(outputs, dim=1)[0])
                                class_list=list(class_to_idx.keys())
                                pred_class=class_list[list(class_to_idx.values()).index(pred.tolist()[0])]
                                output_text='Predicted class: '+pred_class+" ("+str(round(float(prob), 4))+")"
                                del frames[0]
                        # show frames
                        if 'pred' in locals():
                                while len(frame_queue)>0:
                                        target_frame=frame_queue.pop(0)
                                        result_img = cv2.putText(target_frame, output_text, (10 , 15), font,
                                                        fontScale, color, thickness, cv2.LINE_AA)
                                        cv2.imshow('Video classification demo', result_img)
                                        out.write(result_img)
                                        if cv2.waitKey(delay) & 0xFF == ord('q'):
                                                break
                else:
                        break
        cap.release()
        cv2.destroyAllWindows()