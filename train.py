import argparse
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import torchvision
import numpy as np
import cv2 as cv

import dataset
from TrackNet import TrackNet

if "ipykernel" in sys.modules:  # executed in a jupyter notebook
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def wbce_loss(output, target):
    return -(
        ((1-output)**2) * target *
        torch.log(torch.clamp(output, min=1e-15, max=1)) +
        output**2 * (1-target) *
        torch.log(torch.clamp(1-output, min=1e-15, max=1))
    ).sum()
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None, help='Path to initial weights the model should be loaded with. If not specified, the model will be initialized with random weights.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint, chekpoint differs from weights due to including information about current loss, epoch and optimizer state.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size of the training dataset.')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size of the validation dataset.')
    parser.add_argument('--shuffle', type=bool, default=True, help='Should the dataset be shuffled before training?.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--train_size', type=float, default=0.8, help='Training dataset size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--dataset', type=str, default='dataset/', help='Path to dataset.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda, mps).')
    parser.add_argument('--type', type=str, default='auto', help='Type of dataset to create (auto, image, video). If auto, the dataset type will be inferred from the dataset directory, defaulting to image.')
    parser.add_argument('--save_period', type=int, default=10, help='Save checkpoint every x epochs (disabled if <1).')
    parser.add_argument('--save_weights_only', type=bool, default=False, help='Save only weights, not the whole checkpoint')
    parser.add_argument('--save_path', type=str, default='weights/', help='Path to save checkpoints at.')
    parser.add_argument('--use_tensorboard', type=bool, default=False, help='Use tensorboard to log training progress.')
    parser.add_argument('--one_output_frame', type=bool, default=False, help='Demand only one output frame instead of three.')
    parser.add_argument('--grayscale', type=bool, default=False, help='Use grayscale images instead of RGB.')
    opt = parser.parse_args()
    print(opt.one_output_frame)
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    device = torch.device(opt.device)
    model = TrackNet(one_output_frame=opt.one_output_frame, grayscale=opt.grayscale).to(device)
    writer = SummaryWriter('runs/tracknet_experiment_1')
    loss_function = torch.nn.HuberLoss()
    # loss_function = wbce_loss

    if opt.weights:
        model.load_state_dict(torch.load(opt.weights))
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    if opt.type == 'auto':
        full_dataset = dataset.GenericDataset.from_dir(opt.dataset, one_output_frame=opt.one_output_frame)
    elif opt.type == 'image':
        full_dataset = dataset.ImagesDataset(opt.dataset,  one_output_frame=opt.one_output_frame)
    elif opt.type == 'video':
        full_dataset = dataset.VideosDataset(opt.dataset, one_output_frame=opt.one_output_frame)

    train_size = int(opt.train_size * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle)
    val_loader = DataLoader(test_dataset, batch_size=opt.val_batch_size, shuffle=opt.shuffle)
    

    if opt.use_tensorboard:
        images, heatmaps = next(iter(train_loader))
        imgs = [images[0,3*i:3*(i+1),:,:] for i in range(3)]
        htms = [heatmaps[0,i,:,:].unsqueeze(0) for i in range(3)]
        imgs = torch.cat([img.unsqueeze(0) for img in imgs])
        htms = torch.cat([htm.unsqueeze(0) for htm in htms])
        writer.add_images('example image sequence', imgs)
        writer.add_images('example heatmaps sequence', htms)
    
    ### Try single batch
    print('Overfitting on a single batch.')
    X,y = next(iter(train_loader))
    if opt.grayscale:
        X_grayscale = [torchvision.transforms.functional.rgb_to_grayscale(X[:,3*i:3*(i+1),:,:]) for i in range(3)]
        X = torch.cat(X_grayscale, axis=1)
        
    X, y = X.to(device), y.to(device)
    print('Error with zeros: ', loss_function(torch.zeros_like(y), y))
    if opt.one_output_frame:
        print(opt.one_output_frame)
        save_image(y[0,:,:], f'results/overfitting_ya.png')
        save_image(X[0,:,:], f'results/overfitting_xa.png')
    else:
        save_image(X[0,:,:], f'results/overfitting_xa.png')
        save_image(y[0,0,:,:], f'results/overfitting_ya-0.png')
        save_image(y[0,1,:,:], f'results/overfitting_ya-1.png')
        save_image(y[0,2,:,:], f'results/overfitting_ya-2.png')
    
    for epoch in range(1, opt.epochs+1):
        print("Epoch: ", epoch)
        model.train()
        
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()

            
        model.eval()
        # Validation
        with torch.inference_mode():
            y_pred = model(X)
            # loss = wbce_loss(y_pred, y)
            # loss = torch.nn.functional.mse_loss(y_pred, y)
            loss = loss_function(y_pred, y)
            if opt.one_output_frame:
                save_image(y_pred[0,:,:], f'results/overfitting_ypred{epoch}.png')
            else:
                save_image(y_pred[0,0,:,:], f'results/overfitting_ypred{epoch}-0.png')
                save_image(y_pred[0,1,:,:], f'results/overfitting_ypred{epoch}-1.png')
                save_image(y_pred[0,2,:,:], f'results/overfitting_ypred{epoch}-2.png')
            print(loss)
    
    print('Finished overfitting on single batch.')
            
            
    ### Training loop
    for epoch in range(1, opt.epochs+1):
        print("Epoch: ", epoch)
        running_loss = 0.0
        model.train()
        
        for batch_idx, (X, y) in enumerate(tqdm(train_loader)):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()

            # running loss calculation
            running_loss += loss.item()
            
            if batch_idx % 10 == 9:
                model.eval()
                writer.add_scalar('training loss', running_loss / (batch_idx+1), (epoch-1) * len(train_loader) + batch_idx)
            
        model.eval()
        # Validation
        with torch.inference_mode():
            for batch_idx, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = wbce_loss(y_pred, y)
                print(loss)
                break
        
        # save the model
        if (epoch) % opt.save_period == 0:
            save_path = (Path(opt.save_path) / str(datetime.now() + f"_epoch:{epoch}")).with_suffix('.pth')
            if opt.save_weights_only:
                print('\n--- Saving weights to: ', save_path)
                save_path = save_path.name+"_weights_only".with_suffix('.pth')
                torch.save(model.state_dict(), save_path)   
            else:
                print('\n--- Saving checkpoint to: ', save_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, save_path)