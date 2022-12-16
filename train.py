import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    device = torch.device(opt.device)
    model = TrackNet().to(device)

    if opt.weights:
        model.load_state_dict(torch.load(opt.weights))
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    if opt.type == 'auto':
        full_dataset = dataset.GenericDataset.from_dir(opt.dataset)
    elif opt.type == 'image':
        full_dataset = dataset.ImagesDataset(opt.dataset)
    elif opt.type == 'video':
        full_dataset = dataset.VideosDataset(opt.dataset)

    train_size = int(opt.train_size * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle)
    val_loader = DataLoader(test_dataset, batch_size=opt.val_batch_size, shuffle=opt.shuffle)
    
    ### Training loop
    for epoch in range(1, opt.epochs+1):
        print("Epoch: ", epoch)
        model.train()
        
        # Training
        for batch_idx, (X, y) in enumerate(tqdm(train_loader)):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = wbce_loss(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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