from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import torchvision
import json

import dataset
from TrackNet import TrackNet

from sys import modules
if "ipykernel" in modules:  # executed in a jupyter notebook
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


def euclidean_loss(output, target):
    return ((output-target)**2).sum()


def construct_my_loss(opt):
    def my_loss(output, target):
        diff = output - target
        diff[diff<0] *= opt.pos_factor
        return (diff**2).mean()
    return my_loss


class AdaptiveWingLoss(torch.nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, default=None, help='Path to initial weights the model should be loaded with. If not specified, the model will be initialized with random weights.')
    parser.add_argument('--project_name', type=str, default='tracknet', help='Wandb project name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint, chekpoint differs from weights due to including information about current loss, epoch and optimizer state.')
    parser.add_argument('--loss_function', type=str, default='my_loss', help='One of: {mse, euc, huber, wbce, l1, adwing, my_loss}')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size of the training dataset.')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size of the validation dataset.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs.')
    parser.add_argument('--train_size', type=float, default=0.8, help='Training dataset size.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate. If equals to 0.0, no dropout is used.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--sequence_length', type=int, default=3, help='Length of the images sequence used as X.')
    parser.add_argument('--pos_factor', type=int, default=2, help='How many times more important is to correctly predict a positive pixel (one including the ball) than a negative one.')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024], help='Size of the images used for training (y, x).')
    parser.add_argument('--dataset', type=str, default='dataset/', help='Path to dataset.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda, mps).')
    parser.add_argument('--type', type=str, default='auto', help='Type of dataset to create (auto, image, video). If auto, the dataset type will be inferred from the dataset directory, defaulting to image.')
    parser.add_argument('--checkpoint_period', type=int, default=1, help='Save checkpoint every x epochs (disabled if <1).')
    parser.add_argument('--log_period', type=int, default=100, help='Log to tensorboard/wandb every x batches.')
    parser.add_argument('--save_path', type=str, default='weights/', help='Path to save checkpoints at.')
    parser.add_argument('--images_dir', type=str, default='images/', help="Path to dataset's images.")
    parser.add_argument('--videos_dir', type=str, default='videos/', help="Path to dataset's videos.")
    parser.add_argument('--csvs_dir', type=str, default='csvs/', help="Path to dataset's csv files.")
    parser.add_argument('--save_weights_only', action='store_true', help='Save only weights, not the whole checkpoint')
    parser.add_argument('--include_dups', action='store_true', help='Allow for constructing sequences with frames already used in previous sequences.')
    parser.add_argument('--no_shuffle', action='store_true', help="Don't shuffle the training dataset.")
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard to log training progress.')
    parser.add_argument('--wandb', action='store_true', help='Use weights & biases to log training progress.')
    parser.add_argument('--one_output_frame', action='store_true', help='Demand only one output frame instead of three.')
    parser.add_argument('--save_images', action='store_true', help="Save output examples to results folder.")
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale images instead of RGB.')
    parser.add_argument('--single_batch_overfit', action='store_true', help='Overfit the model on a single batch.')
    opt = parser.parse_args()
    return opt


#TODO: add val accuracy
def training_loop(opt, device, model, writer, loss_function, optimizer, train_loader, val_loader, save_path):
    best_val_loss = float('inf')
    for epoch in range(opt.epochs):
        tqdm.write("Epoch: " + str(epoch))
        running_loss = 0.0

        model.train()
        pbar = tqdm(train_loader)
        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()

            # running loss calculation
            running_loss += loss.item()
            pbar.set_description(f'Loss: {running_loss / (batch_idx+1):.6f}')

            if batch_idx % opt.log_period == 0:
                with torch.inference_mode():
                    images = [
                        torch.unsqueeze(y[0,0,:,:], 0).repeat(3,1,1).cpu(),
                        torch.unsqueeze(y_pred[0,0,:,:], 0).repeat(3,1,1).cpu(),
                    ]
                    if opt.grayscale:
                        images.append(X[0,:,:,:].cpu())
                        res = X[0,:,:,:] * y[0,0,:,:]
                    else:
                        images.append(X[0,(2,1,0),:,:].cpu())
                        res = X[0, (2,1,0),:,:] * y[0,0,:,:]
                    images.append(res.cpu())
                    grid = torchvision.utils.make_grid(images, nrow=1)#, padding=2)
                    if opt.wandb:
                        wandb_grid = wandb.Image(grid, caption="Image, predicted output and ball mask")
                        wandb.log({
                            'train':{
                                'RunningLoss': running_loss / (batch_idx+1),
                                "ImageResult": wandb_grid,
                            } },
                            step = epoch * len(train_loader) + batch_idx
                        )
                    if opt.tensorboard:
                        writer.add_image('ImageResult', grid, epoch*len(train_loader) + batch_idx)
                        writer.add_scalar('RunningLoss/train', running_loss / (batch_idx+1), epoch * len(train_loader) + batch_idx)
                    if opt.save_images:
                        save_image(grid, f'results/epoch_{epoch}_batch{batch_idx}.png')


        if val_loader is not None:
            best = False
            val_loss = validation_loop(device, model, loss_function, val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best = True

            # save the model
            if epoch % opt.checkpoint_period == opt.checkpoint_period - 1:
                if opt.save_weights_only:
                    tqdm.write('\n--- Saving weights to: ' + str(save_path))
                    torch.save(model.state_dict(), save_path / 'last.pth')
                    if best:
                        torch.save(model.state_dict(), save_path / 'best.pth')
                else:
                    tqdm.write('\n--- Saving checkpoint to: ' + str(save_path))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        }, save_path / 'last.pt')
                    if best:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_loss,
                            }, save_path / 'best.pt')

            if opt.tensorboard:
                writer.add_scalars('Loss', {'train': running_loss / len(train_loader), 'val': val_loss}, epoch)
            if opt.wandb:
                wandb.log({
                    'train/Loss': running_loss / len(train_loader),
                    'val/Loss': val_loss,
                },
                    step = epoch
                )
                wandb.save(str(save_path/'best.pth'))
                wandb.save(str(save_path/'last.pth'))
                wandb.save(str(save_path/'best.pt'))
                wandb.save(str(save_path/'last.pt'))


def validation_loop(device, model, loss_function, val_loader):
    model.eval()
    loss_sum = 0
    with torch.inference_mode():
        for X, y in tqdm(val_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss_sum += loss_function(y_pred, y)
        tqdm.write('Validation loss: ' + str(loss_sum/len(val_loader)))

    return loss_sum/len(val_loader)


if __name__ == '__main__':
    opt = parse_opt()
    device = torch.device(opt.device)
    model = TrackNet(opt).to(device)

    loss_functions = {
        'mse': torch.nn.MSELoss(),
        'euc': euclidean_loss,
        'huber': torch.nn.HuberLoss(),
        'wbce': wbce_loss,
        'l1': torch.nn.L1Loss(),
        'adwing': AdaptiveWingLoss(),
        'my_loss': construct_my_loss(opt)
    }
    loss_function = loss_functions[opt.loss_function]

    if opt.weights:
        model.load_state_dict(torch.load(opt.weights))

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)


    if opt.type == 'auto':
        full_dataset = dataset.GenericDataset.from_dir(opt)
    elif opt.type == 'image':
        full_dataset = dataset.ImagesDataset(opt)
    elif opt.type == 'video':
        full_dataset = dataset.VideosDataset(opt)
    else:
        raise Exception("type argument must be one of {'auto', 'image', 'video'}")

    train_size = int(opt.train_size * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(not opt.no_shuffle))
    val_loader = DataLoader(test_dataset, batch_size=opt.val_batch_size)

    images, heatmaps = next(iter(train_loader))

    # initialize logging
    writer = None
    if opt.tensorboard:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter()
        writer.add_graph(model, images.to(device))
    if opt.wandb:
        import wandb
        wandb.init(
            project=opt.project_name,
            config=vars(opt)
        )
        wandb.watch(model, criterion=loss_function, log='all', log_freq=opt.log_period)

    print('Loss using zeros: ', loss_function(torch.zeros_like(heatmaps), heatmaps), '\n')

    save_path = Path(opt.save_path) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.json", "w") as file:
        json.dump(vars(opt), file)

    if opt.single_batch_overfit:
        print('Overfitting on a single batch.')
        training_loop(opt, device, model, writer, loss_function, optimizer, [(images, heatmaps)], None, save_path)

    else:
        print("Starting training")
        training_loop(opt, device, model, writer, loss_function, optimizer, train_loader, val_loader, save_path)

    wandb.finish()
