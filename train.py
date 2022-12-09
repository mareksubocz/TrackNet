import torch
from TrackNet import TrackNet

BATCH_SIZE = 3
NUM_EPOCHS = 10


def wbce_loss(output, target):
    return -(
        ((1-output)**2) * target *
        torch.log(torch.clamp(output, min=1e-15, max=1)) +
        output**2 * (1-target) *
        torch.log(torch.clamp(1-output, min=1e-15, max=1))
    ).sum()

if __name__ == '__main__':
    # output = torch.tensor([0., 0., 0.1, 0.4, 0.9, 0.3, 0.2, 0.])
    # target = torch.tensor([0., 0., 0.2, 0.4, 1.0, 0.4, 0.2, 0.])
    # print(output)
    # print(target)
    # print(wbce_loss(output, target))
    model = TrackNet()
    optimizer = torch.optim.SGD(
      model.parameters(), lr=0.01, momentum=0.9
    )

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (features, targets) in enumerate(train_loader):

            forward_pass_outputs = model(features)
            loss = wbce_loss(forward_pass_outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
