import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim

DETECTED_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_color_distortion(s=1.0):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def simCLR_criterion(batch1, batch2, temp=0.1):
    batch_size = batch1.size(0)
    x = torch.cat([batch1, batch2], dim=0)
    x = x / x.norm(dim=1)[:, None]
    x = torch.mm(x, x.t())
    x = torch.exp(x / temp)
    sums = x.sum(dim=0)
    x = torch.cat((torch.diagonal(x, offset=batch_size, dim1=1, dim2=0), torch.diagonal(x, offset=batch_size, dim1=0, dim2=1)))
    return -torch.log(x / (sums-x)).mean()

    
def simCLR_train_iteration(model, train_loader, projector, augment, optimizer, scheduler, criterion=simCLR_criterion, logger=None, device=DETECTED_DEVICE):
    model.train()
    total_loss = 0
    batches = len(train_loader)
    for batch_idx, (batch, _) in enumerate(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch1, batch2 = augment(batch), augment(batch)
        h1, h2 = model(batch1), model(batch2)
        z1, z2 = projector(h1), projector(h2)
        loss = criterion(z1, z2)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        total_loss += batch_loss
        if logger is not None:
            logger.debug(f'batch: {batch_idx}/{batches} batch_loss: {batch_loss}')
    scheduler.step()
    mean_loss = total_loss / batches
    return mean_loss 

def simCLR_validate_iteration(model, val_loader, projector, augment, criterion=simCLR_criterion, logger=None, device=DETECTED_DEVICE):
    model.eval()
    projector.eval()
    total_loss = 0
    with torch.no_grad():
        for _, (batch, _) in enumerate(val_loader):
            batch = batch.to(device)
            batch1, batch2 = augment(batch), augment(batch)
            h1, h2 = model(batch1), model(batch2)
            z1, z2 = projector(h1), projector(h2)
            loss = criterion(z1, z2)
            total_loss += loss.item()
    mean_loss = total_loss / len(val_loader)
    return mean_loss

def simCLR_train(
    train_loader, 
    val_loader=None,
    model=models.resnet50(),
    projector=nn.Sequential(nn.Linear(1000, 128), nn.ReLU(), nn.Linear(128, 128)),
    optimizer=None, 
    scheduler=None,
    augment=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(),
        transforms.GaussianBlur(kernel_size=3),
    ]),
    criterion=simCLR_criterion, 
    num_epochs=100,
    logger=None, 
    device=DETECTED_DEVICE
):
    logger.info('Starting training')
    model = model.to(device)
    projector = projector.to(device)
    if optimizer is None:
        optimizer = optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=0.075, weight_decay=1e-6)
    if scheduler is None:
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)
        ], milestones=[10])

    for epoch in range(num_epochs):
        train_loss = simCLR_train_iteration(model, train_loader, projector, augment, optimizer, scheduler, criterion, logger, device)
        if val_loader is not None:
            val_loss = simCLR_validate_iteration(model, val_loader, projector, augment, criterion, logger, device)
        else:
            val_loss = 'N/A'
        if logger is not None:
            logger.info(f'epoch: {epoch} train_loss: {train_loss} val_loss: {val_loss}')

    logger.info('Training complete')

    return model, projector

if __name__ == '__main__':
    import time, logging, torchvision
    model = models.mobilenet_v3_small()
    dataset = torchvision.datasets.ImageFolder(root="~/datasets/yugioh/val", transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.4862, 0.4405, 0.4220], [0.2606, 0.2404, 0.2379])]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    logging.basicConfig(
        filename='train.log',
        filemode='a',
        format='',
        level=logging.INFO
    )

    start = time.time()
   
    model, projector = simCLR_train(dataloader, model=model, logger=logging.getLogger(), num_epochs=1)

    torch.save(model.state_dict(), 'model.pth')
    torch.save(projector.state_dict(), 'projector.pth')

    print('Time:', (time.time() - start)*1000)

