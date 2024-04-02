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
            logger.debug(f'batch: {batch_idx+1}/{batches} batch_loss: {batch_loss}')
    scheduler.step()
    mean_loss = total_loss / batches
    return mean_loss

def simCLR_validate_iteration(model, val_loader, projector, augment, criterion=simCLR_criterion, logger=None, device=DETECTED_DEVICE):
    model.eval()
    projector.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (batch, _) in enumerate(val_loader):
            batch = batch.to(device)
            batch1, batch2 = augment(batch), augment(batch)
            h1, h2 = model(batch1), model(batch2)
            z1, z2 = projector(h1), projector(h2)
            loss = criterion(z1, z2)
            total_loss += loss.item()
            if logger is not None:
                logger.debug(f'val_batch: {batch_idx+1}/{len(val_loader)} val_batch_loss: {loss.item()}')
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
    logger.debug('Starting training')
    model = model.to(device)
    projector = projector.to(device)
    if optimizer is None:
        optimizer = optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=0.001, weight_decay=0.01)
    if scheduler is None:
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.33, end_factor=1.0, total_iters=5),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)
        ], milestones=[5])
    
    logger.info('epoch,train_loss,val_loss,lr...')

    for epoch in range(num_epochs):
        train_loss = simCLR_train_iteration(model, train_loader, projector, augment, optimizer, scheduler, criterion, logger, device)
        if val_loader is not None:
            val_loss = simCLR_validate_iteration(model, val_loader, projector, augment, criterion, logger, device)
        else:
            val_loss = 'N/A'
        if logger is not None:
            lr_string = ','.join(map(str, scheduler.get_last_lr()))
            logger.info(f'{epoch+1},{train_loss},{val_loss},{lr_string}')

    logger.debug('Training complete')

    return model, projector

if __name__ == '__main__':
    import logging, torchvision, sys
    
    model = models.mobilenet_v3_small()
    
    train_set = torchvision.datasets.ImageFolder(root="~/datasets/yugioh/train", transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.4862, 0.4405, 0.4220], [0.2606, 0.2404, 0.2379])]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True)

    val_set = torchvision.datasets.ImageFolder(root="~/datasets/yugioh/val", transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.4862, 0.4405, 0.4220], [0.2606, 0.2404, 0.2379])]))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=True)

    root_logger = logging.getLogger()
    loging_formatter = logging.Formatter('')
    root_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('train.log', mode='w')
    file_handler.setFormatter(loging_formatter)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(loging_formatter)
    console_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    model, projector = simCLR_train(train_loader, val_loader=val_loader, model=model, logger=root_logger)

    torch.save(model.state_dict(), 'model.pth')
    torch.save(projector.state_dict(), 'projector.pth')


