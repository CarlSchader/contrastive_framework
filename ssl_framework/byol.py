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

def NT_Xent(batch1, batch2, temp=0.1):
    batch_size = batch1.size(0)
    x = torch.cat([batch1, batch2], dim=0)
    x = x / x.norm(dim=1)[:, None]
    x = torch.mm(x, x.t())
    x = torch.exp(x / temp)
    sums = x.sum(dim=0)
    x = torch.cat((torch.diagonal(x, offset=batch_size, dim1=1, dim2=0), torch.diagonal(x, offset=batch_size, dim1=0, dim2=1)))
    return -torch.log((x / (sums-x))).mean()

def simCLR_train_iteration(model, train_loader, projector, augment, optimizer, scheduler, criterion=NT_Xent, logger=None, device=DETECTED_DEVICE, gradient_clip=None):
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

        if gradient_clip is not None and gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            torch.nn.utils.clip_grad_norm_(projector.parameters(), gradient_clip)

        optimizer.step()
        batch_loss = loss.item()
        total_loss += batch_loss
        if logger is not None:
            logger.debug(f'batch: {batch_idx+1}/{batches} batch_loss: {batch_loss}')
    scheduler.step()
    mean_loss = total_loss / batches
    return mean_loss

def simCLR_validate_iteration(model, val_loader, projector, augment, criterion=NT_Xent, logger=None, device=DETECTED_DEVICE):
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
    criterion=NT_Xent, 
    num_epochs=100,
    logger=None, 
    device=DETECTED_DEVICE,
    gradient_clip=1.0, # can be None or 0.0 to disable gradient clipping
    epoch_complete_hook=lambda epoch, train_loss, val_loss, model, projector, optimizer, scheduler: None
):
    logger.debug('Starting training')
    model = model.to(device)
    projector = projector.to(device)
    if optimizer is None:
        optimizer = optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=0.0001, weight_decay=0.1)
    if scheduler is None:
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.33, end_factor=1.0, total_iters=5),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)
        ], milestones=[5])
    
    logger.info('epoch,train_loss,val_loss,lr...')
    with torch.autograd.detect_anomaly():
        for epoch in range(num_epochs):
            train_loss = simCLR_train_iteration(model, train_loader, projector, augment, optimizer, scheduler, criterion, logger, device, gradient_clip=gradient_clip)
            if val_loader is not None:
                val_loss = simCLR_validate_iteration(model, val_loader, projector, augment, criterion, logger, device)
            else:
                val_loss = 'N/A'
            if logger is not None:
                lr_string = ','.join(map(str, scheduler.get_last_lr()))
                logger.info(f'{epoch+1},{train_loss},{val_loss},{lr_string}')

            if epoch_complete_hook is not None:
                epoch_complete_hook(epoch, train_loss, val_loss, model, projector, optimizer, scheduler)

    logger.debug('Training complete')

    return model, projector

if __name__ == '__main__':
    import logging, torchvision, os, sys, datetime

    model = models.resnet50()
    batch_size = 128

    # model = models.mobilenet_v3_large()
    # batch_size = 512

    # train_set = torchvision.datasets.ImageFolder(root="/home/carl/datasets/imagenet/ILSVRC/Data/CLS-LOC/train", transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True)
    #
    # val_set = torchvision.datasets.ImageFolder(root="/home/carl/datasets/imagenet/ILSVRC/Data/CLS-LOC/val", transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=True)

    train_set = torchvision.datasets.ImageFolder(root="~/datasets/yugioh/train", transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.4862, 0.4405, 0.4220], [0.2606, 0.2404, 0.2379])]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = torchvision.datasets.ImageFolder(root="~/datasets/yugioh/val", transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.4862, 0.4405, 0.4220], [0.2606, 0.2404, 0.2379])]))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    logging_file = 'train.log'
    root_logger = logging.getLogger()
    loging_formatter = logging.Formatter('')
    root_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(logging_file, mode='w')
    file_handler.setFormatter(loging_formatter)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(loging_formatter)
    console_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    save_dir = os.path.join('weights', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    def checkpoint_hook(epoch, train_loss, val_loss, hook_model, hook_projector, hook_optimizer, hook_scheduler):
        global best_val_loss

        with open(logging_file, 'r') as f:
            log_string = f.read()

        save_dict = {
            'model': hook_model.state_dict(), 
            'projector': hook_projector.state_dict(), 
            'optimizer': hook_optimizer.state_dict(), 
            'scheduler': hook_scheduler.state_dict(),
            'epoch': epoch,
            'log': log_string 
        }
        torch.save(save_dict, os.path.join(save_dir, f'latest.pth'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(save_dict, os.path.join(save_dir, 'best.pth'))
        if (epoch+1) % 10 == 0:
            torch.save(save_dict, os.path.join(save_dir, f'{epoch}.pth'))

    simCLR_train(train_loader, val_loader=val_loader, model=model, logger=root_logger, epoch_complete_hook=checkpoint_hook)

    # b1 = torch.randn(512, 128)
    # b2 = torch.randn(512, 128)
    #
    # print(simCLR_criterion(b1, b2))
    # print(simCLR_criterion_torch(b1, b2))


