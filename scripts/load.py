import torch

def load_checkpoint(checkpoint_path, model, projector, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    projector.load_state_dict(checkpoint['projector'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    
    return model, projector, optimizer, scheduler, epoch


if __name__ == '__main__':
    import sys, os
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models, transforms
    from PIL import Image

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from ssl_framework.simclr import get_color_distortion

    weights_path = sys.argv[1]
    image_path = sys.argv[2]

    num_epochs = 100
    
    transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.4862, 0.4405, 0.4220], [0.2606, 0.2404, 0.2379])])
    augment =   transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(),
        transforms.GaussianBlur(kernel_size=3),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    im1, im2 = augment(image), augment(image)

    model = models.mobilenet_v3_small()
    projector = nn.Sequential(nn.Linear(1000, 128), nn.ReLU(), nn.Linear(128, 128))

    model.eval()
    projector.eval()

    optimizer = optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=0.001, weight_decay=0.1)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
        optim.lr_scheduler.LinearLR(optimizer, start_factor=0.33, end_factor=1.0, total_iters=5),
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)
    ], milestones=[5])
    
    model, projector, optimizer, scheduler, epoch = load_checkpoint(weights_path, model, projector, optimizer, scheduler)
    
    # print("Model")
    # for param in model.parameters():
    #     print(param)
    #
    # print("Projector")
    # for param in projector.parameters():
    #     print(param)

    with torch.no_grad():
        h1, h2 = model(im1), model(im2)
        z1, z2 = projector(h1), projector(h2)
        # print("H1")
        # print(h1)
        # print("H2")
        # print(h2)
        # print("Z1")
        # print(z1)
        # print("Z2")
        # print(z2)
        # print("Loss")
        # print(simCLR_criterion(z1, z2))

    print('epoch:', epoch)
    print(optimizer)
    # print(scheduler)
    
