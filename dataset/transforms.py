from torchvision import transforms

"""DA data transformers."""
def image_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


"""Get transformers."""
def dg_train_transformer(img_size, scales=[0.8, 1], random_h_f=0.5, jitter=0.4):
    img_tr = [transforms.RandomResizedCrop((int(img_size), int(img_size)), (scales[0],scales[1]))]
    if random_h_f > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(random_h_f))
    if jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=min(0.5, jitter)))
    img_tr.append(transforms.RandomGrayscale(0.1))
    img_tr.append(transforms.ToTensor())
    img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(img_tr)

def dg_test_transformer(img_size):
    img_tr = [transforms.Resize((img_size,img_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

def digit_transforms(scale, is_random=False):
    transform_list = []
    if is_random:
        transform_list.append(transforms.RandomCrop(scale,padding=4))
        transform_list.append(transforms.RandomHorizontalFlip())
    else:
        transform_list.append(transforms.Resize((scale,scale)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5,),(0.5,)))

    return transforms.Compose(transform_list)


