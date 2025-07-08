import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Custom ImageFolder to sort class folders numerically (not alphabetically)
class FixedImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        # Sort folder names by integer order: ['0', '1', ..., '42']
        classes = sorted(os.listdir(directory), key=lambda x: int(x))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def get_data_loaders(data_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Use FixedImageFolder to maintain correct label mapping
    train_dataset = FixedImageFolder(train_dir, transform=transform)
    test_dataset = FixedImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_to_idx = train_dataset.class_to_idx   # {'0': 0, '1': 1', ..., '42': 42}
    classes = [int(cls) for cls in train_dataset.classes]  # [0, 1, ..., 42] as integers

    return train_loader, test_loader, class_to_idx, classes
