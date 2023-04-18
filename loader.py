import torch
import torchvision


def create_loaders(hparams):
    input_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(hparams.image_size),
            torchvision.transforms.Normalize(hparams.image_net_mean, hparams.image_net_std),
        ]
    )
    target_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.Resize(
                hparams.image_size,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.Lambda(lambda x: x - torch.ones_like(x)),
            torchvision.transforms.Lambda(lambda x: x.squeeze(dim=0).long()),
        ]
    )
    training_dataset = torchvision.datasets.OxfordIIITPet(
        root=hparams.data_path,
        split="trainval",
        download=True,
        target_types="segmentation",
        transform=input_transforms,
        target_transform=target_transforms,
    )
    testing_dataset = torchvision.datasets.OxfordIIITPet(
        root=hparams.data_path,
        split="test",
        download=True,
        target_types="segmentation",
        transform=input_transforms,
        target_transform=target_transforms,
    )
    training_loader = torch.utils.data.DataLoader(
        training_dataset, shuffle=True, batch_size=hparams.batch_size
    )
    testing_loader = torch.utils.data.DataLoader(
        testing_dataset, shuffle=True, batch_size=hparams.batch_size
    )
    return training_loader, testing_loader


def test_loader(hparams, loader):
    samples = next(iter(loader))
    images, masks = samples

    assert len(images.shape) == 4
    assert images.shape[0] == hparams.batch_size
    assert images.shape[1] == 3
    assert images.shape[2] == hparams.image_size[0]
    assert images.shape[3] == hparams.image_size[1]
    assert images.dtype == torch.float32

    assert len(masks.shape) == 3
    assert masks.shape[0] == hparams.batch_size
    assert masks.shape[1] == hparams.prediction_size[0]
    assert masks.shape[2] == hparams.prediction_size[1]
    assert masks.dtype == torch.long
    assert (masks <= 2).all()
    assert (masks >= 0).all()


def augment_data(images, ground_truth):
    """
    Args:
        images of shape (B, 3, H, W)
        ground_truth of shape (B, H', W')
    Returns augmented images and ground_truth.
    Augmentation is random.
    """

    batch_size = images.shape[0]
    assert len(images.shape) == 4 # (B, 3, H, W)
    assert len(ground_truth.shape) == 3 # (B, H', W')

    assert ground_truth.shape[0] == batch_size

    ## TODO {
    angle = (float(torch.rand(1)[0])-0.5)*20
    aug_images = torchvision.transforms.functional.rotate(images, angle)
    aug_gt = torchvision.transforms.functional.rotate(ground_truth, angle)
    ## }

    assert aug_images.shape == images.shape
    assert ground_truth.shape == aug_gt.shape
    assert torch.logical_or(aug_gt == 0, torch.logical_or(aug_gt == 1, aug_gt == 2)).all()

    return aug_images, aug_gt

