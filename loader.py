import torch
import torchvision
from PIL import Image


def load_image_from_file(hparams, path):
    input_transforms_raw = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(hparams.image_size),
        ]
    )
    input_transforms_normalized = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(hparams.image_size),
            torchvision.transforms.Normalize(hparams.image_net_mean, hparams.image_net_std),
        ]
    )
    with Image.open(path) as image:
        return input_transforms_raw(image), input_transforms_normalized(image)
    return None, None


def create_loader(hparams, data_split, sample_limit=None):
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
                hparams.prediction_size,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.Lambda(lambda x: x - torch.ones_like(x)),
            torchvision.transforms.Lambda(lambda x: x.squeeze(dim=0).long()),
        ]
    )
    dataset = torchvision.datasets.OxfordIIITPet(
        root=hparams.data_dir,
        split=data_split,
        download=True,
        target_types="segmentation",
        transform=input_transforms,
        target_transform=target_transforms,
    )
    if sample_limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(sample_limit))
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=hparams.batch_size
    )
    return loader


def test_loader(hparams, loader):
    samples = next(iter(loader))
    images, masks = samples

    assert len(images.shape) == 4
    assert images.shape[0] == hparams.batch_size
    assert images.shape[1] == 3
    assert images.shape[2] == hparams.image_size[0]
    assert images.shape[3] == hparams.image_size[1]
    assert images.dtype == torch.float32

    print(masks.shape, hparams.batch_size, hparams.prediction_size)
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

