import torch
import torchvision
import matplotlib.pyplot as plt


def get_pet_mask(mask):
    """
    Given a mask from Oxford-IIIT Pet Dataset after subtraction of 1
    returns the array that encodes parts belonging to the pet.
    """
    return (mask == 0).type(torch.float)


def get_pet_background(mask):
    """
    Given a mask from Oxford-IIIT Pet Dataset after subtraction of 1
    returns the array that encodes parts belonging to the background.
    """
    return (mask == 1).type(torch.float)


def get_pet_outline(mask):
    """
    Given a mask from Oxford-IIIT Pet Dataset after subtraction of 1
    returns the array that encodes parts belonging to the pet outline.
    """
    return (mask == 2).type(torch.float)


def tensor_to_image(tensor):
    return tensor.permute(1, 2, 0).numpy()


def layer_to_tensor(layer):
    return layer.repeat(3, 1, 1)


def visualize_data(images, masks, supress=False):
    """
    Args:
        images: tensor of shape (BATCH, 3, H, W)
        masks: tensor of shape (BATCH, H, W)
    Draws a grid of images of size BATCH x 4.
    I'th row consists of the image, pet mask, pet outline, and background mask.
    """
    assert len(images.shape) == 4
    assert len(masks.shape) == 3
    num_images = images.shape[0]
    assert masks.shape[0] == num_images

    resizer = torchvision.transforms.Resize(
        images[0].shape[1:],
        interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        antialias=None)

    figure, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images), squeeze=False)

    for ax, image, mask in zip(axes, images, masks):
        for slot, picture in zip(
                ax,
                [image,
                 image * resizer(torch.maximum(
                     layer_to_tensor(get_pet_mask(mask)),
                     layer_to_tensor(get_pet_outline(mask)))),
                 image * resizer(layer_to_tensor(get_pet_background(mask)))]):
            tmp = tensor_to_image(picture)
            slot.axis('off')
            slot.grid(False)
            slot.imshow(tmp)

    plt.savefig('result.png')
    if not supress: plt.show()
