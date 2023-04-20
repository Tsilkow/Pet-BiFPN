import torch
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


def visualize_data(images, masks):
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

    figure, axes = plt.subplots(num_images, 4, figsize=(20, 5*num_images), squeeze=False)

    for i, ax in enumerate(axes):
        for j, img in enumerate(
            [images[i],
             layer_to_tensor(get_pet_mask(masks[i])),
             layer_to_tensor(get_pet_outline(masks[i])),
             layer_to_tensor(get_pet_background(masks[i]))]):
            tmp = tensor_to_image(img)
            ax[j].imshow(tmp)

    plt.show()
