import torch
from loader import create_loaders, test_loader, augment_data
from visualisation import visualize_data
from evaluation import test_metrics, one_hot_encode_prediction, eval_fn


class Hyperparameters:
    def __init__(self):
        self.batch_size = 16
        self.image_size = (128, 128) # width and height input images are scaled to
        self.prediction_size = (64, 64) # width and height of predictions
        self.image_net_mean = [0.485, 0.456, 0.406] # mean for backbone-specific value rescaling
        self.image_net_std = [0.229, 0.224, 0.225] # standard deviation for backbone-specific value rescaling
        self.data_path = './data'


def train(
    model,
    optimizer,
    training_loader,
    testing_loader,
    num_epoch,
    eval_fn,
    device,
    weight=torch.tensor([1.0, 1.0, 1.0]),
    augment_fn=(lambda im, gt: (im, gt)),
):
    """
    Args:
        model - model that given an image outputs a tensor
                with logits for determining whether a pixel belongs
                to the pet, the background, or the pet outline.
        eval_fn - function that given a model performs its evaluation on a given set;
                  called at the end of every epoch to report both test and train performance
        weight - used as weights for CrossEntropyLoss
        augment_fn - optional function that given batched input image, and batched target performs random augmentations,
                      used only for train set augmentation

    """
    ## TODO {
    weight = weight.to(device)
    model = model.to(device)
    for param in model.backbone.backbone.parameters():
        assert(param.requires_grad == False)
    loss = torch.nn.CrossEntropyLoss(weight)
    for e in range(num_epoch):
        model.train()
        for i, (images, masks) in enumerate(training_loader):
            print(f'\r---EPOCH {e+1}/{num_epoch}: {str(i).rjust(3)}/{len(training_loader)}      ', end='')
            optimizer.zero_grad()
            images, masks = augment_fn(images, masks)
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            ground_truth = one_hot_encode_prediction(masks).to(torch.float).to(device)
            error = loss(output, ground_truth)
            error.backward()
            optimizer.step()
    eval_fn(model, testing_loader, device)
    ## }


def create_model_and_optimizer():
    ## TODO {
    model = Net()
    optimizer = torch.optim.Adam(params=model.non_backbone_parameters())
    for param in model.backbone.backbone.parameters():
        param.requires_grad = False
    ## }
    return model, optimizer


def save_checkpoint(model, dir):
    torch.save(model.state_dict(), dir)


def load_checkpoint(model, dir):
    model.load_state_dict(torch.load(dir))


if __name__ == '__main__':
    hparams = Hyperparameters()
    training_loader, testing_loader = create_loaders(hparams)
    test_loader(hparams, training_loader)
    test_loader(hparams, testing_loader)
    test_metrics()

    visualize_data(training_loader[0][:4], training_loader[1][:4])

    model, optimizer = create_model_and_optimizer()
    train(model, optimizer, training_loader, testing_loader, 3, eval_fn, augment_fn=augment_data)
    save_checkpoint(model, './models/bifpn.model')
