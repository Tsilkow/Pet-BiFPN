import argparse
import torch
from loader import create_loader, test_loader, augment_data
from visualisation import visualize_data
from evaluation import test_metrics, one_hot_encode_prediction, eval_fn
from model import Net


class Hyperparameters:
    def __init__(self):
        self.epoch_count = 10
        self.batch_size = 16
        self.image_size = (128, 128) # width and height input images are scaled to
        self.prediction_size = (64, 64) # width and height of predictions
        self.image_net_mean = [0.485, 0.456, 0.406] # mean for backbone-specific value rescaling
        self.image_net_std = [0.229, 0.224, 0.225] # standard deviation for backbone-specific value rescaling
        self.data_dir = './data'
        self.models_dir = './models'
        self.feature_channels = 128
        self.training_sample_limit = None
        self.testing_sample_limit = None
        self.device = 'cpu'



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


def create_model_and_optimizer(hparams):
    ## TODO {
    model = Net(hparams).to(hparams.device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--sanity-test', action=argparse.BooleanOptionalAction,
        help='performs nominal training and testing to confirm code is working')
    parser.add_argument(
        '-g', '--gpu', action=argparse.BooleanOptionalAction,
        help='flag for using GPU as a device; if not specified only CPU will be used')
    args = parser.parse_args()
    if args.sanity_test:
        hparams.epoch_count = 1
        hparams.training_sample_limit = hparams.batch_size
        hparams.testing_sample_limit = hparams.batch_size
    if args.gpu:
        hparams.device = 'gpu'
        
    training_loader = create_loader(hparams, 'trainval', hparams.training_sample_limit)
    testing_loader = create_loader(hparams, 'test', hparams.testing_sample_limit)

    test_loader(hparams, training_loader)
    test_loader(hparams, testing_loader)
    test_metrics()

    # images, masks = next(iter(training_loader))
    # visualize_data(images[:4], masks[:4])

    model, optimizer = create_model_and_optimizer(hparams)
    train(
        model, optimizer, training_loader, testing_loader, hparams.epoch_count, eval_fn,
        device=hparams.device, augment_fn=augment_data)
    save_checkpoint(model, f'{hparams.models_dir}/bifpn.model')
