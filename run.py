import time
import argparse
import torch
from loader import load_image_from_file, create_loader, test_loader, augment_data
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
        self.images_dir = './images'
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
    weight = weight.to(device)
    model = model.to(device)
    for param in model.backbone.backbone.parameters():
        assert param.requires_grad is False
    loss = torch.nn.CrossEntropyLoss(weight)
    for e in range(num_epoch):
        model.train()
        for i, (images, masks) in enumerate(training_loader):
            print(f'\rEPOCH {e+1}/{num_epoch}: {str(i).rjust(3)}/{len(training_loader)}      ', end='')
            optimizer.zero_grad()
            images, masks = augment_fn(images, masks)
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            ground_truth = one_hot_encode_prediction(masks).to(torch.float).to(device)
            error = loss(output, ground_truth)
            error.backward()
            optimizer.step()
    eval_fn(model, testing_loader, device)


def create_model_and_optimizer(hparams):
    model = Net(hparams).to(hparams.device)
    optimizer = torch.optim.Adam(params=model.non_backbone_parameters())
    for param in model.backbone.backbone.parameters():
        param.requires_grad = False
    return model, optimizer


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    hparams = Hyperparameters()
    model_signature = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--sanity-test', action=argparse.BooleanOptionalAction,
        help='performs nominal training and testing to confirm code is working')
    parser.add_argument(
        '-g', '--gpu', action=argparse.BooleanOptionalAction,
        help='flag for using GPU as a device; if not specified only CPU will be used')
    parser.add_argument(
        '-n', '--name',
        help=f'filename used to save model in {hparams.models_dir}')
    parser.add_argument(
        '-l', '--load',
        help=f'loads model from specified file in {hparams.models_dir}; if not specified new model is created')
    parser.add_argument(
        '-t', '--train', action=argparse.BooleanOptionalAction,
        help=f'flag for running training of the model; if not specified, existing model will be used to create visualisations')
    parser.add_argument(
        '-i', '--input',
        help=f'filename of custom input in {hparams.images_dir} to run segmentation on; if unspecified, test dataset will be used')
    args = parser.parse_args()
    
    if args.sanity_test:
        hparams.epoch_count = 1
        hparams.training_sample_limit = hparams.batch_size
        hparams.testing_sample_limit = hparams.batch_size
    if args.gpu:
        hparams.device = 'gpu'
    if args.name is not None:
        model_signature = args.name

    model, optimizer = create_model_and_optimizer(hparams)
    
    if args.load is not None:
        model = load_checkpoint(model, f'{hparams.models_dir}/{args.load}')

    images, masks = None, None

    if args.train:
        training_loader = create_loader(hparams, 'trainval', hparams.training_sample_limit)
        testing_loader = create_loader(hparams, 'test', hparams.testing_sample_limit)
        train(
        model, optimizer, training_loader, testing_loader, hparams.epoch_count, eval_fn,
        device=hparams.device, augment_fn=augment_data)
        save_checkpoint(model, f'{hparams.models_dir}/bifpn-{model_signature}.model')
        images, _ = next(iter(testing_loader))
        images = images[:4]

    if args.input is not None:
        images = load_image_from_file(hparams, f'{hparams.images_dir}/{args.input}')

    if images is not None:
        logits = model(images.to(hparams.device)).detach()
        prediction = torch.argmax(logits, dim=-3, keepdim=False)
        visualize_data(images.cpu(), prediction.cpu())
