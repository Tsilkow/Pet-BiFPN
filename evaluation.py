import numpy as np


def calc_iou(prediction, ground_truth):
    """
    Given prediction of shape (B, C, H, W)
    and ground_truth of shape (B, C, H, W)
    outputs an array of shape (B, C).
    such that at the position (b, c) is the
    value of the intersection of the prediction and the ground_truth mask 
    (i.e. number of points where both are 1)
    divided by their union (i.e. number of points where at least one is 1) 
    (assume 0/0 = 0).
    """

    assert len(prediction.shape) == 4
    assert prediction.shape == ground_truth.shape
    assert np.logical_or(prediction == 1, prediction == 0).all()
    assert np.logical_or(ground_truth == 1, ground_truth == 0).all()
    ## TODO {
    intersection = np.logical_and(prediction == 1, ground_truth == 1).sum(axis=(2, 3))
    union = np.logical_or(prediction == 1, ground_truth == 1).sum(axis=(2, 3))
    result = np.divide(intersection, union, out=np.zeros(prediction.shape[:2]), where=(union != 0))
    ## }
    assert result.shape == prediction.shape[:2]
    return result


def calc_accuracy(prediction, ground_truth):
    """
    Given prediction of shape (B, C, H, W)
    and ground_truth of shape (B, C, H, W)
    outputs an array of shape (B, C).
    such that at the position (b, c) is the
    fraction of all pixels correctly classified
    """

    assert len(prediction.shape) == 4
    assert prediction.shape == ground_truth.shape
    assert np.logical_or(prediction == 1, prediction == 0).all()
    assert np.logical_or(ground_truth == 1, ground_truth == 0).all()
    ## TODO {
    correct = (prediction == ground_truth).sum(axis=(2, 3))
    result = correct / (ground_truth.shape[2]*ground_truth.shape[3])
    ## }
    assert result.shape == prediction.shape[:2]
    return result


def test_metrics():
    # Some tests

    a = np.zeros((1, 1, 64, 64))
    a[:, :, 32:48, 32:48] = 1
    b = np.zeros((1, 1, 64, 64))
    b[:, :, 47:63, 47:63] = 1

    assert np.isclose(
        calc_iou(np.concatenate([a, a], axis=1), np.concatenate([a, b], axis=1)),
        np.array([1.0, 1.0 / 511.0])[None, :],
    ).all()

    c = np.zeros((1, 1, 64, 64))

    assert np.isclose(
        calc_iou(c, c),
        np.array([0.0])[:, None],
    ).all()
    
    assert np.isclose(
        calc_accuracy(np.concatenate([a, a], axis=1), np.concatenate([a, b], axis=1)),
        np.array([1.0, (4096-510)/4096])[None, :],
    ).all()


def one_hot_encode_prediction(predictions):
    """
    Given predictions of shape (B, H, W)
    with number
    0 - representing pixels belonging to the pet,
    1 - background,
    2 - outline
    one hot encodes it as
    single tensor of shape (B, 3, H, W), such that
    element (b, c, h, w) is 1 if the pixel corresponds
    to class c and 0 otherwise.
    """

    assert len(predictions.shape) == 3

    predictions_oh = torch.nn.functional.one_hot(predictions, num_classes=3)
    predictions_oh = predictions_oh.permute(0, 3, 1, 2)  # B, C, H, W

    assert len(predictions_oh.shape) == 4
    assert predictions_oh.shape[1] == 3

    return predictions_oh


def eval_fn(model, test_loader, device):
    model.eval()

    iou = []
    acc = []
    total = 0
    for i, data in enumerate(test_loader):
        x, y = data
        print(f'\r---EVALUATION: {str(i+1).rjust(3)}/{len(test_loader)}    ', end='')
        ground_truth = one_hot_encode_prediction(y).cpu().numpy()
        x, y = x.to(device), y.to(device)

        logits = model(x)
        prediction = torch.argmax(logits.detach(), dim=-3, keepdim=False)
        prediction = one_hot_encode_prediction(prediction).cpu().numpy()

        iou.append(np.sum(calc_iou(prediction, ground_truth), axis=0))
        acc.append(np.sum(calc_accuracy(prediction, ground_truth), axis=0))
        total += prediction.shape[0]

    iou = np.stack(iou, axis=-1).sum(-1) / total
    acc = np.stack(acc, axis=-1).sum(-1) / total
    assert len(iou.shape) == 1
    assert len(acc.shape) == 1
    assert iou.shape[0] == 3
    assert acc.shape[0] == 3

    print(f"\nIOU PET: {iou[0]}")
    print(f"IOU BG: {iou[1]}")
    print(f"IOU OUT: {iou[2]}")

    print(f"ACC PET: {acc[0]}")
    print(f"ACC BG: {acc[1]}")
    print(f"ACC OUT: {acc[2]}")
