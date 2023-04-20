""" PyTorch BiFPN with pretrained EfficientNets

Paper: https://arxiv.org/abs/1911.09070
"""

from itertools import chain
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class MatchChannels(nn.Module):
    """
    Given the input of shape (BATCH, in_channels, H, W),
    converts it to the one of shape (BATCH, out_channels, H, W)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        assert len(x.shape) == 4
        x = self.conv(x)
        x = self.bn(x)
        return x


class BackBoneWrapper(nn.Module):
    """
    Given the input x of shape (BATCH, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    produces the list of features, each having out_channels channels.
    Features are taken from the output self.backbone.extract_endpoints(x)
    and converted using MatchChannels to have an appropriate fnumber of channels.
    Features are ordered from the top to the bottom
    (first the ones of high resolution; note that FEATURE_FILTERS maintains this order).
    """

    def __init__(self, image_size, out_channels, device):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained("efficientnet-b0").to(device)

        dummy_input = torch.zeros((1, 3, image_size[0], image_size[1]))
        features = self.backbone.extract_endpoints(dummy_input)
        feature_shapes = {feature_name: feature_data.shape
                          for feature_name, feature_data in features.items()}
        self.feature_channels = dict(list(
            {feature_name: feature_shape[1]
             for feature_name, feature_shape in feature_shapes.items()}.items()
        ))
        self.num_feature_levels = len(self.feature_channels.items())

        ## TODO{
        self.channel_matchers = nn.ModuleList([MatchChannels(in_channels, out_channels)
                                               for in_channels in self.feature_channels.values()])
        ## }

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[1] == 3

        feature_level_dict = self.backbone.extract_endpoints(x)

        ## TODO{
        result = [self.channel_matchers[i](input)
                  for i, input in enumerate(feature_level_dict.values())]
        ## }

        assert len(result) == self.num_feature_levels
        for i in range(self.num_feature_levels):
            assert (
                result[i].shape[-2:]
                == feature_level_dict[f"reduction_{i+1}"].shape[-2:]
            )
        return result


class FeatureFusionBlock(nn.Module):
    """
    Used to fuse features from different levels in the feature pyramid.
    Given
        current_feature (of shape (BATCH, feature_channels, H, W))
        previous_feature (of shape (BATCH, feature_channels, H', W'))
        and optionally additional_feature (of shape (B, feature_channels, H, W))
    fuses them using the following equation
        for the case without additional_feature
            ACT(BN(CONVS(p1*current_feature + p2*resize(previous_feature))))
        for the case with additional_feature
            ACT(BN(CONVS(p1*current_feature + p2*resize(previous_feature) + p3*additional_feature)))
        where CONVS are convolution(s) used to process features after addition 
            (choice about the number of them and their properties is up to you)
        BN is the batch norm
        ACT is an activation function (for example it can be relu)
        p1, p2, p3, are scalars calculated using learnable weights (see the EfficientDet paper).

    """

    def __init__(self, feature_channels, use_additional, device):
        """
        Args:
            feature_channels - number of channels that each feature has
            use_additional - whether additional_feature will be provided
        """
        super().__init__()
        self.use_additional = use_additional
        ## TODO {
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(feature_channels)
        self.weights_for_2 = nn.Parameter(torch.rand(2))
        self.weights_for_3 = nn.Parameter(torch.rand(3))
        self.convs = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1),
            nn.Conv2d(feature_channels, feature_channels, 1, 1, 0),
        )
        ## }

    def forward(self, current_feature, previous_feature, additional_feature=None):
        # Below we check that self.use_additional iff additional_feature is not None
        assert not self.use_additional or additional_feature is not None
        assert self.use_additional or additional_feature is None

        assert len(current_feature.shape) == len(previous_feature.shape)
        assert current_feature.shape[:2] == previous_feature.shape[:2]
        if additional_feature is not None:
            assert current_feature.shape == additional_feature.shape

        ## TODO {
        if self.use_additional and additional_feature is not None:
            # ACT(BN(CONVS(p1*current_feature + p2*resize(previous_feature) + p3*additional_feature)))
            p = nn.functional.softmax(self.weights_for_3, dim=0)
            combined = self.relu(self.bn(self.convs(
                p[0]*current_feature
                + p[1]*nn.functional.interpolate(previous_feature, current_feature.shape[-2:])
                + p[2]*additional_feature)))
        else: 
            # ACT(BN(CONVS(p1*current_feature + p2*resize(previous_feature))))
            p = nn.functional.softmax(self.weights_for_2, dim=0)
            combined = self.relu(self.bn(self.convs(
                p[0]*current_feature 
                + p[1]*nn.functional.interpolate(previous_feature, current_feature.shape[-2:]))))
        ## }

        assert combined.shape == current_feature.shape

        return combined


class BiFPN(nn.Module):
    """
    Implements BiFPN similar to the one presented in EfficinetDet Paper.
    Given num_feature_levels features, each having feature_channels channels
    performs up and down feature fusion process using FeatureFusionBlocks.
    """

    def __init__(self, num_feature_levels, feature_channels, device):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_feature_levels = num_feature_levels
        
        ## TODO {
        self.fusions_down = []
        self.fusions_up = [None] # initial fusing is skipped, None is added for alignment purposes

        # Matchings and fusions on the way down (towards higher resolution)
        for i in range(num_feature_levels-1):
            self.fusions_down.append(
                FeatureFusionBlock(self.feature_channels, False, device))
        self.fusions_down.append(None) # final fusing is skipped, None is added for alignment purposes

        # Matchings and fusions on the way up (towards lower resolution)
        for i in range(1, num_feature_levels):
            self.fusions_up.append(
                FeatureFusionBlock(self.feature_channels, (i != num_feature_levels-1), device))
        ## }

    def forward(self, features, skip_down=False):
        """
        Args:
            features - self.num_feature_levels features 
                       ordered from the highest to the lowest resolution.
            skip_down - if true then skip the second part of the feature fusion
                        (from high resolution to low resolution)
        Returns:
            list of fused features ordered in the same way as the input
        """
        assert len(features) == self.num_feature_levels

        ## TODO {
        result = [None for feat in features]
        for i, (feat, fuser) in reversed(list(enumerate(zip(features, self.fusions_down)))):
            if i == len(features)-1: result[i] = feat
            else: result[i] = fuser(feat, result[i+1])
        if not skip_down:
            for i, (feat, fuser) in enumerate(zip(features, self.fusions_up)):
                if i == 0: result[i] = feat
                elif i == len(features)-1: result[i] = fuser(feat, result[i-1])
                else: result[i] = fuser(result[i], result[i-1], feat)
        ## }

        assert len(result) == self.num_feature_levels
        assert result[0].shape == features[0].shape

        for res, feat in zip(result, features):
            assert res.shape == feat.shape

        return result


class SegmentationHead(nn.Module):
    """
    Given an input of shape (B, feature_channels, H, W)
    Produces the output of shape (B, num_classes, H', W') (where H', W' = output_shape)
    consisting of logits that can be used to classify each pixel.
    To do so uses additional convolution(s) that operate on input with inner_channels channels.
    """

    def __init__(
        self,
        feature_channels,
        output_shape,
        inner_channels=64,
        num_classes=3
    ):
        super().__init__()
        self.output_shape = output_shape
        self.num_classes = num_classes
        ## TODO {
        self.convs = nn.Sequential(
            nn.Conv2d(feature_channels, inner_channels, 3, 1, 1),
            nn.Conv2d(inner_channels, num_classes, 1, 1, 0),
        )
        ## }

    def forward(self, x):
        ## TODO {
        x = self.convs(x)
        result = nn.functional.interpolate(x, self.output_shape)
        ## }

        assert result.shape[0] == x.shape[0]
        assert result.shape[1] == self.num_classes
        assert result.shape[2:] == self.output_shape
        return result


class Net(nn.Module):
    """
    Uses BackBoneWrapper with feature_channels as a backbone.
    Uses BiFPN.
    Returns a tensor of shape (BATCH, 3, H, W) (where H, W = output_shape)
    with logits for pet, background, and outline. 
    """

    def __init__(self, hparams, device):
        super().__init__()
        ## TODO {
        self.backbone = BackBoneWrapper(
            hparams.prediction_size, hparams.feature_channels, device)
        self.bifpn1 = BiFPN(6, hparams.feature_channels, device)
        #self.bifpn2 = BiFPN(len(FEATURE_CHANNELS), feature_channels).to(device)
        self.segmentation = SegmentationHead(
            hparams.feature_channels, hparams.prediction_size)
        ## }

    def non_backbone_parameters(self):
        """
        Returns all parameters except the backbone ones
        """
        ## TODO {
        tmp = [module.parameters() for module in self.backbone.channel_matchers]
        tmp.append(self.bifpn1.parameters())
        #tmp.append(self.bifpn2.parameters())
        tmp.append(self.segmentation.parameters())
        parameters = chain(*tmp)
        ## }
        return parameters

    def forward(self, x):
        ## TODO {
        x = self.backbone(x)
        #x = self.bifpn1(x, False)
        x = self.bifpn1(x, True)[0]
        segmentation = self.segmentation(x)
        ## }
        assert segmentation.shape[0] == x.shape[0]
        assert segmentation.shape[1] == 3 # logits for pet, background and outline
        return segmentation
