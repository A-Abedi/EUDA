import torch.nn as nn

from dinov2.dinov2.hub.backbones import dinov2_vitb14_reg, dinov2_vitl14_reg, dinov2_vitg14_reg


__backbone_factory = {
    "DINOv2_base": dinov2_vitb14_reg,
    "DINOv2_large": dinov2_vitg14_reg,
    "DINOv2_huge": dinov2_vitl14_reg
}

__maximum_layer_node = {
    "huge": 8192,
    "large": 4096,
    "base": 2048,
    "small": 256
}


def get_backbone(backbone_type: str, backbone_size: str) -> nn.Module:
    backbone_name = f"{backbone_type}_{backbone_size}"

    try:
        return __backbone_factory[backbone_name]()
    except KeyError:
        raise NotImplementedError(f"There is no backbone type {backbone_name} and size {backbone_size}")


def get_bottleneck(bottleneck_size: str, input_feature: int) -> nn.Module:
    previous_layer = __maximum_layer_node[bottleneck_size]

    layers = [
        nn.Linear(input_feature, previous_layer),
    ]

    while True:
        if previous_layer == 256:
            break

        current_layer = previous_layer // 2

        layers.append(nn.Linear(previous_layer, current_layer))

        previous_layer = current_layer

    bottleneck = nn.Sequential(
        *layers
    )

    return bottleneck


class EUDA(nn.Module):
    def __init__(self,
                 backbone_type: str,
                 backbone_size: str,
                 bottleneck_size: str,
                 num_classes: int):
        super().__init__()

        self.backbone = get_backbone(backbone_type, backbone_size)
        self.bottleneck = get_bottleneck(bottleneck_size, self.backbone.norm.normalized_shape[0])

        self.head = nn.Linear(256, num_classes)

    def forward(self, x, return_features_only=False):
        features = self.backbone(x)
        features = self.bottleneck(features)

        if not return_features_only:
            return features, self.head(features)

        return self.head(features)
