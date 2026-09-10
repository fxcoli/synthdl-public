import torch.nn as nn
import torchvision.models as models

MODEL_IMAGE_SIZES = {
    "resnet18": 224,
    "resnet34": 224,
    "resnet50": 224,
    "resnext": 224,
    "convnext": 224,
    "convnext_tiny": 224,
    "efficientnet": 480,
    "efficientnet_b0": 224,
    "efficientnet_b1": 240,
    "efficientnet_b2": 260,
    "efficientnet_b3": 300,
    "efficientnet_b4": 380,
    "efficientnet_b5": 456,
    "swin": 256,
    "swin_t": 224,
}


def build_classifier(
    model_name: str, freeze_backbone: bool, num_classes: int, dropout: float = 0.0
) -> nn.Module:
    """
    Instantiates a specified PyTorch Torchvision architecture, initialises default weights,
    and replaces the final classification layer to match the target class dimensionality.
    Allows for gradient exclusion on the feature extraction backbone.

    Arguments:
        model_name -- String identifier corresponding to a key in MODEL_IMAGE_SIZES.
        num_classes -- Target dimensionality for the output tensor.
        freeze_backbone -- Boolean dictating whether backbone parameters require gradients.
    """
    out_features = max(num_classes, 1)

    if model_name in ["resnet18", "resnet34", "resnet50"]:
        base_net = getattr(models, model_name)(weights="DEFAULT")
        if freeze_backbone:
            for param in base_net.parameters():
                param.requires_grad = False
        base_net.fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(base_net.fc.in_features, out_features)
        )

    elif model_name == "resnext":
        base_net = models.resnext101_32x8d(
            weights=models.ResNeXt101_32X8D_Weights.DEFAULT
        )
        if freeze_backbone:
            for param in base_net.parameters():
                param.requires_grad = False
        base_net.fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(base_net.fc.in_features, out_features)
        )

    elif model_name == "convnext":
        base_net = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        if freeze_backbone:
            for param in base_net.parameters():
                param.requires_grad = False
        base_net.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_net.classifier[-1].in_features, out_features),
        )

    elif model_name == "convnext_tiny":
        base_net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        if freeze_backbone:
            for param in base_net.parameters():
                param.requires_grad = False
        base_net.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_net.classifier[-1].in_features, out_features),
        )

    elif model_name == "efficientnet":
        base_net = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.DEFAULT
        )
        if freeze_backbone:
            for param in base_net.parameters():
                param.requires_grad = False
        base_net.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_net.classifier[-1].in_features, out_features),
        )

    elif model_name in [
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
    ]:
        base_net = getattr(models, model_name)(weights="DEFAULT")
        if freeze_backbone:
            for param in base_net.parameters():
                param.requires_grad = False
        base_net.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_net.classifier[-1].in_features, out_features),
        )

    elif model_name == "swin":
        base_net = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
        if freeze_backbone:
            for param in base_net.parameters():
                param.requires_grad = False
        base_net.head = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(base_net.head.in_features, out_features)
        )

    elif model_name == "swin_t":
        base_net = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        if freeze_backbone:
            for param in base_net.parameters():
                param.requires_grad = False
        base_net.head = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(base_net.head.in_features, out_features)
        )

    else:
        raise ValueError(f"Model {model_name} is not supported")
    return base_net