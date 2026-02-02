from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet1d18(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        embedding_size: int = 512,
        dropout_rate_first: float = 0.0,
        dropout_rate_subsequent: float = 0.0,
        zero_init_residual: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _block = BasicBlock
        _layers = [2, 2, 2, 2]
        _norm_layer = norm_layer or nn.BatchNorm1d

        self._norm_layer = _norm_layer
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = _norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(_block, 64, _layers[0], dropout_rate=dropout_rate_first)
        self.layer2 = self._make_layer(_block, 128, _layers[1], stride=2, dropout_rate=dropout_rate_subsequent)
        self.layer3 = self._make_layer(_block, 256, _layers[2], stride=2, dropout_rate=dropout_rate_subsequent)
        self.layer4 = self._make_layer(_block, 512, _layers[3], stride=2, dropout_rate=dropout_rate_subsequent)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512 * _block.expansion, embedding_size)
        self.fc2 = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: type, planes: int, blocks: int, stride: int = 1, dropout_rate: float = 0.0) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer, dropout_rate=dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.fc1(x)
        return self.fc2(torch.relu(emb))


class LinearCoAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.affinity_transform = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_feat: Tensor, key_feat: Tensor) -> Tensor:
        query_feat_transformed = self.affinity_transform(query_feat)
        affinity = torch.bmm(query_feat_transformed, key_feat.transpose(1, 2))
        attn_weights = self.softmax(affinity)
        enhanced_query = torch.bmm(attn_weights, key_feat)
        return enhanced_query


class MultiModalResNet18(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        embedding_size: int = 512,
        dropout_rate: float = 0.0,
        dropout_rate_first: Optional[float] = None,
        dropout_rate_subsequent: Optional[float] = None,
        block: type = BasicBlock,
        layers: List[int] = None,
    ) -> None:
        super().__init__()
        layers = layers or [2, 2, 2, 2]
        if dropout_rate_first is None:
            dropout_rate_first = dropout_rate
        if dropout_rate_subsequent is None:
            dropout_rate_subsequent = dropout_rate

        norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.stem_ppg = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_vpg = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_apg = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.co_attn_pv = LinearCoAttention(dim=64)
        self.co_attn_pa = LinearCoAttention(dim=64)

        self.fusion_conv = conv1x1(64 * 3, 64)

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate_first)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate_subsequent)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_rate=dropout_rate_subsequent)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_rate=dropout_rate_subsequent)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512 * block.expansion, embedding_size)
        self.fc2 = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: type, planes: int, blocks: int, stride: int = 1, dropout_rate: float = 0.0) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer, dropout_rate=dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward_features_till_layer2(self, x_ppg: Tensor, x_vpg: Tensor, x_apg: Tensor) -> Tensor:
        ppg_embed = self.stem_ppg(x_ppg)
        vpg_embed = self.stem_vpg(x_vpg)
        apg_embed = self.stem_apg(x_apg)

        ppg_p = ppg_embed.permute(0, 2, 1)
        vpg_p = vpg_embed.permute(0, 2, 1)
        apg_p = apg_embed.permute(0, 2, 1)
        ppg_from_v = self.co_attn_pv(ppg_p, vpg_p)
        ppg_from_a = self.co_attn_pa(ppg_p, apg_p)

        combined = torch.cat([ppg_p, ppg_from_v, ppg_from_a], dim=2)
        fused = self.fusion_conv(combined.permute(0, 2, 1))

        x = self.bn1(fused)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward(self, x_ppg: Tensor, x_vpg: Tensor, x_apg: Tensor) -> Tensor:
        x = self.forward_features_till_layer2(x_ppg, x_vpg, x_apg)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.fc1(x)
        return self.fc2(torch.relu(emb))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class TaskModel(nn.Module):
    def __init__(self, num_classes: int = 4, feature_dim: int = 256, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.feature_extractor1 = MultiModalResNet18(num_classes=feature_dim, dropout_rate=dropout_rate)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x_tuple: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        x_ppg, x_vpg, x_apg = x_tuple
        shared_features = self.feature_extractor1(x_ppg, x_vpg, x_apg)
        return self.classifier(shared_features)


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        num_frozen_layers: int = 0,
        embedding_size: int = 64,
        reduction_ratio: int = 16,
        adapt_conv_out_channels: int = 128,
        dropout_rate_first: float = 0.5,
        dropout_rate_subsequent: float = 0.25,
        pretrained_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.ssl_model = MultiModalResNet18(
            num_classes=4,
            embedding_size=embedding_size,
            dropout_rate_first=dropout_rate_first,
            dropout_rate_subsequent=dropout_rate_subsequent,
        )
        self.feature_extractor1 = MultiModalResNet18(
            num_classes=4,
            embedding_size=embedding_size,
            dropout_rate_first=dropout_rate_first,
            dropout_rate_subsequent=dropout_rate_subsequent,
        )

        loaded = False
        if pretrained:
            loaded = self._load_pretrained_weights(self.ssl_model, pretrained_checkpoint)

        if loaded and num_frozen_layers > 0:
            self._freeze_ssl_groups(num_frozen_layers)
        elif pretrained and not loaded and num_frozen_layers > 0:
            print("[Warning] pretrained=True but checkpoint not loaded; NOT freezing ssl_model.")

        gate_input_channels = 128
        self.gating_network = nn.Sequential(
            nn.Conv1d(gate_input_channels, gate_input_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(gate_input_channels // 4, gate_input_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.adapt_conv = nn.Conv1d(128, adapt_conv_out_channels, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(adapt_conv_out_channels, embedding_size)
        self.stenosis_layer = nn.Linear(embedding_size, 1)
        self.regurgitation_layer = nn.Linear(embedding_size, 1)
        self.sigmoid = nn.Sigmoid()

    def _load_pretrained_weights(self, model: nn.Module, checkpoint_path: Optional[str]) -> bool:
        if not checkpoint_path:
            print("No pretrained checkpoint provided; skip loading.")
            return False

        try:
            pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
            model_dict = model.state_dict()

            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if not k.startswith('feature_extractor1.'):
                    continue
                new_k = k.replace('feature_extractor1.', '', 1)
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    filtered_dict[new_k] = v

            model_dict.update(filtered_dict)
            missing, unexpected = model.load_state_dict(model_dict, strict=False)

            print(f"Loaded {len(filtered_dict)} tensors into ssl_model from {checkpoint_path}")
            if missing:
                print(f"[Info] Missing keys (not loaded): {len(missing)}")
            if unexpected:
                print(f"[Info] Unexpected keys (ignored): {len(unexpected)}")

            return len(filtered_dict) > 0
        except Exception as exc:
            print(f"Failed to load pretrained weights: {exc}")
            return False

    def _freeze_ssl_groups(self, num_frozen_layers: int) -> None:
        modules_to_freeze = [
            (self.ssl_model.stem_ppg, self.ssl_model.stem_vpg, self.ssl_model.stem_apg),
            (self.ssl_model.co_attn_pv, self.ssl_model.co_attn_pa, self.ssl_model.fusion_conv),
            self.ssl_model.bn1,
            self.ssl_model.layer1,
            self.ssl_model.layer2,
            self.ssl_model.layer3,
            self.ssl_model.layer4,
        ]

        print(f"\nFreezing {num_frozen_layers} groups of modules...")
        for i in range(min(num_frozen_layers, len(modules_to_freeze))):
            module_group = modules_to_freeze[i]
            if isinstance(module_group, tuple):
                for module in module_group:
                    for param in module.parameters():
                        param.requires_grad = False
            else:
                for param in module_group.parameters():
                    param.requires_grad = False

    def forward(self, x_tuple: Tuple[Tensor, Tensor, Tensor]):
        x_ppg, x_vpg, x_apg = x_tuple
        ssl_layer2 = self.ssl_model.forward_features_till_layer2(x_ppg, x_vpg, x_apg)
        current_layer2 = self.feature_extractor1.forward_features_till_layer2(x_ppg, x_vpg, x_apg)

        gate_weights = self.gating_network(current_layer2)
        enhanced_current = current_layer2 * gate_weights
        fused = ssl_layer2 + enhanced_current

        adapted = self.adapt_conv(fused)
        x = self.avgpool(adapted)
        x = torch.flatten(x, 1)
        emb = torch.relu(self.fc(x))
        stenosis = self.sigmoid(self.stenosis_layer(emb))
        regurgitation = self.sigmoid(self.regurgitation_layer(emb))
        return stenosis, regurgitation