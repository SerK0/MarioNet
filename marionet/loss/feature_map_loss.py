import typing as tp
import torch
import torch.nn as nn


class FeatureMapLoss:
    """
    Feature map loss.
    Takes lists of feature maps as input and returns sum loss over all of them.
    """

    def __init__(
        self,
        criterion: tp.Optional[
            tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ) -> None:
        """
        :param tp.Optional[LossFunction] criterion: individual feature map loss
                                                    if not specified defaults to nn.L1Loss()
        :returns: None
        """
        if criterion is None:
            criterion = nn.L1Loss()

        self.criterion = criterion

    def __call__(
        self,
        output_feature_maps: tp.List[torch.Tensor],
        target_feature_maps: tp.List[torch.Tensor],
    ) -> torch.Tensor:
        """
        :param tp.List[torch.Tensor] output_feature_maps: output feature maps
        :param tp.List[torch.Tensor] target_feature_maps: target feature maps
        :returns: sum loss
        :rtype: torch.Tensor
        """
        return sum(
            self.criterion(output_feature_map, target_feature_map)
            for output_feature_map, target_feature_map in zip(
                output_feature_maps, target_feature_maps
            )
        )
