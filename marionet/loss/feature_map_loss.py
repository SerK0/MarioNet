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
        reenacted_feature_maps: tp.List[torch.Tensor],
        driver_feature_maps: tp.List[torch.Tensor],
    ) -> torch.Tensor:
        """
        :param tp.List[torch.Tensor] reenacted_feature_maps: feature maps
            on reenacted image (i.e. generator output)
        :param tp.List[torch.Tensor] driver_feature_maps: feature maps
            on driver (i.e. ground-truth) image
        :returns: sum loss
        :rtype: torch.Tensor
        """
        return sum(
            self.criterion(reenacted_feature_map, driver_feature_map)
            for reenacted_feature_map, driver_feature_map in zip(
                reenacted_feature_maps, driver_feature_maps
            )
        )
