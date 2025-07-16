import torch
import torch.nn as nn
from torch import Tensor
from fastmri.fftc import fft2c_new as fft2c
from fastmri.fftc import ifft2c_new as ifft2c
from fastmri.math import complex_abs, complex_conj, complex_mul

from .modules import FeatureEncoder, FeatureDecoder
from .utils import sens_expand, sens_reduce, image_uncrop, image_crop, FeatureImage
from .attention import AttentionPE
from .modules import Unet2d

class AttentionFeatureVarNetBlock(nn.Module):
    def __init__(
        self,
        encoder: FeatureEncoder,
        decoder: FeatureDecoder,
        acceleration: int,
        feature_processor: Unet2d,
        attention_layer: AttentionPE,
        use_extra_feature_conv: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_processor = feature_processor
        self.attention_layer = attention_layer
        self.use_image_conv = use_extra_feature_conv
        self.dc_weight = nn.Parameter(torch.ones(1))
        feature_chans = self.encoder.feature_chans
        self.acceleration = acceleration

        self.input_norm = nn.InstanceNorm2d(feature_chans)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if use_extra_feature_conv:
            self.output_norm = nn.InstanceNorm2d(feature_chans)
            self.output_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=feature_chans,
                    out_channels=feature_chans,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(
                    in_channels=feature_chans,
                    out_channels=feature_chans,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        self.zero: Tensor
        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))

    def encode_from_kspace(self, kspace: Tensor, feature_image: FeatureImage) -> Tensor:
        image = sens_reduce(kspace, feature_image.sens_maps)

        return self.encoder(
            image, means=feature_image.means, variances=feature_image.variances
        )

    def decode_to_kspace(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            feature_image.features,
            means=feature_image.means,
            variances=feature_image.variances,
        )

        return sens_expand(image, feature_image.sens_maps)

    def compute_dc_term(self, feature_image: FeatureImage) -> Tensor:
        est_kspace = self.decode_to_kspace(feature_image)

        return self.dc_weight * self.encode_from_kspace(
            torch.where(
                feature_image.mask, est_kspace - feature_image.ref_kspace, self.zero
            ),
            feature_image,
        )

    def apply_model_with_crop(self, feature_image: FeatureImage) -> Tensor:
        if feature_image.crop_size is not None:
            features = image_uncrop(
                self.feature_processor(
                    image_crop(feature_image.features, feature_image.crop_size)
                ),
                feature_image.features.clone(),
            )
        else:
            features = self.feature_processor(feature_image.features)

        return features

    def forward(self, feature_image: FeatureImage) -> FeatureImage:
        feature_image = feature_image._replace(
            features=self.input_norm(feature_image.features)
        )

        new_features = feature_image.features - self.compute_dc_term(feature_image)
        """
        new_features_np = feature_image.features.cpu().numpy()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'new_features_before_{timestamp}.mat'
        savemat(file_name, {'new_features_before': new_features_np})

        new_ref_kspace = feature_image.ref_kspace.cpu().numpy()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'kspace_{timestamp}.mat'
        savemat(file_name, {'kspace_': new_ref_kspace})
        """
        feature_image = feature_image._replace(
            features=self.attention_layer(feature_image.features, self.acceleration)
        )
        new_features = new_features - self.apply_model_with_crop(feature_image)

        if self.use_image_conv:
            new_features = self.output_norm(new_features)
            new_features = new_features + self.output_conv(new_features)

        return feature_image._replace(features=new_features)


class FeatureVarNetBlock(nn.Module):
    def __init__(
        self,
        encoder: FeatureEncoder,
        decoder: FeatureDecoder,
        feature_processor: Unet2d,
        use_extra_feature_conv: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_processor = feature_processor
        self.use_image_conv = use_extra_feature_conv
        self.dc_weight = nn.Parameter(torch.ones(1))
        feature_chans = self.encoder.feature_chans

        self.input_norm = nn.InstanceNorm2d(feature_chans)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if use_extra_feature_conv:
            self.output_norm = nn.InstanceNorm2d(feature_chans)
            self.output_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=feature_chans,
                    out_channels=feature_chans,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(
                    in_channels=feature_chans,
                    out_channels=feature_chans,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        self.zero: Tensor
        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))

    def encode_from_kspace(self, kspace: Tensor, feature_image: FeatureImage) -> Tensor:
        image = sens_reduce(kspace, feature_image.sens_maps)

        return self.encoder(
            image, means=feature_image.means, variances=feature_image.variances
        )

    def decode_to_kspace(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            feature_image.features,
            means=feature_image.means,
            variances=feature_image.variances,
        )

        return sens_expand(image, feature_image.sens_maps)

    def compute_dc_term(self, feature_image: FeatureImage) -> Tensor:
        est_kspace = self.decode_to_kspace(feature_image)

        return self.dc_weight * self.encode_from_kspace(
            torch.where(
                feature_image.mask, est_kspace - feature_image.ref_kspace, self.zero
            ),
            feature_image,
        )

    def apply_model_with_crop(self, feature_image: FeatureImage) -> Tensor:
        if feature_image.crop_size is not None:
            features = image_uncrop(
                self.feature_processor(
                    image_crop(feature_image.features, feature_image.crop_size)
                ),
                feature_image.features.clone(),
            )
        else:
            features = self.feature_processor(feature_image.features)

        return features

    def forward(self, feature_image: FeatureImage) -> FeatureImage:
        feature_image = feature_image._replace(
            features=self.input_norm(feature_image.features)
        )

        new_features = (
            feature_image.features
            - self.compute_dc_term(feature_image)
            - self.apply_model_with_crop(feature_image)
        )

        if self.use_image_conv:
            new_features = self.output_norm(new_features)
            new_features = new_features + self.output_conv(new_features)

        return feature_image._replace(features=new_features)


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fft2c(complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return complex_mul(ifft2c(x), complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight

        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc - model_term