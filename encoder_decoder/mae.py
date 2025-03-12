from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining
import torch
import torch.nn as nn
import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

class MAEEncoder(nn.Module):
    def __init__(self, config):
        super(MAEEncoder, self).__init__()
        self.config = config
        self.encoder = ViTMAEForPreTraining(config).vit

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        outputs = self.encoder(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        return outputs
    
    
class MAEDecoder(nn.Module):
    def __init__(self, config):
        super(MAEDecoder, self).__init__()
        self.config = config
        self.decoder = ViTMAEForPreTraining(config).decoder

    def forward(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        interpolate_pos_encoding: bool = False,
    ):
        decoder_outputs = self.decoder(hidden_states,
                                       ids_restore, 
                                       interpolate_pos_encoding=interpolate_pos_encoding
                                       )
        return decoder_outputs
    
def patchify(pixel_values, interpolate_pos_encoding: bool = False, config: ViTMAEConfig = None):
    """
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values.
        interpolate_pos_encoding (`bool`, *optional*, default `False`):
            interpolation flag passed during the forward pass.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
            Patchified pixel values.
    """
    patch_size, num_channels = config.patch_size, config.num_channels
    # sanity checks
    if not interpolate_pos_encoding and (
        pixel_values.shape[2] != pixel_values.shape[3] or pixel_values.shape[2] % patch_size != 0
    ):
        raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
    if pixel_values.shape[1] != num_channels:
        raise ValueError(
            "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
        )

    # patchify
    batch_size = pixel_values.shape[0]
    num_patches_h = pixel_values.shape[2] // patch_size
    num_patches_w = pixel_values.shape[3] // patch_size
    patchified_pixel_values = pixel_values.reshape(
        batch_size, num_channels, num_patches_h, patch_size, num_patches_w, patch_size
    )
    patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
    patchified_pixel_values = patchified_pixel_values.reshape(
        batch_size, num_patches_h * num_patches_w, patch_size**2 * num_channels
    )
    return patchified_pixel_values

def forward_loss(pixel_values, pred, mask, interpolate_pos_encoding: bool = False, config: ViTMAEConfig = None):
    """
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values.
        pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
            Predicted pixel values.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        interpolate_pos_encoding (`bool`, *optional*, default `False`):
            interpolation flag passed during the forward pass.

    Returns:
        `torch.FloatTensor`: Pixel reconstruction loss.
    """
    target = patchify(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, config=config)
    if config.norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss