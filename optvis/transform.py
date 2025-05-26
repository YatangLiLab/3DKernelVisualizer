# Copyright 2020 The Lucent Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
import numpy as np
import kornia
from kornia.geometry.transform import translate
from einops import rearrange
from kornia.geometry.transform import get_affine_matrix3d, warp_affine3d

try:
    from kornia import warp_affine, get_rotation_matrix2d
except ImportError:
    from kornia.geometry.transform import warp_affine, get_rotation_matrix2d



KORNIA_VERSION = kornia.__version__


def jitter(d):
    assert d > 1, "Jitter parameter d must be more than 1, currently {}".format(d)

    def inner(image_t):
        dx = np.random.choice(d)
        dy = np.random.choice(d)
        return translate(image_t, torch.tensor([[dx, dy]]).float().to(image_t.device))

    return inner


def pad(w, mode="reflect", constant_value=0.5):
    if mode != "constant":
        constant_value = 0

    def inner(image_t):
        return F.pad(image_t, [w] * 4, mode=mode, value=constant_value,)

    return inner


def random_scale(scales):
    def inner(image_t):
        scale = np.random.choice(scales)
        shp = image_t.shape[2:]
        scale_shape = [int(_roundup(scale * d)) for d in shp]
        pad_x = max(0, _roundup((shp[1] - scale_shape[1]) / 2))
        pad_y = max(0, _roundup((shp[0] - scale_shape[0]) / 2))
        upsample = torch.nn.Upsample(
            size=scale_shape, mode="bilinear", align_corners=True
        )
        return F.pad(upsample(image_t), [pad_y, pad_x] * 2)

    return inner


def random_rotate(angles, units="degrees"):
    def inner(image_t):
        b, _, h, w = image_t.shape
        # kornia takes degrees
        alpha = _rads2angle(np.random.choice(angles), units)
        angle = torch.ones(b) * alpha
        if KORNIA_VERSION < '0.4.0':
            scale = torch.ones(b)
        else:
            scale = torch.ones(b, 2)
        center = torch.ones(b, 2)
        center[..., 0] = (image_t.shape[3] - 1) / 2
        center[..., 1] = (image_t.shape[2] - 1) / 2
        M = get_rotation_matrix2d(center, angle, scale).to(image_t.device)
        rotated_image = warp_affine(image_t.float(), M, dsize=(h, w))
        return rotated_image

    return inner


def compose(transforms):
    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return inner


def _roundup(value):
    return np.ceil(value).astype(int)


def _rads2angle(angle, units):
    if units.lower() == "degrees":
        return angle
    if units.lower() in ["radians", "rads", "rad"]:
        angle = angle * 180.0 / np.pi
    return angle


def normalize():
    # ImageNet normalization for torchvision models
    # see https://pytorch.org/docs/stable/torchvision/models.html
    normal = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def inner(image_t):
        return torch.stack([normal(t) for t in image_t])

    return inner


def preprocess_inceptionv1():
    # Original Tensorflow's InceptionV1 model
    # takes in [-117, 138]
    # See https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/other_models/InceptionV1.py#L56
    # Thanks to ProGamerGov for this!
    return lambda x: x * 255 - 117


standard_transforms = [
    pad(12, mode="constant", constant_value=0.5),
    jitter(8),
    random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
    random_rotate(list(range(-10, 11)) + 5 * [0]),
    jitter(4),
]

def identity(x):
    return x

def rearrange_transform(tensor, transforms, squeeze_nt=False, warp_list=False):
    n, c, t = tensor.shape[:3]
    tmp = rearrange(tensor, 'n c t h w -> (n t) c h w')
    if squeeze_nt:
        tmp = transforms(tmp).unsqueeze(0)
    else:
        tmp = rearrange(transforms(tmp), '(n t) c h w -> n c t h w', n=n, c=c, t=t)

    if warp_list:
        return [tmp]
    return tmp

def pad3d(w, t, mode="reflect", constant_value=0.5):
    if mode != "constant":
        constant_value = 0

    def inner(image_t):
        return F.pad(image_t, ([w] * 4)+([t]*2), mode=mode, value=constant_value,)

    return inner

def jitter_scale3d(jitter_s, jitter_t, scales):
    def inner(image_t):
        dx = np.random.choice(jitter_s)
        dy = np.random.choice(jitter_s)
        dz = np.random.choice(jitter_t)
        trans = torch.tensor([[dx, dy, dz]])


        scale = np.random.choice(scales)
        scale = torch.tensor([scale])

        affine = get_affine_matrix3d(translations=trans, center=torch.tensor([[0., 0., 0.]]),
                                     scale=scale, angles=torch.tensor([[0., 0., 0.]]))

        shape = image_t.shape[-3:]
        return warp_affine3d(image_t, affine[:,:-1].to(image_t.device), shape, align_corners=False)

    return inner


transforms3d = [
    jitter_scale3d(8, 4, [1 + (i - 5) / 50.0 for i in range(11)])
]

def normalize3d():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    def inner(image_t):
        dtype = image_t.dtype
        m = torch.as_tensor(mean, dtype=dtype, device=image_t.device).reshape(1,3,1,1,1)
        s = torch.as_tensor(std, dtype=dtype, device=image_t.device).reshape(1,3,1,1,1)
        return (image_t-m) / s

    return inner

def flownet_rearange():
    def inner(image_t):
        return rearrange(image_t, 't c h w -> (t c) h w')
    return inner

def slowfast_wrapper():
    def inner(image_t):
        return [image_t]
    return inner