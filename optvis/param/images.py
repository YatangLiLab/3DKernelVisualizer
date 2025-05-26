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

"""High-level wrapper for paramaterizing images."""

from __future__ import absolute_import, division, print_function

from optvis.param.spatial import *
from optvis.param.color import to_valid_rgb
from util import DEFAULT_DEVICE


def image(w, h=None, sd=None, batch=None, decorrelate=True,
          channels=None, device=DEFAULT_DEVICE, t=0):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    if t == 0:
        shape = [batch, ch, h, w]
        param_f = pixel_image
    else:
        shape = [batch, ch, t, h, w]
        param_f = image3d

    params, image_f = param_f(shape, sd=sd, device=device)
    if channels:
        output = to_valid_rgb(image_f, decorrelate=False)
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output
