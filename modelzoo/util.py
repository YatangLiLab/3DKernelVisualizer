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

"""Utility functions for modelzoo models."""

from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from modelzoo import C3D, VQVAE, cu, build_model, get_cfg, flownets
from collections import namedtuple
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model_layers(model, getLayerRepr=False):
    """
    If getLayerRepr is True, return a OrderedDict of layer names, layer representation string pair.
    If it's False, just return a list of layer names
    """
    layers = OrderedDict() if getLayerRepr else []
    # recursive function to get layers
    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                if getLayerRepr:
                    layers["_".join(prefix+[name])] = layer.__repr__()
                else:
                    layers.append("_".join(prefix + [name]))
                get_layers(layer, prefix=prefix+[name])

    get_layers(model)
    return layers


def load_model(model_name, device, print_layers=False, **kwargs):
    if model_name == 'c3d':
        model = C3D().to(device)
        model.load_state_dict(torch.load(os.path.join(BASE_DIR, kwargs['ckpt_path'])))


    elif model_name == 'vqvae':
        Args = namedtuple("Args", ["sequence_length", "resolution", "embedding_dim", "n_codes", "n_hiddens", "n_res_layers", "downsample"])
        args = Args(sequence_length=16, resolution=64, embedding_dim=256, n_codes=2048, n_hiddens=240, n_res_layers=4, downsample=(4, 4, 4))
        print(args)

        model = VQVAE(args).load_from_checkpoint(os.path.join(BASE_DIR, kwargs['ckpt_path'])).to(device)


    elif model_name == 'flownet':
        model = flownets(torch.load(os.path.join(BASE_DIR, kwargs['ckpt_path']))).to(device)

    elif model_name == 'i3d':
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(BASE_DIR, kwargs['cfg_path']))
        model = build_model(cfg).to(device)
        model.eval()
        cu.load_test_checkpoint(cfg, model)


    if print_layers:
        print(get_model_layers(model))
    model.eval()
    return model