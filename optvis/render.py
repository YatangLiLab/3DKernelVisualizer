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

import warnings
from collections import OrderedDict
import numpy as np
from tqdm import tqdm, trange
from PIL import Image
import torch
import os
import imageio
from optvis import objectives, transform, param
from misc.io import show
from util import norm


def render_vis(
    model,
    objective_f,
    model_name,
    param_f=None,
    optimizer=None,
    transforms=None,
    thresholds=(512,),
    verbose=False,
    preprocess=True,
    progress=True,
    show_image=False,
    save_image=False,
    image_name=None,
    show_inline=False,
    fixed_image_size=None,
    return_param=False,
    transform3d=False,
    t=0,
    size=64,
):
    if param_f is None:
        param_f = lambda: param.image(size, t=t, device=next(model.parameters()).device)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = transform.transforms3d if transform3d else transform.standard_transforms
    transforms = transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            if transform3d:
                transforms.append(transform.normalize3d())
            else:
                # Assume we use normalization for torchvision.models
                # See https://pytorch.org/docs/stable/torchvision/models.html
                transforms.append(transform.normalize())

    # Upsample images smaller than 224
    image_shape = image_f().shape
    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[4] < 224 or image_shape[3] < 224:
        if transform3d:
            new_size = (32, 224, 224)
        else:
            new_size = (224, 224)
    else:
        new_size = None
    if new_size:
        transforms.append(torch.nn.Upsample(size=new_size, mode="trilinear" if transform3d else "bilinear", align_corners=True))

    if model_name == 'flownet':
        transforms.append(transform.flownet_rearange())
    if transform3d and model_name == 'i3d':
        transforms.append(transform.slowfast_wrapper())

    transform_f = transform.compose(transforms)

    hook, features = hook_model(model, image_f, return_hooks=True)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        if transform3d:
            print(transform_f(image_f())[0].shape if model_name == 'i3d' else transform_f(image_f()).shape)
            model(transform_f(image_f()))
        else:
            print(transform.rearrange_transform(image_f(), transform_f, squeeze_nt=model_name=='flownet', warp_list=model_name=='i3d').shape)
            model(transform.rearrange_transform(image_f(), transform_f, squeeze_nt=model_name=='flownet', warp_list=model_name=='i3d'))
        print("Initial loss: {:.3f}".format(objective_f(hook)))

    images = []
    params_result = dict()
    losses = []
    try:
        for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
            def closure():
                optimizer.zero_grad()
                try:
                    if transform3d:
                        img_input = transform_f(image_f())
                        model(img_input)
                    else:
                        img_input = transform.rearrange_transform(image_f(), transform_f, squeeze_nt=model_name=='flownet', warp_list=model_name=='i3d')
                        model(img_input)
                except RuntimeError as ex:
                    if i == 1:
                        # Only display the warning message
                        # on the first iteration, no need to do that
                        # every iteration
                        warnings.warn(
                            "Some layers could not be computed because the size of the "
                            "image is not big enough. It is fine, as long as the non"
                            "computed layers are not used in the objective function"
                            f"(exception details: '{ex}')"
                        )
                loss = objective_f(hook)
                loss.backward()
                return loss

            loss = closure()
            losses.append(loss.item())
            optimizer.step()
            if i in thresholds:
                image = tensor_to_img_array(image_f())
                if verbose:
                    print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                    if show_inline:
                        show(image)
                images.append(image)
                if return_param:
                    params_result['params'] = []
                    for p in params:
                        params_result['params'].append(p.detach().cpu().numpy().copy())
                    params_result['loss'] = losses
            if progress and i%10 == 0:
                print(loss)
    except KeyboardInterrupt:
        print("Interrupted optimization at step {:d}.".format(i))
        if verbose:
            print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
        images.append(tensor_to_img_array(image_f()))
        if return_param:
            params_result.append(params.detach.cpu().numpy().copy())

    # Clear hooks
    for module_hook in features.values():
        if module_hook.module is None:
            continue
        del module_hook.module._forward_hooks[module_hook.hook.id]

    if save_image:
        export(image_f(), image_name)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())

    if return_param:
        return images, params_result
    return images


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    if len(image.shape) == 4:
        image = np.transpose(image, [0, 2, 3, 1])
    elif len(image.shape) == 5:
        image = np.transpose(image, [0, 2, 3, 4, 1])
    # Check if the image is single channel and convert to 3-channel
    if len(image.shape) == 4 and image.shape[3] == 1:  # Single channel image
        image = np.repeat(image, 3, axis=3)
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze(2)
    Image.fromarray(image).show()


def export(tensor, image_name=None):
    image_name = image_name or "image.jpg"
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).save(image_name)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        # This doesn't actually do anything
        self.hook.remove()


def hook_model(model, image_f, return_hooks=False):
    features = OrderedDict()

    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    if return_hooks:
        return hook, features
    return hook


def render_save(model, layer, model_name, output_dir, temporal_size=16, spatial_size=128, filter=0, transform3d=True,
                      verbose=False, fixed_image_size=None, thresholds=(1500,)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = render_vis(model, f'{layer}:{filter}', model_name=model_name, t=temporal_size, size=spatial_size, show_inline=False,
                       thresholds=thresholds, progress=verbose,
                       transform3d=transform3d, verbose=verbose, fixed_image_size=fixed_image_size)[0]

    images = (norm(images[0])*255).astype(np.uint8)
    path = os.path.join(output_dir, f'{model_name}_{layer}_{filter}_stage1.gif')
    imageio.mimsave(path, images, duration=0.1, loop=0)
    return images, path
