import torch
import torch.nn.functional as F
from torch.optim import Adam
import imageio
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from util import norm


def deform_image2d(shape, sd=None, device='cuda:0'):
    sd = sd or 0.01
    flow_sd = 0.01
    N, C, T, H, W = shape

    tensor = (torch.randn(N, C, H, W) * sd).to(device).requires_grad_(True)
    flow_field = (torch.randn(2, N, T, H, W) * flow_sd).to(device).requires_grad_(True)

    def inner():
        grid_x, grid_y = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid_x = grid_x.float().reshape(1, H, W).repeat(N, 1, 1).to(device)
        grid_y = grid_y.float().reshape(1, H, W).repeat(N, 1, 1).to(device)

        deformed_image = []
        for t in range(T):
            grid_x += flow_field[0,:,t] * H
            grid_y += flow_field[1,:,t] * W

            grid_x = 2.0 * grid_x / (W - 1) - 1.0
            grid_y = 2.0 * grid_y / (H - 1) - 1.0

            grid = torch.stack((grid_x, grid_y), dim=-1)

            deformed_image.append(F.grid_sample(tensor, grid, mode='bilinear', padding_mode='zeros'))
        deformed_image = torch.stack(deformed_image, dim=0).permute(1, 2, 0, 3, 4)
        deformed_color_image = deformed_image
        return deformed_color_image

    return [tensor, flow_field], inner


def total_variation_loss(x):
    if len(x.shape) == 5:
        diff_x = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        diff_y = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        diff_t = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])

        loss = torch.mean(diff_x) + torch.mean(diff_y) + torch.mean(diff_t)
    else:
        diff_x = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_y = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        loss = torch.mean(diff_x) + torch.mean(diff_y)
    return loss


def load_gif(path=None, gif=None):
    if gif is None and path:
        gif = imageio.mimread(path)
    video = torch.tensor(gif).float()
    video /= 255
    video -= 0.5
    video = video.permute(3, 0, 1, 2).unsqueeze(0)
    return gif, video


def get_decomp(video, device='cuda:0', num_epochs=2000, verbose=False):
    video = video.to(device)
    param, imagef = deform_image2d(video.shape)

    optimizer = Adam(param, lr=0.1)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        reconstructed_video = imagef()

        loss_reconstruction = F.mse_loss(reconstructed_video, video)
        loss_tv = total_variation_loss(param[1])
        loss_flow_l2 = ((param[1]).abs()).mean()
        loss_tv2 = total_variation_loss(param[0])
        loss_l2 = ((param[0] - video[:, :, 0]) ** 2).mean()

        loss = loss_reconstruction + loss_tv + loss_tv2 + loss_l2

        loss.backward()
        optimizer.step()

        if verbose and epoch % 100 == 0:
            print(
                f"Epoch {epoch}, Loss: {loss_reconstruction.item()}, {loss_tv.item()}, {loss_tv2.item()}, {loss_l2.item()}, {loss_flow_l2.item()}")

    return param, imagef


def convert_to_polar(param, sigma=1):
    dx = gaussian_filter(param[1][0].cpu().detach().squeeze().numpy(), sigma)
    dy = gaussian_filter(param[1][1].cpu().squeeze().detach().numpy(), sigma)

    deform_x = gaussian_filter(np.diff(dx), sigma)
    deform_y = gaussian_filter(np.diff(dy), sigma)
    magnitude, angle = cv2.cartToPolar(deform_x, deform_y)
    return magnitude, angle


def flow_2_hsv(angle, magnitude, scale=1, max_value=None):
    # print(np.unique(magnitude))
    # print(magnitude.max()/scale/2*128)
    if max_value is None:
        magnitude = (np.clip(magnitude / magnitude.max()*scale, 0, 1) *255).astype(np.uint8)
    else:
        magnitude = (np.clip(magnitude / magnitude.max()*scale, 0, 1) / (max_value/(magnitude.max()/scale/2*128)) *255).astype(np.uint8)
    angle = (angle / (2 * np.pi) * 179).astype(np.uint8)

    hsv = np.zeros((angle.shape[0],angle.shape[1],angle.shape[2],3), dtype=np.uint8)
    hsv[...,0] = angle
    hsv[...,1] = 255
    hsv[...,2] = magnitude
    rgb = [cv2.cvtColor(h, cv2.COLOR_HSV2RGB) for h in hsv]
    return rgb


def decompose(input_path=None, gif=None, sigma=1, scale=4, max_value=2):
    gif, video = load_gif(path=input_path, gif=gif)

    param, imagef = get_decomp(video, verbose=False)
    magnitude, angle = convert_to_polar(param, sigma=sigma)

    rgb = flow_2_hsv(angle, magnitude, scale=scale, max_value=max_value)
    rgb.insert(0, (norm(param[0][0].cpu().detach().permute(1,2,0).numpy())*255).astype(np.uint8))

    out_name = '.'.join(input_path.split('.')[:-1])
    imageio.mimsave(out_name+'_deform.gif', rgb, duration=0.1, loop=0)
    imageio.mimsave(out_name+'_recon.gif', (norm(imagef()[0].permute(1,2,3,0).cpu().detach().numpy())*255).astype(np.uint8), duration=0.1, loop=0)
