# Feature Visualization in 3D Convolutional Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2505.07387-b31b1b.svg?style=flat)](https://arxiv.org/abs/2505.07387)  
Chunpeng Li, Ya-tang Li  

Official implementation of the paper **"Feature Visualization in 3D Convolutional Neural Networks"**, which can disentangle texture and motion preferences of a target 3D conv kernel with a data-driven decomposition method. It enables interpretable kernel visualizations for 3D models such as I3D, C3D, and 3D VQ-VAE.


---

## 🛠 Installation
Clone the repository and install the minimal dependencies:
```bash
git clone https://github.com/YatangLiLab/3DKernelVisualizer.git
cd 3DKernelVisualizer
pip install -r requirements.txt
```
#### 🔌 Optional: Install Extra Dependencies for Specific Models
To support different models, please install their dependencies manually.  
- For **C3D**:  
    There is no extra library needed.  
    Download the [pretrained weights](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle) and place it under the `modelzoo/pretrained/` directory.
- For **I3D**  
    Follow [INSTALL.md](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md) of SlowFast  
    Download the [pretrained weights](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_8x8_R50.pkl)  
    Change the config `modelzoo/SlowFast/configs/Kinetics/c2/I3D_8x8_R50.yaml`  
    ```yaml
      TRAIN:
        CHECKPOINT_FILE_PATH: # path for download pretrained weights
      NUM_GPUS: 1
    ```
- For **3D VQ-VAE**  
    Follow [VideoGPT](https://github.com/wilson1yan/VideoGPT) for extra dependencies.  
    Download the [pretrained weights](https://drive.google.com/uc?id=1uuB_8WzHP_bbBmfuaIV7PK_Itl3DyHY5) and place it under the `modelzoo/pretrained/` directory.


## 🚀 Usage
To visualize:

```bash
python run.py --model i3d --output_path ./output
```

See modelzoo/config.yaml for model-specific options.

####  📁 Output Files
For each convolutional kernel, the visualization process produces 3 output files, named as `{model_name}_{layer_name}_{filter_index}_{type}.gif`  

| Suffix        | Description                                                                      |
|---------------|----------------------------------------------------------------------------------|
| `_stage1.gif` | **Stage 1 result**: the initial optimal input of target kernel                   |
| `_deform.gif` | **Decomposition**: the second-stage result showing static and dynamic components |
| `_recon.gif`  | **Reconstruction**: the reconstructed result from decomposed components          |

All outputs are saved under the `--output_path` directory.

> 💡 Tip: for best visualization, please assign `sigma`, `scale`, `max_value` in `run.py` for each kernel of deformation.


## 📈 Results  
Loading may take some time...


- #### Visualization of I3D 

| layer   | example 1                                                                                                                                                             | example 2                                                                                                                                                             | example 3                                                                                                                                                             |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| stage 1 | ![I3D s1_pathway0_stem_conv kernel 0 static](results/i3d/i3d_stage1_f0_static.png )<br/>![I3D s1_pathway0_stem_conv kernel 0](results/i3d/i3d_stage1_f0.gif)          | ![I3D s1_pathway0_stem_conv kernel 1 static](results/i3d/i3d_stage1_f1_static.png) <br/>![I3D s1_pathway0_stem_conv kernel 1](results/i3d/i3d_stage1_f1.gif)          | ![I3D s1_pathway0_stem_conv kernel 8 static](results/i3d/i3d_stage1_f8_static.png) <br/>![I3D s1_pathway0_stem_conv kernel 8](results/i3d/i3d_stage1_f8.gif)          |
| stage 2 | ![I3D s2_pathway0_res0_branch2_c kernel 2 static](results/i3d/i3d_stage2_f2_static.png)<br/>![I3D s2_pathway0_res0_branch2_c kernel 2](results/i3d/i3d_stage2_f2.gif) | ![I3D s2_pathway0_res0_branch2_c kernel 3 static](results/i3d/i3d_stage2_f3_static.png)<br/>![I3D s2_pathway0_res0_branch2_c kernel 3](results/i3d/i3d_stage2_f3.gif) | ![I3D s2_pathway0_res0_branch2_c kernel 8 static](results/i3d/i3d_stage2_f8_static.png)<br/>![I3D s2_pathway0_res0_branch2_c kernel 8](results/i3d/i3d_stage2_f8.gif) |
| stage 3 | ![I3D s3_pathway0_res0_branch2_c kernel 0 static](results/i3d/i3d_stage3_f0_static.png)<br/>![I3D s3_pathway0_res0_branch2_c kernel 0](results/i3d/i3d_stage3_f0.gif) | ![I3D s3_pathway0_res0_branch2_c kernel 1 static](results/i3d/i3d_stage3_f1_static.png)<br/>![I3D s3_pathway0_res0_branch2_c kernel 1](results/i3d/i3d_stage3_f1.gif) | ![I3D s3_pathway0_res0_branch2_c kernel 8 static](results/i3d/i3d_stage3_f8_static.png)<br/>![I3D s3_pathway0_res0_branch2_c kernel 8](results/i3d/i3d_stage3_f8.gif) |
| stage 4 | ![I3D s4_pathway0_res0_branch2_c kernel 0 static](results/i3d/i3d_stage4_f0_static.png)<br/>![I3D s4_pathway0_res0_branch2_c kernel 0](results/i3d/i3d_stage4_f0.gif) | ![I3D s4_pathway0_res0_branch2_c kernel 1 static](results/i3d/i3d_stage4_f1_static.png)<br/>![I3D s4_pathway0_res0_branch2_c kernel 1](results/i3d/i3d_stage4_f1.gif) | ![I3D s4_pathway0_res0_branch2_c kernel 8 static](results/i3d/i3d_stage4_f8_static.png)<br/>![I3D s4_pathway0_res0_branch2_c kernel 8](results/i3d/i3d_stage4_f8.gif) |
| stage 5 | ![I3D s5_pathway0_res0_branch2_c kernel 0 static](results/i3d/i3d_stage5_f0_static.png)<br/>![I3D s5_pathway0_res0_branch2_c kernel 0](results/i3d/i3d_stage5_f0.gif) | ![I3D s5_pathway0_res0_branch2_c kernel 1 static](results/i3d/i3d_stage5_f1_static.png)<br/>![I3D s5_pathway0_res0_branch2_c kernel 1](results/i3d/i3d_stage5_f1.gif) | ![I3D s5_pathway0_res0_branch2_c kernel 4 static](results/i3d/i3d_stage5_f4_static.png)<br/>![I3D s5_pathway0_res0_branch2_c kernel 4](results/i3d/i3d_stage5_f4.gif) |


- #### Visualization of C3D 

| layer   | example 1                                                                                                                     | example 2                                                                                                                     | example 3                                                                                                                     |
|---------|-------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| conv 1  | ![C3D conv 1 kernel 1 static](results/c3d/c3d_conv1_f1_static.png )<br/>![C3D conv 1 kernel 1](results/c3d/c3d_conv1_f1.gif)  | ![C3D conv 1 kernel 4 static](results/c3d/c3d_conv1_f4_static.png) <br/>![C3D conv 1 kernel 4](results/c3d/c3d_conv1_f4.gif)  | ![C3D conv 1 kernel 5 static](results/c3d/c3d_conv1_f5_static.png) <br/>![C3D conv 1 kernel 5](results/c3d/c3d_conv1_f5.gif)  |
| conv 2  | ![C3D conv 2 kernel 0 static](results/c3d/c3d_conv2_f0_static.png)<br/>![C3D conv 2 kernel 0](results/c3d/c3d_conv2_f0.gif)   | ![C3D conv 2 kernel 3 static](results/c3d/c3d_conv2_f3_static.png)<br/>![C3D conv 2 kernel 3](results/c3d/c3d_conv2_f3.gif)   | ![C3D conv 2 kernel 5 static](results/c3d/c3d_conv2_f5_static.png)<br/>![C3D conv 2 kernel 5](results/c3d/c3d_conv2_f5.gif)   |
| conv 3b | ![C3D conv 3b kernel 1 static](results/c3d/c3d_conv3_f1_static.png)<br/>![C3D conv 3b kernel 1](results/c3d/c3d_conv3_f1.gif) | ![C3D conv 3b kernel 5 static](results/c3d/c3d_conv3_f5_static.png)<br/>![C3D conv 3b kernel 5](results/c3d/c3d_conv3_f5.gif) | ![C3D conv 3b kernel 8 static](results/c3d/c3d_conv3_f8_static.png)<br/>![C3D conv 3b kernel 8](results/c3d/c3d_conv3_f8.gif) |
| conv 4b | ![C3D conv 4b kernel 1 static](results/c3d/c3d_conv4_f1_static.png)<br/>![C3D conv 4b kernel 1](results/c3d/c3d_conv4_f1.gif) | ![C3D conv 4b kernel 2 static](results/c3d/c3d_conv4_f2_static.png)<br/>![C3D conv 4b kernel 2](results/c3d/c3d_conv4_f2.gif) | ![C3D conv 4b kernel 5 static](results/c3d/c3d_conv4_f5_static.png)<br/>![C3D conv 4b kernel 5](results/c3d/c3d_conv4_f5.gif) |
| conv 5b | ![C3D conv 5b kernel 0 static](results/c3d/c3d_conv5_f0_static.png)<br/>![C3D conv 5b kernel 0](results/c3d/c3d_conv5_f0.gif) | ![C3D conv 5b kernel 3 static](results/c3d/c3d_conv5_f3_static.png)<br/>![C3D conv 5b kernel 3](results/c3d/c3d_conv5_f3.gif) | ![C3D conv 5b kernel 8 static](results/c3d/c3d_conv5_f8_static.png)<br/>![C3D conv 5b kernel 8](results/c3d/c3d_conv5_f8.gif) |



- #### Visualization of VQ-VAE 

| layer    | example 1                                                                                                                                                                             | example 2                                                                                                                                                                             | example 3                                                                                                                                                                             |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| enc_c0   | ![VQ-VAE encoder_convs_0 kernel 0 static](results/vqvae/vqvae_conv0_f0_static.png )<br/>![VQ-VAE encoder_convs_0 kernel 0](results/vqvae/vqvae_conv0_f0.gif)                          | ![VQ-VAE encoder_convs_0 kernel 2 static](results/vqvae/vqvae_conv0_f2_static.png) <br/>![VQ-VAE encoder_convs_0 kernel 2](results/vqvae/vqvae_conv0_f2.gif)                          | ![VQ-VAE encoder_convs_0 kernel 3 static](results/vqvae/vqvae_conv0_f3_static.png) <br/>![VQ-VAE encoder_convs_0 kernel 3](results/vqvae/vqvae_conv0_f3.gif)                          |
| enc_s3b5 | ![VQ-VAE encoder_res_stack_3_block_5 kernel 2 static](results/vqvae/vqvae_block3_f2_static.png)<br/>![VQ-VAE encoder_res_stack_3_block_5 kernel 2](results/vqvae/vqvae_block3_f2.gif) | ![VQ-VAE encoder_res_stack_3_block_5 kernel 3 static](results/vqvae/vqvae_block3_f3_static.png)<br/>![VQ-VAE encoder_res_stack_3_block_5 kernel 3](results/vqvae/vqvae_block3_f3.gif) | ![VQ-VAE encoder_res_stack_3_block_5 kernel 4 static](results/vqvae/vqvae_block3_f4_static.png)<br/>![VQ-VAE encoder_res_stack_3_block_5 kernel 4](results/vqvae/vqvae_block3_f4.gif) |


## 📑 Citation

If you find this code helpful, please cite our paper:
```BibTeX
@inproceedings{li20253dkernelvis,
      title={Feature Visualization in 3D Convolutional Neural Networks}, 
      author={Chunpeng Li and Ya-tang Li},
      booktitle={Lecture Notes in Computer Science},
      volume={15862},
      pages={402--412},
      year={2025},
      publisher={Springer},
}
```

## 🙏 Acknowledgements

This codebase draws inspiration and reuses components from [Lucent](https://github.com/greentfrapp/lucent).  
We also acknowledge the use of pretrained weights from 
[SlowFast](https://github.com/facebookresearch/SlowFast), 
[VideoGPT](https://github.com/wilson1yan/VideoGPT), 
[C3D (PyTorch)](https://github.com/DavideA/c3d-pytorch), 
[FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)

## 📜 License
This project is under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
