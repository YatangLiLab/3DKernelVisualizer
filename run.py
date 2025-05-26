import yaml
import util
from optvis import render, decomposer
from modelzoo.util import load_model
from tqdm import tqdm
import argparse


device = util.DEFAULT_DEVICE

with open("modelzoo/config.yaml") as f:
    cfg = yaml.safe_load(f)


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model name in ["i3d", "c3d", "vqvae", "flownet"]')
parser.add_argument('--output_path', help='output path', default='output')
args = parser.parse_args()

model = load_model(args.model, device=device, **cfg[args.model])


for layer in tqdm(cfg[args.model]['layers']):
    stage1, path = render.render_save(model, layer, args.model, args.output_path, filter=0, verbose=False)
    print(path)

    # for best visualization, please assign sigma, scale, max_value for each kernel of deformation
    decomposer.decompose(gif=stage1, input_path=path, sigma=1, scale=4, max_value=2)

