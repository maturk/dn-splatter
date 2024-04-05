# Evaluation instructions

We include models for evaluating rgb, depth and/or mesh metrics using various method.
The following methods are supported.

## Methods
### Nerfacto
RGB only supervision
```bash
ns-train gnerfacto --data [PATH] dtu/nrgbd/replica/scannet/mushroom
```
### Depth-Nerfacto
RGB and Depth supervision
```bash
ns-train gdepthfacto --data [PATH] dtu/nrgbd/replica/scannet/mushroom
```
### NeusFacto
RGB, Depth, and Normal supervision
```bash
ns-train gneusfacto --data [PATH] sdfstudio-data --include-mono-prior True
```

### Evaluation metrics
To evaluate depth,rgb image and point cloud metrics, run the following command:

```bash
ns-eval --load-config [PATH_TO_YAML] --output-path [PATH_TO_JSON]
```

# SDFStudio Instrucitons
## Train on data downloaded with ns-download-data sdfstudio
SDFStudio data can be correctly loaded with the `gsdf` dataparser.
If using gneusfacto, remember to correctly set the `--load-for-sdfstudio True` flag in gsdf. It is default False.
```bash
ns-train gneusfacto --data ./datasets/DTU/scan65/ gsdf --load-for-sdfstudio True
```

For dn_splatter, run:
```bash
ns-train dn_splatter --pipeline.model.use-depth-loss False  --data ./datasets/DTU/scan65 gsdf
```

## Convert non-sdfstudio datasets to SDFStudio format
Transfer from nerfstudio data format to sdfstudio data format
```bash
python dn_splatter/eval/nerfstudio_to_sdfstudio.py --input_dir [DATA_PATH] \
--output_dir [OUTPUT_PATH] \
--data-type colmap --scene-type indoor --normal_from_pretrain/normal_from_depth
```

## Mesh Metrics
We report the following metrics
```
accuracy  # lower better
completeness  # lower better
chamferL1  # lower better
normals_correctness  # higher better
F-score # higher better
```
Run with
```bash
python dn_splatter/eval/eval_mesh.py --path-to-pd-mesh [PATH_TO_PREDICTED_PLY] --path-to-gt-mesh [PATH_TO_GT_PLY]
```
