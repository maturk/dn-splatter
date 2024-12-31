# Evaluation instructions

In this document we briefly describe the evaluation protocol and methods used in the DN-Splatter project. 

- Under `dn_splatter/eval/` you will find various scripts for rgb, depth, mesh, and normal evaluation. 
- Under `dn_splatter/eval/baseline_models` you will find the configuration and model files for baseline models: Nerfacto, Depth-Nerfacto, and Neusfacto as well as some convenient scripts for converting between Nerfstudio and SDFStudio dataset formats. 

## Computing Evaluation Metrics
We include models for evaluating rgb, depth and/or mesh metrics using various method.
The following methods are supported.

### Mesh Metrics

We report the following metrics
```
accuracy  # lower better
completeness  # lower better
chamferL1  # lower better
normals_correctness  # higher better
F-score # higher better
```

Evaluate mesh metrics and cull the predicted mesh based on training camera view visibility. Regions of the mesh not seen in the training dataset are ignored when computing mesh metrics.

Run with:
```bash
python dn_splatter/eval/eval_mesh_vis_cull.py --path-to-pd-mesh [PATH_TO_PREDICTED_PLY] --path-to-gt-mesh [PATH_TO_GT_PLY]
```
For MuSHRoom dataset, use:
```bash
python dn_splatter/eval/eval_mesh_mushroom_vis_cull.py --path-to-pd-mesh [PATH_TO_PREDICTED_PLY] --path-to-gt-mesh [PATH_TO_GT_PLY]
```

### RGB/Depth metrics
To evaluate depth estimation and novel-view rgb metrics, run the following command:

```bash
ns-eval --load-config [PATH_TO_YAML] --output-path [PATH_TO_JSON]
```

## Running Baseline Models
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
python dn_splatter/eval/baseline_models/nerfstudio_to_sdfstudio.py --input_dir [DATA_PATH] \
--output_dir [OUTPUT_PATH] \
--data-type colmap --scene-type indoor --normal_from_pretrain/normal_from_depth
```

