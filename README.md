# AGS-Mesh: Adaptive Gaussian Splatting and Meshing with Geometric Priors for Indoor Room Reconstruction Using Smartphones

### <p align="center">[🌐Project Page](https://github.com/XuqianRen/ags_mesh_website) | [🖨️ArXiv]() </p>


<p align="center">
    <img src="assets/pipeline_ags_mesh.png" alt="Pipeline" width="600"/>
</p>

## Installation

```bash
# download
git clone https://github.com/maturk/dn-splatter.git --recursive

#create a new environment
git checkout ags-mesh-2dgs
conda env create --file environment.yml
conda activate ags_mesh

# install additional package:
pip install fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl

# install IsoOctree library
git clone https://github.com/SpectacularAI/IsoOctree.git --recursive
cd IsoOctree

../install_IsoOctree.sh

cd ..
```

## Data Preparation
```bash
# First generate depth mask based on depth normal and pre-train normal consistency check:
python depth_normal_consistency.py --data_dir <dataset_path> --transform_name <transformation script.json>

# this command will generate depth_normals and depth_normals_mask for the RGBD sequences

## note: we put pose in transform.json for scannetpp iphone sequences, so please first run dn-splatter with scannetpp dataset to get the transform.json

# Second, please follow dn-splatter main branch to generate the pre-train normal

```
Commandline arguments you should adjust accordingly to generate depth mask
```bash
--data_dir # path to the sequence folder
--transform_name # transforms.json or transformation_colmap.json file
```

## Training
To train a scene, simply use
```bash
# currently we only support Scannetpp and MushRoom dataset
python train.py -s <path to MuSHRoom or Scannetpp dataset> 
--model_path <output_path> --depth_supervision --normal_supervision
```

## Testing

### Novel view synthesize evaluation
```bash
python render.py -m <output_path> -s <path to MuSHRoom or Scannetpp dataset>   --iteration 30000 --skip_mesh  --skip_train
```

### Bounded Mesh Extraction
To export a mesh within a bounded volume, simply use
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset> \
--iteration 30000 --voxel_size 0.01 --depth_trunc 10 --sdf_trunc 0.03 --skip_train --skip_test
```
Commandline arguments you should adjust accordingly for meshing for bounded TSDF fusion, use
```bash
--voxel_size # voxel size
--depth_trunc # depth truncation
--sdf_trunc # sdf truncation distance
```

### IsoOctree Mesh Extraction
To export a mesh with IsoOctree based method, use:
```bash
# first export rendered rgb, depth and normal of train set:
python render.py -m <output_path> -s <path to MuSHRoom or Scannetpp dataset>   --iteration 30000 --skip_mesh  --skip_test

# get mesh with IsoOctree-based method
python isooctree.py  <output_path/train/ours_iteration/> --transformation_path <pose_json_path> \
--tsdf_rel 0.03 --output_mesh_file <output_path/output.ply> --subdivision_threshold=100
```
Commandline arguments you may need to adjust accordingly for meshing to get better quality
```bash
--tsdf_rel # TSDF fusion tuning param
--disable_normals # do not use normals in TSDF fusion
--tsdf_trunc # sdf truncation distance
```


### Mesh Evaluation
To evaluate reconstructed mesh with reference mesh:
```bash
# mushroom dataset
python eval/eval_mesh_mushroom.py --gt_mesh_path <path to dataset scene> --pred_mesh_path <path to output.ply> \
--output <output path>

# scannetpp dataset

python eval/eval_mesh.py --gt_mesh <path to gt mesh> --pred_mesh <path to predicted mesh> --dataset_path <path to scannetpp sequence> --transformation_file <path to transforms.json>
```

### NVS Evaluation
To evaluate NVS quality, run
```bash
 python eval/eval_nvs.py --data <render folder path>
```


### Examples
Train with MuSHRoom/Scannetpp dataset:
```bash
# MuSHRoom
python train.py -s dataset/room_datasets/vr_room/iphone/ --model_path output/mushroom/vr_room --depth_supervision --normal_supervision --test_iterations 300 4000 7000 20000 25000 30000

# Scannetpp
python train.py -s dataset/scannetpp/data/8b5caf3398/iphone --model_path output/scannetpp/8b5caf3398/ --depth_supervision --normal_supervision


# export rendered images of test set
python render.py -m output/mushroom/vr_room -s dataset/room_datasets/vr_room/iphone/  --iteration 30000 --skip_mesh  --skip_train

# evaluate nvs
python eval/eval_nvs.py --data output/MuSHRoom/vr_room/test/test/ours_30000


# use isooctree-based mesh extraction
## first get rendered training image

### mushroom
python render.py -m output/mushroom/vr_room -s dataset/room_datasets/vr_room/iphone/  --iteration 30000 --skip_mesh  --skip_test

### scannetpp
python render.py -m output/scannetpp/8b5caf3398 -s dataset/scannetpp/data/8b5caf3398/iphone --iteration 30000 --skip_mesh  --skip_test

## get mesh with IsoOctree-based method 
### mushroom
python isooctree.py  output/MuSHRoom/vr_room/train/ours_30000/ --transformation_path dataset/room_datasets/vr_room/iphone/long_capture/transformations_colmap.json --tsdf_rel 0.03 --output_mesh_file output.ply --subdivision_threshold=100

### scannetpp
python isooctree.py  output/scannetpp/8b5caf3398/train/ours_30000 --transformation_path  dataset/scannetpp/data/8b5caf3398/iphone/transforms.json --tsdf_rel 0.03 --output_mesh_file output.ply --subdivision_threshold=100

## evaluate isooctree mesh
### mushroom
python eval/eval_mesh_mushroom.py --gt_mesh_path dataset/room_datasets/vr_room --pred_mesh_path output.ply --output ./

### scannetpp
python eval/eval_mesh.py --gt_mesh dataset/scannetpp/data/8b5caf3398/scans/mesh_aligned_0.05.ply --pred_mesh output.ply --dataset_path dataset/scannetpp/data/8b5caf3398/iphone --transformation_file dataset/scannetpp/data/8b5caf3398/iphone/transforms.json


# or use the bounded mesh extraction from 2dgs

## mushroom
python render.py -m output/mushroom/vr_room -s dataset/room_datasets/vr_room/iphone/  --iteration 30000 --voxel_size 0.01 --depth_trunc 10 --sdf_trunc 0.03 --skip_train --skip_test

## scannetpp
python render.py -m output/scannetpp/8b5caf3398 -s dataset/scannetpp/data/8b5caf3398/iphone --iteration 30000 --voxel_size 0.01 --depth_trunc 10 --sdf_trunc 0.03 --skip_train --skip_test

```

