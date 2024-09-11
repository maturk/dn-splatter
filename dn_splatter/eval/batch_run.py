"""Script for running dataset scenes in serial under different configurations."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import GPUtil

from dn_splatter.scripts.render_model import RenderModel


@dataclass
class Config:
    # Output settings
    output_dir: str = "experiments/"
    experiment_name: str = "splatfacto"
    timestamp: str = ""  # do not change
    # Paths
    mushroom_dataset_path: str = "./datasets/room_datasets"
    replica_dataset_path: str = ""
    scannetpp_dataset_path: str = "./datasets/scannetpp"

    # regularization strategy
    regularization_strategy: str = "dn-splatter"

    iterations: int = 30000
    # Depth configs
    use_depth_loss: bool = False
    depth_lambda: float = 0.2
    use_depth_smooth_loss: bool = False
    depth_loss_type: str = "EdgeAwareLogL1"  # MSE, LogL1, HuberL1, L1, EdgeAwareLogL1
    smooth_loss_lambda: float = 0.1
    # Normal configs
    use_normal_loss: bool = False
    predict_normals: bool = True
    normal_supervision: str = "mono"  # mono, depth
    load_pcd_normals: bool = False
    normal_lambda: float = 0.1
    use_normal_tv_loss: bool = False
    # Other configs
    use_sparse_loss: bool = False
    use_binary_opacities: bool = False
    two_d_gaussians: bool = False
    use_scale_regularization = False
    # Init seed (only for MuSHRoom atm)
    num_init_points: int = 1_000_000


# Default configuration
splatfacto_config = Config(experiment_name="splatfacto")
dn_splatter_config = Config(
    experiment_name="dn_splatter",
    use_depth_loss=True,
    use_normal_loss=True,
    load_pcd_normals=True,
    use_normal_tv_loss=True,
    normal_supervision="mono",
    two_d_gaussians=True,
)

mushroom_scenes = [
    "coffee_room",
    "honka",
    "kokko",
    "sauna",
    "computer",
    "vr_room",
]

scannetpp_scenes = ["b20a261fdf", "8b5caf3398"]

# what different runs to do
run_configs = [dn_splatter_config]


def train_scene(
    gpu,
    scene,
    config,
    dataset: Literal["mushroom", "replica", "scannetpp"] = "mushroom",
    model: Literal["dn", "nerfacto", "gdepthfacto", "splatfacto"] = "dn",
    dry_run: bool = False,
    skip_train: bool = False,
    skip_rgb_eval: bool = False,
    skip_mesh_eval: bool = False,
    use_tsdf: bool = False,
    skip_render: bool = False,
):
    SCENE = scene
    # current dataset
    if dataset == "mushroom":
        DATASET_PATH = config.mushroom_dataset_path
    elif dataset == "replica":
        DATASET_PATH = config.replica_dataset_path
    elif dataset == "scannetpp":
        DATASET_PATH = config.scannetpp_dataset_path

    # experiment paths and save directories
    if model == "dn":
        experiment_path = os.path.join(
            config.output_dir, config.experiment_name, "dn-splatter", SCENE
        )
    elif model == "nerfacto":
        experiment_path = os.path.join(
            config.output_dir, config.experiment_name, "gnerfacto", SCENE
        )
    elif model == "gdepthfacto":
        experiment_path = os.path.join(
            config.output_dir, config.experiment_name, "gdepthfacto", SCENE
        )
    elif model == "splatfacto":
        experiment_path = os.path.join(
            config.output_dir, config.experiment_name, "splatfacto", SCENE
        )
    config_path = os.path.join(experiment_path, "config.yml")
    results_path = os.path.join(experiment_path, "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    print(f"Running {SCENE}")
    if not skip_train:
        if model == "dn":
            command = [
                f"OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES={gpu}",
                "ns-train",
                "dn-splatter",
                "--timestamp",
                SCENE,
                "--output-dir",
                config.output_dir,
                "--experiment-name",
                config.experiment_name,
                "--max-num-iterations",
                str(config.iterations),
                "--viewer.quit-on-train-completion",
                "True",
                "--pipeline.model.use-depth-loss",
                str(config.use_depth_loss),
                "--pipeline.model.depth-loss-type",
                config.depth_loss_type,
                "--pipeline.model.depth-lambda",
                str(config.depth_lambda),
                "--pipeline.model.use-depth-smooth-loss",
                str(config.use_depth_smooth_loss),
                "--pipeline.model.smooth-loss-lambda",
                str(config.smooth_loss_lambda),
                "--pipeline.model.predict_normals",
                str(config.predict_normals),
                "--pipeline.model.use-normal-loss",
                str(config.use_normal_loss),
                "--pipeline.model.normal-supervision",
                config.normal_supervision,
                "--pipeline.model.normal-lambda",
                str(config.normal_lambda),
                "--pipeline.model.use-normal-tv-loss",
                str(config.use_normal_tv_loss),
                "--pipeline.model.use-sparse-loss",
                str(config.use_sparse_loss),
                "--pipeline.model.use-binary-opacities",
                str(config.use_binary_opacities),
                "--pipeline.model.two-d-gaussians",
                str(config.two_d_gaussians),
                "--pipeline.model.use-scale-regularization",
                str(config.use_scale_regularization),
                "--pipeline.model.regularization_strategy",
                str(config.regularization_strategy),
            ]

        elif model == "splatfacto":
            command = [
                f"OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES={gpu}",
                "ns-train",
                "splatfacto",
                "--timestamp",
                SCENE,
                "--output-dir",
                config.output_dir,
                "--experiment-name",
                config.experiment_name,
                "--max-num-iterations",
                str(config.iterations),
                "--viewer.quit-on-train-completion",
                "True",
            ]

        elif model == "nerfacto":
            command = [
                f"OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES={gpu}",
                "ns-train",
                "gnerfacto",
                "--timestamp",
                SCENE,
                "--output-dir",
                config.output_dir,
                "--experiment-name",
                config.experiment_name,
                "--max-num-iterations",
                str(config.iterations),
                "--viewer.quit-on-train-completion",
                "True",
            ]
        elif model == "gdepthfacto":
            command = [
                f"OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES={gpu}",
                "ns-train",
                "gdepthfacto",
                "--timestamp",
                SCENE,
                "--output-dir",
                config.output_dir,
                "--experiment-name",
                config.experiment_name,
                "--max-num-iterations",
                str(config.iterations),
                "--viewer.quit-on-train-completion",
                "True",
            ]
        if dataset == "mushroom":
            command.extend(
                [
                    "mushroom",
                    "--data",
                    f"{DATASET_PATH}/{SCENE}",
                    "--load-pcd-normals",
                    str(config.load_pcd_normals),
                    "--normal-format",
                    "opengl",
                    "--mode",
                    "iphone",
                    "--load-depth-confidence-masks",
                    str(False),
                    "--num-init-points",
                    str(config.num_init_points),
                ]
            )
        elif dataset == "scannetpp":
            command.extend(
                [
                    "scannetpp",
                    "--data",
                    f"{DATASET_PATH}",
                    "--sequence",
                    f"{SCENE}",
                    "--load-pcd-normals",
                    str(config.load_pcd_normals),
                    "--normal-format",
                    "opengl",
                    "--load-depth-confidence-masks",
                    str(False),
                ]
            )
        elif dataset == "replica":
            command.extend(
                [
                    "replica",
                    "--data",
                    f"{DATASET_PATH}",
                    "--sequence",
                    f"{SCENE}",
                    "--load-pcd-normals",
                    str(config.load_pcd_normals),
                    "--normal-format",
                    "opengl",
                ]
            )
        if model == "nerfacto" or model == "gdepthfacto":
            command.extend(["--auto-scale-poses", str(True)])

        print("command being ran: ", command)
        if not dry_run:
            start_time = time.time()
            os.system((" ").join(command))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Training time: {elapsed_time}")
            train_time_output = {"train_time": elapsed_time}
            with open(results_path / Path("train_time.json"), "w") as outfile:
                json.dump(train_time_output, outfile)

    if not skip_render:
        print("\t Rendering images to disk")
        if not dry_run:
            RenderModel(
                load_config=Path(config_path),
                output_dir=Path(os.path.join(experiment_path, "renders")),
                render_rgb=True,
                render_depth=True,
                render_normal=True,
            ).main()

    if not skip_rgb_eval:
        eval_command = [
            f" CUDA_VISIBLE_DEVICES={gpu}",
            "ns-eval",
            "--load-config",
            str(config_path),
            "--output-path",
            str(os.path.join(results_path, "metrics.json")),
        ]
        print("\t Running rgb/depth eval")
        print("command being ran: ", eval_command)
        if not dry_run:
            os.system((" ").join(eval_command))

    if not skip_mesh_eval:
        if not use_tsdf:
            mesh_command = [
                f" CUDA_VISIBLE_DEVICES={gpu}",
                "gs-mesh",
                "dn",
                "--load-config",
                str(config_path),
                "--output-dir",
                str(os.path.join(experiment_path, "mesh")),
            ]
            print("\t Extracting mesh")
            print("command being ran: ", mesh_command)
            if not dry_run:
                os.system((" ").join(mesh_command))

            if dataset == "mushroom":
                eval_mesh_command = [
                    f" CUDA_VISIBLE_DEVICES={gpu}",
                    "python",
                    "dn_splatter/eval/eval_mesh_mushroom_vis_cull.py",
                    "--gt-mesh-path",
                    str(os.path.join(DATASET_PATH, SCENE)),
                    "--pred-mesh-path",
                    str(
                        os.path.join(
                            experiment_path,
                            "mesh",
                            "DepthAndNormalMapsPoisson_poisson_mesh.ply",
                        )
                    ),
                    "--output",
                    str(os.path.join(results_path)),
                    "--output-same-as-pred-mesh",
                    str(False),
                ]
            elif dataset == "scannetpp":
                eval_mesh_command = [
                    f" CUDA_VISIBLE_DEVICES={gpu}",
                    "python",
                    "dn_splatter/eval/eval_mesh_vis_cull.py",
                    "--gt-mesh-path",
                    str(
                        os.path.join(
                            DATASET_PATH, SCENE, "scans", "mesh_aligned_0.05.ply"
                        )
                    ),
                    "--pred-mesh-path",
                    str(
                        os.path.join(
                            experiment_path,
                            "mesh",
                            "DepthAndNormalMapsPoisson_poisson_mesh.ply",
                        )
                    ),
                    "--output",
                    str(os.path.join(results_path)),
                    "--output-same-as-pred-mesh",
                    str(False),
                    "--transformation-file",
                    str(os.path.join(DATASET_PATH, SCENE, "iphone", "transforms.json")),
                    "--dataset-path",
                    str(os.path.join(DATASET_PATH, SCENE, "iphone")),
                ]
        elif use_tsdf:
            mesh_command = [
                f" CUDA_VISIBLE_DEVICES={gpu}",
                "gs-mesh",
                "o3dtsdf",
                "--load-config",
                str(config_path),
                "--output-dir",
                str(os.path.join(experiment_path, "mesh")),
            ]
            print("\t Extracting mesh")
            print("command being ran: ", mesh_command)
            if not dry_run:
                # os.system((" ").join(mesh_command))
                print("skipping this for now")

            if dataset == "mushroom":
                eval_mesh_command = [
                    f" CUDA_VISIBLE_DEVICES={gpu}",
                    "python",
                    "dn_splatter/eval/eval_mesh_mushroom_vis_cull.py",
                    "--gt-mesh-path",
                    str(os.path.join(DATASET_PATH, SCENE)),
                    "--pred-mesh-path",
                    str(
                        os.path.join(
                            experiment_path,
                            "mesh",
                            "Open3dTSDFfusion_mesh.ply",
                        )
                    ),
                    "--output",
                    str(os.path.join(results_path)),
                    "--output-same-as-pred-mesh",
                    str(False),
                ]
            elif dataset == "scannetpp":
                eval_mesh_command = [
                    f" CUDA_VISIBLE_DEVICES={gpu}",
                    "python",
                    "dn_splatter/eval/eval_mesh_vis_cull.py",
                    "--gt-mesh-path",
                    str(
                        os.path.join(
                            DATASET_PATH, SCENE, "scans", "mesh_aligned_0.05.ply"
                        )
                    ),
                    "--pred-mesh-path",
                    str(
                        os.path.join(
                            experiment_path,
                            "mesh",
                            "Open3dTSDFfusion_mesh.ply",
                        )
                    ),
                    "--output",
                    str(os.path.join(results_path)),
                    "--output-same-as-pred-mesh",
                    str(False),
                    "--transformation-file",
                    str(os.path.join(DATASET_PATH, SCENE, "iphone", "transforms.json")),
                    "--dataset-path",
                    str(os.path.join(DATASET_PATH, SCENE, "iphone")),
                ]

        print("\t Evaluating mesh")
        print("command being ran: ", eval_mesh_command)
        if not dry_run:
            os.system((" ").join(eval_mesh_command))


def worker(gpu, scene, config):
    """This worker function starts a job and returns when it's done."""
    print(f"Starting {config.experiment_name} job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, config)
    print(f"Finished {config.experiment_name} job on GPU {gpu} with scene {scene}\n")


def dispatch_jobs(jobs, executor, config):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(
            GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1, maxLoad=0.1)
        )
        available_gpus = list(all_available_gpus - reserved_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(
                worker, gpu, job, config
            )  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)
            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(
                future
            )  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., releasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


def main():
    """Launch batch_configs in serial but process each config in parallel (multi gpu)"""

    for config in run_configs:
        # num jobs = num scenes to run for current config
        jobs = mushroom_scenes

        # Run multiple gpu train scripts
        # Using ThreadPoolExecutor to manage the thread pool
        with ThreadPoolExecutor(max_workers=8) as executor:
            dispatch_jobs(jobs, executor, config)


if __name__ == "__main__":
    main()
