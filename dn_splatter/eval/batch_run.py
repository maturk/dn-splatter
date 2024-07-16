"""Script for running dataset scenes in serial under different configurations."""
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from dn_splatter.scripts.render_model import RenderModel


@dataclass
class Config:
    # Output settings
    output_dir: str = "experiments/"
    experiment_name: str = "splatfacto"
    timestamp: str = ""  # do not change
    # Paths
    mushroom_dataset_path: str = "datasets/room_datasets"
    replica_dataset_path: str = ""
    scannetpp_dataset_path: str = ""

    iterations: int = 30000
    # Depth configs
    use_depth_loss: bool = False
    sensor_depth_lambda: float = 0.2
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


# Default configuration
splatfacto_config = Config(experiment_name="splatfacto")
dn_splatter_config = Config(
    experiment_name="dn_splatter",
    use_depth_loss=True,
    use_depth_smooth_loss=True,
    use_normal_loss=True,
    load_pcd_normals=True,
    use_normal_tv_loss=True,
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

scannetpp_scenes = ["8b5caf3398", "b20a261fdf"]

# what different runs to do
run_configs = [splatfacto_config, dn_splatter_config]


def main(
    dataset: Literal["mushroom", "replica", "scannetpp"] = "mushroom",
    dry_run: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
    skip_mesh: bool = False,
    skip_render: bool = False,
    skip_collect: bool = False,
):
    for config in run_configs:
        config_time_start = time.time()
        if dataset == "mushroom":
            scenes = mushroom_scenes
        else:
            scenes = scannetpp_scenes

        for SCENE in scenes:
            # current dataset
            if dataset == "mushroom":
                DATASET_PATH = config.mushroom_dataset_path
            elif dataset == "replica":
                DATASET_PATH = config.replica_dataset_path
            elif dataset == "scannetpp":
                DATASET_PATH = config.scannetpp_dataset_path
            else:
                raise NotImplementedError

            # experiment paths and save directories
            experiment_path = os.path.join(
                config.output_dir, config.experiment_name, "dn-splatter", SCENE
            )
            config_path = os.path.join(experiment_path, "config.yml")
            results_path = os.path.join(experiment_path, "results")
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            print(f"Running {SCENE}")
            if not skip_train:
                command = [
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
                    "--pipeline.model.sensor-depth-lambda",
                    str(config.sensor_depth_lambda),
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

                if not dry_run:
                    start_time = time.time()
                    subprocess.run(command)
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

            # save eval metrics
            if not skip_eval:
                eval_command = [
                    "ns-eval",
                    "--load-config",
                    str(config_path),
                    "--output-path",
                    str(os.path.join(results_path, "metrics.json")),
                ]
                print("\t Running rgb/depth eval")
                print("command being ran: ", eval_command)
                if not dry_run:
                    subprocess.run(eval_command)

                if not skip_mesh:
                    mesh_command = [
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
                        subprocess.run(mesh_command)

                    if dataset == "mushroom":
                        eval_mesh_command = [
                            "python",
                            "dn_splatter/eval/eval_mesh_mushroom_vis_cull.py",
                            "--gt-mesh-path",
                            str(os.path.join(config.mushroom_dataset_path, SCENE)),
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
                    else:
                        raise NotImplementedError

                    print("\t Evaluating mesh")
                    print("command being ran: ", eval_mesh_command)
                    if not dry_run:
                        subprocess.run(eval_mesh_command)

        # collect results from files and average them
        if not skip_collect:
            average_metrics = {}
            average_mesh_metrics = {}
            average_train_time_metrics = {}
            for SCENE in scenes:
                # current dataset
                if dataset == "mushroom":
                    DATASET_PATH = config.mushroom_dataset_path
                elif dataset == "replica":
                    DATASET_PATH = config.replica_dataset_path
                elif dataset == "scannetpp":
                    DATASET_PATH = config.scannetpp_dataset_path
                else:
                    raise NotImplementedError

                # experiment paths and save directories
                experiment_path = os.path.join(
                    config.output_dir, config.experiment_name, "dn-splatter", SCENE
                )
                config_path = os.path.join(experiment_path, "config.yml")
                results_path = os.path.join(experiment_path, "results")

                if not skip_mesh:
                    mesh_metrics = json.load(
                        open(os.path.join(results_path, "mesh_metrics.json"))
                    )
                metrics = json.load(open(os.path.join(results_path, "metrics.json")))
                metrics = metrics["results"]
                train_time_metrics = json.load(
                    open(os.path.join(results_path, "train_time.json"))
                )

                if not skip_mesh:
                    for key, value in mesh_metrics.items():
                        if average_mesh_metrics.get(key, []) == []:
                            average_mesh_metrics[key] = [value]
                        else:
                            temp = average_mesh_metrics[key]
                            temp.append(value)
                            average_mesh_metrics[key] = temp

                for key, value in metrics.items():
                    if average_metrics.get(key, []) == []:
                        average_metrics[key] = [value]
                    else:
                        temp = average_metrics[key]
                        temp.append(value)
                        average_metrics[key] = temp

                for key, value in train_time_metrics.items():
                    if average_train_time_metrics.get(key, []) == []:
                        average_train_time_metrics[key] = [value]
                    else:
                        temp = average_train_time_metrics[key]
                        temp.append(value)
                        average_train_time_metrics[key] = temp

            if not skip_mesh:
                for key in average_mesh_metrics.keys():
                    temp = average_mesh_metrics[key]
                    temp = sum(temp) / len(temp)
                    average_mesh_metrics[key] = temp

            for key in average_metrics.keys():
                temp = average_metrics[key]
                temp = sum(temp) / len(temp)
                average_metrics[key] = temp

            for key in average_train_time_metrics.keys():
                temp = average_train_time_metrics[key]
                temp = sum(temp) / len(temp)
                average_train_time_metrics[key] = temp

            average_metrics_save_path = Path(experiment_path).parent.parent
            if not skip_mesh:
                with open(
                    average_metrics_save_path / Path("average_mesh_metrics.json"), "w"
                ) as outfile:
                    json.dump(average_mesh_metrics, outfile)
            with open(
                average_metrics_save_path / Path("average_metrics.json"), "w"
            ) as outfile:
                json.dump(average_metrics, outfile)
            with open(
                average_metrics_save_path / Path("average_train_time.json"), "w"
            ) as outfile:
                json.dump(average_train_time_metrics, outfile)
            print("!!!Finished!!!")

            config_time_end = time.time()
            with open(
                average_metrics_save_path / Path("config_run_time.json"), "w"
            ) as outfile:
                print(f"TOTAL TIME FOR RUN: {config_time_end - config_time_start}")
                try:
                    json.dump(
                        {"total_config_run_time": config_time_end - config_time_start},
                        outfile,
                    )
                except:  # noqa: E722
                    print(f"TOTAL TIME FOR RUN: {config_time_end - config_time_start}")
            print("!!!Finished!!!")


if __name__ == "__main__":
    tyro.cli(main)
