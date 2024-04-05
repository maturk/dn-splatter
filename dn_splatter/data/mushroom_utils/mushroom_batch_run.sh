#!/bin/bash

# Runs kinect and iphone datasets for all loss types back-to-back
# NOTE: make sure you 'chmod +x' to be able to execute it

# Configs: Mushroom dataset
dataset_name="activity" 
root_path_to_mushroom="" # dont add a trailing '/' here
depth_lambda=0.1
iters_to_save=25000

# kinect commands
python_command_no_depth="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save} --pipeline.model.save_eval_name ${dataset_name}-kinect-no-depth --pipeline.model.use-depth-loss False --data ${root_path_to_mushroom}/${dataset_name}/kinect/long_capture mushroom --mode kinect"
python_command_mse="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save} --pipeline.model.save_eval_name ${dataset_name}-kinect-depth-mse --pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type MSE --pipeline.model.depth-lambda ${depth_lambda} --data ${root_path_to_mushroom}/${dataset_name}/kinect/long_capture mushroom --mode kinect"
python_command_logl1="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save} --pipeline.model.save_eval_name ${dataset_name}-kinect-depth-logl1 --pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type LogL1 --pipeline.model.depth-lambda ${depth_lambda} --data ${root_path_to_mushroom}/${dataset_name}/kinect/long_capture mushroom --mode kinect"
python_command_huberl1="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save}  --pipeline.model.save_eval_name ${dataset_name}-kinect-depth-huberl1 --pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type HuberL1 --pipeline.model.depth-lambda ${depth_lambda} --data ${root_path_to_mushroom}/${dataset_name}/kinect/long_capture mushroom --mode kinect"
python_command_dssiml1="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save} --pipeline.model.save_eval_name ${dataset_name}-kinect-depth-dssiml1 --pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type DSSIML1 --pipeline.model.depth-lambda ${depth_lambda} --data ${root_path_to_mushroom}/${dataset_name}/kinect/long_capture mushroom --mode kinect"
python_command_l1="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save} --pipeline.model.save_eval_name ${dataset_name}-kinect-depth-l1 --pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type L1 --pipeline.model.depth-lambda ${depth_lambda} --data ${root_path_to_mushroom}/${dataset_name}/kinect/long_capture mushroom --mode kinect"


# execute kinect commands
echo "Evaluating on the kinect no-depth sequence..."
$python_command_no_depth
echo "Evaluating on the kinect depth-mse sequence..."
$python_command_mse
echo "Evaluating on the kinect depth-logl1 sequence..."
$python_command_logl1
echo "Evaluating on the kinect depth-huberl1 sequence..."
$python_command_huberl1
echo "Evaluating on the kinect depth-dssiml1 sequence..."
$python_command_dssiml1
echo "Evaluating on the kinect depth-l1 sequence..."
$python_command_l1

# iphone commands
python_command_no_depth="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save} --pipeline.model.save_eval_name ${dataset_name}-iphone-no-depth --pipeline.model.use-depth-loss False --data ${root_path_to_mushroom}/${dataset_name}/iphone/long_capture mushroom --mode iphone"
python_command_mse="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save} --pipeline.model.save_eval_name ${dataset_name}-iphone-depth-mse --pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type MSE --pipeline.model.depth-lambda ${depth_lambda} --data ${root_path_to_mushroom}/${dataset_name}/iphone/long_capture mushroom --mode iphone"
python_command_logl1="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save} --pipeline.model.save_eval_name ${dataset_name}-iphone-depth-logl1 --pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type LogL1 --pipeline.model.depth-lambda ${depth_lambda} --data ${root_path_to_mushroom}/${dataset_name}/iphone/long_capture mushroom --mode iphone"
python_command_huberl1="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save} --pipeline.model.save_eval_name ${dataset_name}-iphone-depth-huberl1 --pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type HuberL1 --pipeline.model.depth-lambda ${depth_lambda} --data ${root_path_to_mushroom}/${dataset_name}/iphone/long_capture mushroom --mode iphone"
python_command_dssiml1="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save} --pipeline.model.save_eval_name ${dataset_name}-iphone-depth-dssiml1 --pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type DSSIML1 --pipeline.model.depth-lambda ${depth_lambda} --data ${root_path_to_mushroom}/${dataset_name}/iphone/long_capture mushroom --mode iphone"
python_command_l1="ns-train dn_splatter --viewer.quit-on-train-completion True --pipeline.model.eval_all_images_at_train_iter True --pipeline.model.iters_to_save ${iters_to_save} --pipeline.model.save_eval_name ${dataset_name}-iphone-depth-l1 --pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type L1 --pipeline.model.depth-lambda ${depth_lambda} --data ${root_path_to_mushroom}/${dataset_name}/iphone/long_capture mushroom --mode iphone"

# execute iphone commands
echo "Evaluating on the iphone no-depth sequence..."
$python_command_no_depth
echo "Evaluating on the depth-mse iphone..."
$python_command_mse
echo "Evaluating on the depth-logl1 iphone..."
$python_command_logl1
echo "Evaluating on the depth-huberl1 iphone..."
$python_command_huberl1
echo "Evaluating on the depth-dssiml1 iphone..."
$python_command_dssiml1
echo "Evaluating on the depth-l1 iphone..."
$python_command_l1
