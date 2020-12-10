import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from nnunet.paths import my_output_identifier

########################################################################################################################
#                                              Experiment  preparing                                                   #
########################################################################################################################
home_dir = os.environ['HOME']
train_dir = os.path.join(home_dir,'all_data/nnUNet/rawdata/Task11_CTPelvic1K')
output_dir = os.path.join(home_dir, 'all_data/nnUNet/nnUNet_raw/Task11_CTPelvic1K')
# command = f'python dataset_conversion/JstPelvisSegmentation_5label.py --train_dir {train_dir} --output_dir {output_dir}'
# command = 'python experiment_planning/plan_and_preprocess_task.py -t Task11_CTPelvic1K -pl 20 -pf 20'

processed_path = os.path.join(home_dir, 'all_data/nnUNet/nnUNet_processed/Task11_CTPelvic1K')
check_save_path = os.path.join(home_dir, 'all_data/nnUNet/nnUNet_processed/Task11_CTPelvic1K/Task11_check')
# command = f'python preprocessing/lumbosacral_joint_sampling.py --processed_path {processed_path} --check_save_path {check_save_path}'



########################################################################################################################
#                                              Experiment  running                                                     #
########################################################################################################################
# TASK = 'Task11_CTPelvic1K'
TASK = 'Task22_ipcai2021'
FOLD = 0
GPU = 0
"""
    Training
"""
# command = f'python run/run_training.py 2d nnUNetTrainer {TASK} {FOLD} --gpu {GPU}' # TASK fold gpu_idx
# command = f'python run/run_training.py 3d_fullres nnUNetTrainer {TASK} {FOLD} --gpu {GPU}'
# command = f'python run/run_training.py 3d_lowres nnUNetTrainer {TASK} {FOLD} --gpu {GPU}'
# command = f'python run/run_training.py 3d_cascade_fullres nnUNetTrainerCascadeFullRes {TASK} {FOLD} --gpu {GPU}'


"""
    Validation:  add " --validation_only" "--valbest"
        pay attention: do_mirroring has been changed to "False"
"""
# command = f'python run/run_training.py 2d nnUNetTrainer {TASK} {FOLD} --gpu {GPU} --validation_only --valbest'
# command = f'python run/run_training.py 3d_fullres nnUNetTrainer {TASK} {FOLD} --gpu {GPU} --validation_only --valbest'
# command = f'python run/run_training.py 3d_lowres nnUNetTrainer {TASK} {FOLD} --gpu {GPU} --validation_only --valbest'
# command = f'python run/run_training.py 3d_cascade_fullres nnUNetTrainerCascadeFullRes {TASK} {FOLD} --gpu {GPU} --validation_only --valbest'

"""
    Testing on data never seen
"""
test_data_path = os.path.join(home_dir, 'all_data/nnUNet/rawdata/ipcai2021_ALL_Test')

# command = f'python inference/predict_simple.py ' \
#          f'-i {test_data_path} ' \
#          f'-o {test_data_path}/{TASK}__{my_output_identifier}__fold{FOLD}_2d_pred ' \
#          f'-t {TASK} ' \
#          f'-tr nnUNetTrainer ' \
#          f'-m 2d ' \
#          f'-f {FOLD} ' \
#          f'--num_threads_preprocessing 12 '\
#          f'--num_threads_nifti_save 6 '\
#          f'--gpu {GPU}'
#
# command = f'python inference/predict_simple.py ' \
#           f'-i {test_data_path} ' \
#           f'-o {test_data_path}/{TASK}__{my_output_identifier}__fold{FOLD}_3dfullres_pred ' \
#           f'-t {TASK} ' \
#           f'-tr nnUNetTrainer ' \
#           f'-m 3d_fullres ' \
#           f'-f {FOLD} ' \
#           f'--gpu {GPU}'
#
#
# command = f'python inference/predict_simple.py ' \
#           f'-i {test_data_path} ' \
#           f'-o {test_data_path}/{TASK}__{my_output_identifier}__fold{FOLD}_3dlowres_pred ' \
#           f'-t {TASK} ' \
#           f'-tr nnUNetTrainer ' \
#           f'-m 3d_lowres ' \
#           f'-f {FOLD} ' \
#           f'--gpu {GPU} ' \
#           f'--overwrite_existing 0'
#
# my_task_lowres = TASK
# my_output_identifier_lowres = 'CTPelvic1K' #your low_res experiment\'s "my_output_identifier" in path
# command = f'python inference/predict_simple.py ' \
#           f'-i {test_data_path} ' \
#           f'-o {test_data_path}/{TASK}__{my_output_identifier_lowres}__{my_output_identifier}__fold{FOLD}_3dcascadefullres_pred ' \
#           f'-t {TASK} ' \
#           f'-tr nnUNetTrainerCascadeFullRes ' \
#           f'-m 3d_cascade_fullres ' \
#           f'-f {FOLD} ' \
#           f'-l {test_data_path}/{my_task_lowres}__{my_output_identifier_lowres}__fold{FOLD}_3dlowres_pred ' \
#           f'--gpu {GPU} ' \
#           f'--overwrite_existing 0'

########################################################################################################################
#                                              evaluation                                                              #
########################################################################################################################
# command = 'python ../evaluation.py'
# command = 'python ../save_evaluation_results2csv.py'
# command = 'python ../save_evaluation_results2csv_Manu.py'

if __name__ == '__main__':
    print('\n'*2)
    print(command,'\n'*2)
    os.system(command)
