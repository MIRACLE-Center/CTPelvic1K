import nnunet
from nnunet.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.summarize_plans import summarize_plans
from nnunet.training.model_restore import recursive_find_trainer


def get_configuration_from_output_folder(folder):
    folder = folder[len(network_training_output_dir):]
    if folder.startswith("/"):
        folder = folder[1:]

    configuration, task, trainer_and_plans_identifier = folder.split("/")
    trainer, plans_identifier = trainer_and_plans_identifier.split("__")
    return configuration, task, trainer, plans_identifier


def get_output_folder(configuration, task, trainer, plans_identifier):
    return join(network_training_output_dir, configuration, task, trainer + "__" + plans_identifier)


def get_default_configuration(network, task, network_trainer, plans_identifier=default_plans_identifier,
                              search_in=(nnunet.__path__[0],
                                         "training",
                                         "network_training"),
                              base_module='nnunet.training.network_training'):

    assert network in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], \
        "network can only be one of the following: \'3d\', \'3d_lowres\', \'3d_fullres\', \'3d_cascade_fullres\'"

    dataset_directory = join(preprocessing_output_dir, task)

    if network == '2d':
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl")
    else:
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl")
    plans = load_pickle(plans_file)
    possible_stages = list(plans['plans_per_stage'].keys())
    if (network == '3d_cascade_fullres' or network == "3d_lowres") and len(possible_stages) == 1:### !!!!!
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
                           "not require the cascade. Run 3d_fullres instead")

    if network == '2d' or network == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    trainer_class = recursive_find_trainer([join(*search_in)], network_trainer, current_module=base_module)

    print('\n~·~~·~~·~~·~~·~~·~~·~~·~ get default configuration ~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~\n')

    output_folder_name = join(network_training_output_dir, network, task, network_trainer + "__" + plans_identifier)

    print("Following configuration: ")
    summarize_plans(plans_file)

    if (network == '2d' or len(possible_stages) > 1) and not network == '3d_lowres':
        batch_dice = True
        print("Loss: BATCH dice + other losses")
    else:
        batch_dice = False
        print("Loss: Simple dice + other losses")

    print('network training output dir: ', network_training_output_dir)
    print('network: ', network)
    print('task: ', task)
    print("stage: ", stage)
    print("class of my trainer is: ", trainer_class)
    print("Training/validation data folder: ", join(dataset_directory, plans['data_identifier']+f'_stage{stage}'))
    print('~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~~·~\n')
    return plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class
