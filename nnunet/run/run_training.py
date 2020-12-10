import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network")    ### '2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'
    parser.add_argument("network_trainer")    #### nnUNetTrainer
    parser.add_argument("task")         ####  Task
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'') ####
    parser.add_argument("--ndet", required=False, default=False, action="store_true",
                        help="Per default training is deterministic, "
                             "nondeterministic allows cudnn.benchmark which will can give up to 20%% performance. "
                             "Set this to do nondeterministic training")  ### ....????? what this means .....

    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation", action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training", action="store_true")
    parser.add_argument("-p", help="plans identifier", default=default_plans_identifier, required=False)
    parser.add_argument("-u", "--unpack_data", required=False, default=1, type=int, help="Leave it as 1, development only")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the vlaidation as well. " ### !!!!
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")

    parser.add_argument("--find_lr", required=False, default=False, action="store_true", help="not used here, just for fun") ### ......
    parser.add_argument("--valbest", required=False, default=False, action="store_true", help="hands off. This is not intended to be used")
    parser.add_argument("--fp16", required=False, default=False, action="store_true", help="enable fp16 training. Makes sense for 2d only! (and only on supported hardware!)")
    parser.add_argument("--gpu", default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    network = args.network
    network_trainer = args.network_trainer
    task = args.task
    fold = args.fold
    deterministic = not args.ndet

    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    unpack = args.unpack_data
    valbest = args.valbest
    fp16 = args.fp16

    if unpack == 0:
        unpack = False
    elif unpack == 1:
        unpack = True
    else:
        raise ValueError("Unexpected value for -u/--unpack_data: %s. Use 1 or 0." % str(unpack))

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class = \
        get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, nnUNetTrainerCascadeFullRes), \
            "If running 3d_cascade_fullres then your trainer class must be derived from nnunetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class, nnUNetTrainer), "network_trainer was found but is not derived from nnunetTrainer" ### ....argument 'network_trainer' ...

    trainer = trainer_class(plans_file, fold,
                            output_folder=output_folder_name,
                            dataset_directory=dataset_directory,
                            batch_dice=batch_dice,
                            stage=stage,
                            unpack_data=unpack,
                            deterministic=deterministic,
                            fp16=fp16,
                            network_dims=network)
    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                trainer.load_latest_checkpoint()
            trainer.run_training()

        ### validation
        elif not valbest:
            trainer.load_latest_checkpoint(train=False)

        if valbest:
            trainer.load_best_checkpoint(train=False)
            val_folder = "validation_best_epoch"
        else:
            val_folder = "validation"

        ### we set do_mirroring False because of asymmetry of anatomy.
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder, do_mirroring=False, use_gaussian=True)

        if network == '3d_lowres':
            trainer.load_best_checkpoint(False)
            print("predicting segmentations for the next stage of the cascade")
            print()
            predict_next_stage(trainer,
                               join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1),
                               fold)
