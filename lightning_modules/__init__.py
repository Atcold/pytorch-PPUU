import argparse

import lightning_modules.mpur
import lightning_modules.mpur_dreaming
import lightning_modules.mpur_km

MODULES_DICT = dict(
    vanilla=mpur.MPURModule,
    dreaming=mpur_dreaming.MPURDreamingModule,
    km=mpur_km.MPURKMModule,
    km_split=mpur_km.MPURKMSplitModule,
    continuous=mpur.MPURContinuousModule,
)


def get_module_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        help="Pick the model type to run",
        required=True,
    )
    args, _ = parser.parse_known_args()
    return get_module(args.model_type)


def get_module(name):
    return MODULES_DICT[name]
