import os


def generate_run_configs(config_dir):
    train_files = sorted(get_train_configs(config_dir))
    dataset_files = sorted(get_dataset_configs(config_dir))
    model_files = get_models_configs(config_dir)
    for train_file, dataset_file in zip(train_files, dataset_files):
        for model_file in model_files:
            config = get_config_str(config_dir, dataset_file, model_file, train_file)
            with open(f"{config_dir}/runs/{get_run_config_file_name(model_file, dataset_file)}", "w") as f:
                f.write(config)


def get_config_str(config_dir, dataset_file, model_file, train_file):
    config = f"train: !include {config_dir}/train/{train_file}\n"
    config += f"dataset: !include {config_dir}/dataset/{dataset_file}\n"
    config += f"model: !include {config_dir}/models/{model_file}\n"
    return config


def get_models_configs(config_dir):
    return os.listdir(f'{config_dir}/models')


def get_dataset_configs(config_dir):
    return os.listdir(f'{config_dir}/dataset')


def get_train_configs(config_dir):
    return os.listdir(f'{config_dir}/train')


def get_run_config_file_name(model_file, dataset_file):
    proportion = dataset_file.split("_")[1].split(".")[0]
    suffix = f"_{proportion}_run.yml"
    return model_file.replace(".yml", suffix)


if __name__ == '__main__':
    config_dir = "m2_mixer/usecases/sst/cfg"
    generate_run_configs(config_dir)
