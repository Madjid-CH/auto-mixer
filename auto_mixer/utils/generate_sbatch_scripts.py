import os


def get_script_name(dataset_name):
    return f"run_{dataset_name}.sh"


def get_header(name):
    return ("#!/bin/bash\n"
            "trap 'echo \"Command failed with error $?. Continuing.\"' ERR\n"
            f"#SBATCH --job-name={name}\n"
            "#SBATCH --time=100:00:00\n"
            "#SBATCH --partition=gpu\n"
            "#SBATCH --ntasks=1\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --gres=gpu:a40-48:1\n"
            "#SBATCH --cpus-per-gpu=20\n"
            f"#SBATCH --output=./OutFiles/out_{name}.log\n"
            f"#SBATCH --error=./OutFiles/err_{name}.log\n"
            "#SBATCH --mail-type=ALL # (BEGIN, END, FAIL or ALL)\n"
            "#SBATCH --mail-user=a.chergui@esi-sba.dz\n"
            "\n")


def get_script_path(configs_dir, dataset_name):
    dir = f"{configs_dir.replace("/cfg", "")}"
    return f"{dir}/scripts/{get_script_name(dataset_name)}"


def generate_sbatch_scripts(configs_dir, dataset_name, project):
    runs_files = os.listdir(f'{configs_dir}/runs')
    script_path = get_script_path(configs_dir, dataset_name)
    with open(script_path, "w") as f:
        f.write(get_header(dataset_name))
        for run_file in runs_files:
            name = get_run_name(run_file)
            f.write(
                f"srun python run.py -c {configs_dir}/runs/{run_file} -n {dataset_name}_{name} -pr {project}\n"
            )


def get_run_name(run_file):
    return run_file.split(".")[0].replace("_run", "")


def get_separate_script_path(config_dir, run_file):
    dir = f"{config_dir.replace('/cfg', '')}"
    return f"{dir}/scripts/run_{get_run_name(run_file)}.sh"


def get_separate_file_header(name):
    return ("#!/bin/bash\n"
            f"#SBATCH --job-name={name}\n"
            "#SBATCH --time=100:00:00\n"
            "#SBATCH --partition=gpu\n"
            "#SBATCH --ntasks=1\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --gres=gpu:a40-48:1\n"
            "#SBATCH --cpus-per-gpu=10\n"
            f"#SBATCH --output=./OutFiles/out_{name}.log\n"
            f"#SBATCH --error=./OutFiles/err_{name}.log\n"
            "#SBATCH --mail-type=FAIL # (BEGIN, END, FAIL or ALL)\n"
            "#SBATCH --mail-user=a.chergui@esi-sba.dz\n"
            "\n")


def generate_separate_sbatch_scripts(configs_dir, dataset_name, project):
    runs_files = os.listdir(f'{configs_dir}/runs')
    for run_file in runs_files:
        script_path = get_separate_script_path(configs_dir, run_file)
        with open(script_path, "w") as f:
            f.write(get_separate_file_header(f"{dataset_name}_{get_run_name(run_file)}"))
            name = get_run_name(run_file)
            f.write(
                f"srun python run.py -c {configs_dir}/runs/{run_file} -n {dataset_name}_{name} -pr {project}\n"
            )


def get_jean_zay_header(name):
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={name}\n"
        "# Other partitions are usable by activating/uncommenting\n"
        "#SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU\n"
        "#SBATCH --nodes=1                    # number of nodes\n"
        "#SBATCH --ntasks-per-node=4          # number of MPI tasks per node (= number of GPUs per node)\n"
        "#SBATCH --gres=gpu:16                 # number of GPUs per node (max 8 with gpu_p2, gpu_p5)\n"
        "# The number of CPUs per task must be adapted according to the partition used. Knowing that here\n"
        "# only one GPU per task is reserved (i.e. 1/4 or 1/8 of the GPUs of the node depending on\n"
        "# the partition), the ideal is to reserve 1/4 or 1/8 of the CPUs of the node for each task:\n"
        "#SBATCH --cpus-per-task=8           # number of cores per task (1/4 of the node here)\n"
        "# /!\\ Caution, \"multithread\" in Slurm vocabulary refers to hyperthreading.\n"
        "#SBATCH --hint=nomultithread         # hyperthreading deactivated\n"
        "#SBATCH --time=99:00:00              # maximum execution time requested (HH:MM:SS)\n"
        "#SBATCH --qos=qos_gpu-t4\n"
        f"#SBATCH --output={name}_%j.out # name of output file\n"
        f"#SBATCH --error={name}_%j.out  # name of error file (here, in common with the output file)\n"
        "\n"
        "# Cleans out modules loaded in interactive and inherited by default\n"
        "module purge\n"
        "\n"
        "module load pytorch-gpu/py3/2.2.0\n"
        "\n"
        "# Echo of launched commands\n"
        "set -x\n"
    )


def get_jean_zay_script_path(config_dir, run_file):
    dir = f"{config_dir.replace('/cfg', '')}"
    return f"{dir}/scripts/jean_zay/run_{get_run_name(run_file)}.sh"


def generate_separate_sbatch_scripts_for_jean_zay(configs_dir, dataset_name, project):
    runs_files = os.listdir(f'{configs_dir}/runs')
    for run_file in runs_files:
        script_path = get_jean_zay_script_path(configs_dir, run_file)
        with open(script_path, "w") as f:
            f.write(get_jean_zay_header(f"{dataset_name}_{get_run_name(run_file)}"))
            name = get_run_name(run_file)
            f.write(
                f"srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u run.py -c {configs_dir}/runs/{run_file} -n {dataset_name}_{name} -pr {project}\n"
            )


if __name__ == '__main__':
    # generate_sbatch_scripts("m2_mixer/usecases/chexpert/cfg", "chexpert", "chexpert")
    # generate_separate_sbatch_scripts("m2_mixer/usecases/sst/cfg", "sst", "sst")
    generate_separate_sbatch_scripts_for_jean_zay("m2_mixer/usecases/imagenet/cfg", "imagenet", "image-modality")
