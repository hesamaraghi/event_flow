#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general
#SBATCH --output=/tudelft.net/staff-bulk/ewi/insy/VisionLab/maraghi/event-based-optical-flow/event_flow/sbatch_output/jupyter/jupyter_%j.out
#SBATCH --open-mode=truncate

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=medium

# The default run (wall-clock) time is 1 minute
#SBATCH --time=36:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# The default number of CPUs per task is 1 (note: CPUs are always allocated to jobs per 2)
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
#SBATCH --cpus-per-task=2

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=8000

# Request a GPU
#SBATCH --gres=gpu:1

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END


# module use /opt/insy/modulefiles
# module load cuda/11.2 cudnn/11.2-8.1.1.33
cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
#mlflow ui --host=0.0.0.0 --port=8888
#jupyter notebook --ip=0.0.0.0 --port=8888
