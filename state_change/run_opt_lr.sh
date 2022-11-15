#! /bin/bash                                                                                  
#SBATCH --gres=gpu:1
#SBATCH -t 1440               # Runtime in minutes
#SBATCH --mem=8000          # Memory per node in MB
#SBATCH -p seas_dgx1,seas_gpu,gpu,gpu_mig               # Partition to submit to                                            
#SBATCH --job-name=opt_sc_lr # Note that %A and %a are the place holders for job id and array id  
#SBATCH --output=../Simulation_Results/out/opt_sc_lr_%a.out                                                           
#SBATCH --error=../Simulation_Results/err/opt_sc_lr_%a.err
#SBATCH --array=0-''         # Run sim_sq.py with ClusterIter set to 0, 1, 2, 3, 4                                                            
                                                                        
export TASK_ID=$SLURM_ARRAY_TASK_ID
export MIN_JOB_ID=$SLURM_ARRAY_TASK_MIN
export MAX_JOB_ID=$SLURM_ARRAY_TASK_MAX
export JOB_ID=$SLURM_JOB_ID

echo "Running opt_state_change_tuneLR.py with TaskID = " $MIN_JOB_ID "..." $MAX_JOB_ID
echo "Task ID:" $TASK_ID
echo "Job ID:" $JOB_ID

module load cuda/11.7.1-fasrc01
python opt_state_change_tuneLR.py $TASK_ID
