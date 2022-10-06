#! /bin/bash                                                                                  
#SBATCH --gres=gpu:1
#SBATCH -t 30               # Runtime in minutes
#SBATCH --mem=1000          # Memory per node in MB
#SBATCH -p gpu_test               # Partition to submit to                                            
#SBATCH --job-name=run_sweepN_sq # Note that %A and %a are the place holders for job id and array id  
#SBATCH --output=../Simulation_Results/out/sweepN_sq_%a.out                                                           
#SBATCH --error=../Simulation_Results/err/sweepN_sq_%a.err                                                            
#SBATCH --array=0-3          # Run sim_sq.py with ClusterIter set to 0, 1, 2, 3
                                                                        
export TASK_ID=$SLURM_ARRAY_TASK_ID
export MIN_JOB_ID=$SLURM_ARRAY_TASK_MIN
export MAX_JOB_ID=$SLURM_ARRAY_TASK_MAX
export JOB_ID=$SLURM_JOB_ID

echo "Running sweepN_sq.py with TaskID = " $MIN_JOB_ID "..." $MAX_JOB_ID
echo "Task ID:" $TASK_ID
echo "Job ID:" $JOB_ID

module load cuda/11.7.1-fasrc01
python sweepN_sq.py $TASK_ID
