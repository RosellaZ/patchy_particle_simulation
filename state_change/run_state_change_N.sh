#! /bin/bash                                                                                  
#SBATCH --gres=gpu:1
#SBATCH -t 1440               # Runtime in minutes
#SBATCH --mem=8000          # Memory per node in MB
#SBATCH -p seas_dgx1,seas_gpu,gpu,gpu_mig           # Partition to submit to                                            
#SBATCH --job-name=state_change_N20 # Note that %A and %a are the place holders for job id and array id  
#SBATCH --output=../Simulation_Results/out/state_change_N20.out                                                           
#SBATCH --error=../Simulation_Results/err/state_change_N20.err                                                            
                                                                        
export JOB_ID=$SLURM_JOB_ID

echo "Running state_change_N.py"
echo "Job ID:" $JOB_ID

module load cuda/11.7.1-fasrc01
python state_change_N.py 20
