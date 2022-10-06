#! /bin/bash                                                                                  
#SBATCH --gres=gpu:1
#SBATCH -t 1000               # Runtime in minutes
#SBATCH --mem=8000          # Memory per node in MB
#SBATCH -p gpu_test               # Partition to submit to                                            
#SBATCH --job-name=opt_sq # Note that %A and %a are the place holders for job id and array id  
#SBATCH --output=../Simulation_Results/out/opt_sq.out                                                           
#SBATCH --error=../Simulation_Results/err/opt_sq.err                                                            
                                                                        
export JOB_ID=$SLURM_JOB_ID

echo "Running optimization_angle.py"
echo "Job ID:" $JOB_ID

module load cuda/11.7.1-fasrc01
python optimization_angle.py
