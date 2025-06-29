# GPT-2 124M with different MoE configs
# sbatch --job-name="onlymoe" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-onlymoe-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-onlymoe-%j.err" run_single_gpu.sh 16 42
# sbatch --job-name="noisyrouter" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-noisyrouter-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-noisyrouter-%j.err" run_single_gpu_noisy_router.sh 16 42
# sbatch --job-name="auxloss" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-auxloss-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-auxloss-%j.err" run_single_gpu_aux_loss.sh 16 42
# sbatch --job-name="zloss" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-zloss-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-zloss-%j.err" run_single_gpu_z_loss.sh 16 42
# sbatch --job-name="fullprec" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-fullprec-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-fullprec-%j.err" run_single_gpu_router_full_prec.sh 16 42
# sbatch --job-name="swinit" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-swinit-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-swinit-%j.err" run_single_gpu_use_switch_init.sh 16 42
# sbatch --job-name="nomoe42" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-nomoe-42-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-nomoe-42-%j.err" run_single_gpu_no_moe.sh 64 42
# sbatch --job-name="nomoe" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-nomoe-2025-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-nomoe-2025-%j.err" run_single_gpu_no_moe.sh 64 2025

# sbatch --job-name="onlymoe" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-onlymoe-2025-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-onlymoe-2025-%j.err" run_single_gpu.sh 16 2025
# sbatch --job-name="noisyrouter" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-noisyrouter-2025-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-noisyrouter-2025-%j.err" run_single_gpu_noisy_router.sh 16 2025
# sbatch --job-name="auxloss" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-auxloss-2025-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-auxloss-2025-%j.err" run_single_gpu_aux_loss.sh 16 2025
# sbatch --job-name="zloss" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-zloss-2025-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-zloss-2025-%j.err" run_single_gpu_z_loss.sh 16 2025
# sbatch --job-name="fullprec" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-fullprec-2025-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-fullprec-2025-%j.err" run_single_gpu_router_full_prec.sh 16 2025
# sbatch --job-name="swinit" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-swinit-2025-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-swinit-2025-%j.err" run_single_gpu_use_switch_init.sh 16 2025

# sbatch --job-name="onlymoek1" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-onlymoek1-2025-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-onlymoek1-2025-%j.err" run_single_gpu_moe_topk_1.sh 16 2025
sbatch --job-name="onlymoek1n" --output="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-onlymoek1-42-%j.out" --error="../slurmoutputs/$(date +%Y%m%d_%H%M%S)-onlymoek1-42-%j.err" run_single_gpu_moe_topk_1.sh 16 42