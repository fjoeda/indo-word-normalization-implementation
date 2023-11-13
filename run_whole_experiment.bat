ECHO activate environment
call conda activate thesis_env

ECHO tuning indocollex
CALL python run_tuning.py --dataset_name indocollex --num_epoch 200

ECHO tuning col_id_norm
CALL python run_tuning.py --dataset_name col_id_norm --num_epoch 200

ECHO tuning combined
CALL python run_tuning.py --dataset_name combined --num_epoch 200

ECHO preform evaluation
CALL python run_evaluation_with_report.py