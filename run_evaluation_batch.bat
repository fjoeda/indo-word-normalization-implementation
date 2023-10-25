TITLE Running Seq2Seq Evaluation

ECHO dataset col_id_norm drop 0.1
CALL python run_evaluation.py --dataset_path .\dataset\col_id_norm_test.json --model_path .\models\word-norm-col-id-norm-drop0.1_epoch_13.pth --config_name smaller-transformer-drop-0.1

ECHO dataset col_id_norm drop 0.1
CALL python run_evaluation.py --dataset_path .\dataset\col_id_norm_test.json --model_path .\models\word-norm-col-id-norm-drop0.3_epoch_13.pth --config_name smaller-transformer-drop-0.3

ECHO dataset indocollex drop 0.1
CALL python run_evaluation.py --dataset_path .\dataset\indo_collex_test.json --model_path .\models\word-norm-indocollex-drop0.1_epoch_43.pth --config_name smaller-transformer-drop-0.1

ECHO dataset indocollex drop 0.3
CALL python run_evaluation.py --dataset_path .\dataset\indo_collex_test.json --model_path .\models\word-norm-indocollex-drop0.3_epoch_61.pth --config_name smaller-transformer-drop-0.3

ECHO dataset combined drop 0.1
CALL python run_evaluation.py --dataset_path .\dataset\combined_test.json --model_path .\models\word-norm-combined-norm-drop0.1_epoch_12.pth --config_name smaller-transformer-drop-0.1

ECHO dataset combined drop 0.3
CALL python run_evaluation.py --dataset_path .\dataset\combined_test.json --model_path .\models\word-norm-combined-norm-drop0.3_epoch_14.pth --config_name smaller-transformer-drop-0.3
