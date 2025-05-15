# AceCoder DAPO implementation Training Guide

just testing if the training recipe works.

building dataset: 

python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 11 --add_public_tests_all True

running the training:

bash examples/train/dapo_testing/train_1.5b_mtrl.sh


