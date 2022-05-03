# python3 main.py --task atis --model_type roberta --model_dir atis_model_roberta --do_train --do_eval --num_train_epochs 18
# python3 main.py --task snips --model_type roberta --model_dir snips_model_roberta --do_train --do_eval --num_train_epochs 20

# python3 main.py --task atis --model_type roberta --model_dir atis_model_roberta --do_train --do_eval --num_train_epochs 5
# python3 main.py --task snips --model_type roberta --model_dir snips_model_roberta --do_train --do_eval --num_train_epochs 29
# python3 main.py --task snips --model_type simcse-roberta --model_dir snips_model_simcse-roberta --do_train --do_eval --num_train_epochs 29
# python3 main.py --task snips --model_type simcse-fewer-wo-roberta --model_dir snips_model_simcse-fewer-wo-roberta --do_train --do_eval --num_train_epochs 29
# python3 main.py --task snips --model_type wordnet-roberta --model_dir snips_model_SimCSE_wordnet-roberta --do_train --do_eval --num_train_epochs 29
# python3 main.py --task snips --model_type simcse-wordnet-roberta --model_dir snips_model_simcse-wordnet-roberta --do_train --do_eval --num_train_epochs 29


# python3 main.py --task snips --model_dir my-sup-simcse-bert-base-uncased --do_eval

python3 main.py --task atis --model_type bert --model_dir atis_model_crf --do_train --do_eval --num_train_epochs 10 --use_crf
python3 main.py --task snips --model_type bert --model_dir snips_model_crf --do_train --do_eval --num_train_epochs 10 --use_crf