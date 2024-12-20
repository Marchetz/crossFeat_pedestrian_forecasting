#the results may differ slightly from the results reported in the paper.

#jaad go
python eval.py --n_trials 10 --jaad --mode GO --model_pretrained_path pretrained_models/jaad_go

#jaad stop
python eval.py --n_trials 10 --jaad --mode STOP --model_pretrained_path pretrained_models/jaad_stop

#pie go
python eval.py --n_trials 10 --pie --mode GO ---model_pretrained_path pretrained_models/pie_go

#pie stop
python eval.py --n_trials 10 --pie --mode STOP --model_pretrained_path pretrained_models/pie_stop

#titan go
python eval.py --n_trials 10 --titan --mode GO --model_pretrained_path pretrained_models/titan_go

#titan stop
python eval.py --n_trials 10 --titan --mode STOP --model_pretrained_path pretrained_models/titan_stop
