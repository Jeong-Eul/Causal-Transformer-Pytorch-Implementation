set PYTHONPATH=.
python run.py -m +dataset=cancer_sim +backbone=ct +backbone/ct_hparams/cancer_sim_domain_conf=\"0\" exp.seed=10 exp.logging=True

