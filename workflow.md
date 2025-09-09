# Workflow

## 0. Conda env
Make conda env called `prodigy36` with python3.6, install requirements with `pip install -r py36_deployment_reqs.txt`.

## 1. Format dataset
Run `format_for_prodigy.py` in `data/social_media_single_trace_dataset/original/`.  
It creates `for_prodigy/` with:
- `prod_train_data.hdf`, `prod_train_label.csv` (all healthy)  
- `prod_test_data.hdf`, `prod_test_label.csv` (normal + abnormal)  
It also cleans up the data.

## 2. Update paths as needed
In `reproducibility_experiments.py` and `reproducibility_plots.py`, paths may need to be changed as needed. 


## 3. Run on SCC
Submit the provided `run.qsub` file.
Results + plots will be saved in `social_media_output/`

Alternatively, change this to a bash script and run on local machine.