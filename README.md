## Tutorial for property guidance pipeline

### Step 1: Prepare the dataset

Prepare the raw data as

```
|-- data
    |-- properties
	    |-- <property>
            |-- cif
            |-- <raw_data>.csv
```

The csv file should at least contain the following 3 columns

```
material_id, cif, <prop>
```

``<prop>`` can be arbitrary property types, like Tc in superconductors.


Split the raw data via the following script

```
python scripts/make_split.py --dir data/properties/<property> --csv <raw_data>.csv
```

The default setting will shuffle the dataset in random seed 42 and split it into train.csv, val.csv and test.csv with ratio 8:1:1. 

### Step 2: Train a property prediction model

```
python diffcsp/run.py model=prediction data=property data.subdir=<property> data.prop=<prop> data.task=<task> data.opt_target=<opt_target> exptag=<property>_<prop> expname=prediction
```

The trained model is saved in ``singlerun/<property>_<prop>/prediction``. The default 3D encoder is DimeNet++, and one can change it into more powerful encoders (e.g. Equiformer).

``<task>`` can be chosen from classification/regression.

``<opt_target>`` have different meanings for different tasks:

For classification, ``<opt_target>`` means the required class to generation.
For regression, ``<opt_target> = 1`` means to generate candidates with higher property (like Tc), while ``<opt_target> = -1`` means to generate candidates with lower property (like formation energy)

### Step 3: Train a time-dependent guidance model

```
python diffcsp/run.py model=guidance data=property data.subdir=<property> data.prop=<prop> data.task=<task> data.opt_target=<opt_target> exptag=<property>_<prop> expname=guidance
```

The trained model is saved in ``singlerun/<property>_<prop>/guidance``.

### Step 4: Generate candidates with guidance

```
python scripts/optimization.py --model_path ${PWD}/singlerun/<property>_<prop>/guidance --uncond_path ${PWD}/singlerun/2023-04-18/pure_pretrain
```

The above command will yield ``eval_opt.pt`` under the ``singlerun/<property>_<prop>/guidance`` directory, which contains 500 optimized structures.

### Step 5: Evaluate the trained model and optimized samples

```
python scripts/eval_optimization.py --dir ${PWD}/singlerun/<property>_<prop>
```

The results are logged in ``singlerun/<property>_<prop>/results`` as 

```
|-- results
    |-- summary.log
    |-- results.csv
    |-- cif
        |-- xx.cif
        ...
```

``summary.log`` summaries the results of the property prediction & guidance model. An example is provided as

```
*************** Property Prediction ***************

Test pcc: 0.4857

*************** Optimization ***************

Top-5 Results: 
489-O1Cu1: 4.4751
385-Cu8: 3.2595
249-Cu6: 3.2047
486-Cu4: 3.1240
163-Cu4: 3.0369
```

### An example script for the entire pipeline

```
export CUDA_VISIBLE_DEVICES=1

python scripts/make_split.py --dir data/properties/SuperCon --csv order_data_tc.csv

python diffcsp/run.py model=prediction data=property data.subdir=SuperCon data.prop=logtc data.task=regression data.opt_target=1 exptag=SuperCon_logtc expname=prediction

python diffcsp/run.py model=guidance data=property data.subdir=SuperCon data.prop=logtc data.task=regression data.opt_target=1 exptag=SuperCon_logtc expname=guidance

python scripts/optimization.py --model_path ${PWD}/singlerun/SuperCon_logtc/guidance --uncond_path ${PWD}/singlerun/2023-04-18/pure_pretrain

python scripts/eval_optimization.py --dir ${PWD}/singlerun/SuperCon_logtc
```

