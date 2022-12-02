1. Run compute_dom_features_no_idyom.py to compute the features that do not require IDyOM, specifying the base directory in the file beforehand. Output files will be in /output_data

2. Compute the IDyOM features (see idyom_feature_instructions.txt). Put two .dat files in the output_data directory, one for each feature.

3. Run produce_full_dom_dataset.py, specifying the base directory and the names of the .dat IDyOM files beforehand. Output files will be in /output_data