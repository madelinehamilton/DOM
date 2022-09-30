1. Make sure that the MIDI dataset has been preprocessed (see the preprocess_dataset directory)

2. Run compute_dom_features_no_idyom.py to compute the features that do not require IDyOM.

3. Compute the IDyOM features (see idyom_feature_instructions.txt). Put two .dat files in the output_data directory, one for each feature.

4. Run produce_full_dom_dataset.py. Output files will be in the output_data directory