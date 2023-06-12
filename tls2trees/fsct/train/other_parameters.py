import os
import numpy as np
import fsct

# Don't change these unless you really understand what you are doing with them/are learning the code base.
# These have been tuned to work on most high resolution forest point clouds without changing them.
other_parameters = dict(preprocess_train_datasets=1,
                        preprocess_validation_datasets=1,
                        clean_sample_directories=1,  # Deletes all samples in the sample directories.
                        perform_validation_during_training=1,
                        # Useful for visually checking how well the model is learning. 
                        # Saves a set of samples called "latest_prediction.las" in the "FSCT/data/" 
                        # directory. Samples have label and prediction values.
                        generate_point_cloud_vis=0,          
                        load_existing_model=1,
                        learning_rate=0.000025,
                        input_point_cloud=None,
                        model_filename="modelV2.pth",
                        sample_box_size_m=np.array([6, 6, 6]),
                        sample_box_overlap=[0.5, 0.5, 0.5],
                        min_sample_points=1000,
                        max_sample_points=20000,
                        subsample=False,
                        subsampling_min_spacing=0.025,
                        cpu_cores=0,  
                        dl_cpu_cores=1,  # Setting this higher can cause CUDA issues on Windows.
                        train_batch_size=2,
                        validation_batch_size=2,
                        device="cuda",  # set to "cuda" or "cpu"
                       )