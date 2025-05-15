"""
This script enables to launch predictions with SynthSeg from a CSV file containing image paths.

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

# python imports
import os
import sys
import json
import pandas as pd
from argparse import ArgumentParser

# add main folder to python path and import ./SynthSeg/predict_synthseg.py
synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(synthseg_home)
model_dir = os.path.join(synthseg_home, 'models')
labels_dir = os.path.join(synthseg_home, 'data/labels_classes_priors')
from SynthSeg.predict_synthseg import predict


# parse arguments
parser = ArgumentParser(description="SynthSeg Dataset Processor", epilog='\n')

# input/outputs
parser.add_argument("--csv", help="Path to the CSV file containing images to process.")
parser.add_argument("--path_column", default="path", help="Column name containing image paths.")
parser.add_argument("--checkpoint", default="synthseg_checkpoint.json", 
                    help="Path to store/load checkpoint file.")
parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID to use for processing. Default is 0.")
parser.add_argument("--parc", action="store_true", help="(optional) Whether to perform cortex parcellation.")
parser.add_argument("--robust", action="store_true", help="(optional) Whether to use robust predictions (slower).")
parser.add_argument("--fast", action="store_true", help="(optional) Bypass some postprocessing for faster predictions.")
parser.add_argument("--ct", action="store_true", help="(optional) Clip intensities to [0,80] for CT scans.")
parser.add_argument("--suffix_segm", default="_segm", 
                    help="(optional) Suffix to append to segmentation output. Default is '_segm'.")
parser.add_argument("--suffix_vol", default="_volumes.csv", 
                    help="(optional) Suffix to append to volumes output. Default is '_volumes.csv'.")
parser.add_argument("--suffix_post", default="_posteriors", 
                    help="(optional) Suffix to append to posteriors output. Default is '_posteriors'.")
parser.add_argument("--suffix_resample", default="_resampled", 
                    help="(optional) Suffix to append to resampled output. Default is '_resampled'.")
parser.add_argument("--suffix_qc", default="_qc.csv", 
                    help="(optional) Suffix to append to QC output. Default is '_qc.csv'.")
parser.add_argument("--post", action="store_true", help="(optional) Generate posteriors outputs.")
parser.add_argument("--resample", action="store_true", help="(optional) Generate resampled outputs.")
parser.add_argument("--qc", action="store_true", help="(optional) Generate QC outputs.")
parser.add_argument("--crop", nargs='+', type=int, help="(optional) Size of 3D patches to analyse. Default is 192.")
parser.add_argument("--threads", type=int, default=1, help="(optional) Number of cores to be used. Default is 1.")
parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")
parser.add_argument("--v1", action="store_true", help="(optional) Use SynthSeg 1.0 (updated 25/06/22).")

# check for no arguments
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

# parse commandline
args = vars(parser.parse_args())

# Check for required arguments
if args['csv'] is None:
    print("Error: --csv argument is required. Please provide a CSV file path.")
    parser.print_help()
    sys.exit(1)

# Load CSV file
if not os.path.exists(args['csv']):
    print(f"Error: CSV file {args['csv']} not found.")
    sys.exit(1)

try:
    df = pd.read_csv(args['csv'])
    print(f"Loaded {len(df)} entries from {args['csv']}")
except Exception as e:
    print(f"Error loading CSV file: {str(e)}")
    sys.exit(1)

# Check that path column exists
if args['path_column'] not in df.columns:
    print(f"Error: Column '{args['path_column']}' not found in CSV file.")
    print(f"Available columns: {', '.join(df.columns)}")
    sys.exit(1)

# Get image paths from CSV
image_paths = df[args['path_column']].tolist()
image_paths = [path for path in image_paths if path and isinstance(path, str)]
print(f"Found {len(image_paths)} valid image paths in CSV file")

# Load checkpoint if it exists
processed_images = set()
checkpoint_file = os.path.abspath(args['checkpoint'])
if os.path.exists(checkpoint_file):
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            processed_images = set(checkpoint_data.get('processed_images', []))
        print(f"Loaded checkpoint: {len(processed_images)} images already processed")
    except Exception as e:
        print(f"Error loading checkpoint file: {str(e)}")
        # Continue without checkpoint

# Filter out already processed images
pending_images = [path for path in image_paths if path not in processed_images]
if len(pending_images) < len(image_paths):
    print(f"Skipping {len(image_paths) - len(pending_images)} already processed images")
print(f"Processing {len(pending_images)} images")

# Prepare output paths based on input paths
segmentation_paths = []
posteriors_paths = [] if args['post'] else [None] * len(pending_images)
resampled_paths = [] if args['resample'] else [None] * len(pending_images)
volumes_paths = []
qc_paths = [] if args['qc'] else [None] * len(pending_images)

for img_path in pending_images:
    # Get path without extension
    base_path = img_path
    if base_path.endswith('.nii.gz'):
        base_path = base_path[:-7]
    elif base_path.endswith('.nii'):
        base_path = base_path[:-4]
    elif base_path.endswith('.mgz'):
        base_path = base_path[:-4]
    
    # Create output paths
    segmentation_paths.append(f"{base_path}{args['suffix_segm']}.nii.gz")
    volumes_paths.append(f"{base_path}{args['suffix_vol']}")
    
    if args['post']:
        posteriors_paths.append(f"{base_path}{args['suffix_post']}.nii.gz")
    
    if args['resample']:
        resampled_paths.append(f"{base_path}{args['suffix_resample']}.nii.gz")
    
    if args['qc']:
        qc_paths.append(f"{base_path}{args['suffix_qc']}")

# print SynthSeg version and checks boolean params for SynthSeg-robust
if args['robust']:
    args['fast'] = True
    assert not args['v1'], 'The flag --v1 cannot be used with --robust since SynthSeg-robust only came out with 2.0.'
    version = 'SynthSeg-robust 2.0'
else:
    version = 'SynthSeg 1.0' if args['v1'] else 'SynthSeg 2.0'
    if args['fast']:
        version += ' (fast)'
print('\n' + version + '\n')

# enforce CPU processing if necessary
if args['cpu']:
    print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if args['gpu_id'] and not args['cpu']:
    print('using GPU %s' % args['gpu_id'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])

# limit the number of threads to be used if running on CPU
import tensorflow as tf
if args['threads'] == 1:
    print('using 1 thread')
else:
    print('using %s threads' % args['threads'])
tf.config.threading.set_inter_op_parallelism_threads(args['threads'])
tf.config.threading.set_intra_op_parallelism_threads(args['threads'])

# path models
if args['robust']:
    args['path_model_segmentation'] = os.path.join(model_dir, 'synthseg_robust_2.0.h5')
else:
    args['path_model_segmentation'] = os.path.join(model_dir, 'synthseg_2.0.h5')
args['path_model_parcellation'] = os.path.join(model_dir, 'synthseg_parc_2.0.h5')
args['path_model_qc'] = os.path.join(model_dir, 'synthseg_qc_2.0.h5')

# path labels
args['labels_segmentation'] = os.path.join(labels_dir, 'synthseg_segmentation_labels_2.0.npy')
args['labels_denoiser'] = os.path.join(labels_dir, 'synthseg_denoiser_labels_2.0.npy')
args['labels_parcellation'] = os.path.join(labels_dir, 'synthseg_parcellation_labels.npy')
args['labels_qc'] = os.path.join(labels_dir, 'synthseg_qc_labels_2.0.npy')
args['names_segmentation_labels'] = os.path.join(labels_dir, 'synthseg_segmentation_names_2.0.npy')
args['names_parcellation_labels'] = os.path.join(labels_dir, 'synthseg_parcellation_names.npy')
args['names_qc_labels'] = os.path.join(labels_dir, 'synthseg_qc_names_2.0.npy')
args['topology_classes'] = os.path.join(labels_dir, 'synthseg_topological_classes_2.0.npy')
args['n_neutral_labels'] = 19

# use previous model if needed
if args['v1']:
    args['path_model_segmentation'] = os.path.join(model_dir, 'synthseg_1.0.h5')
    args['labels_segmentation'] = args['labels_segmentation'].replace('_2.0.npy', '.npy')
    args['labels_qc'] = args['labels_qc'].replace('_2.0.npy', '.npy')
    args['names_segmentation_labels'] = args['names_segmentation_labels'].replace('_2.0.npy', '.npy')
    args['names_qc_labels'] = args['names_qc_labels'].replace('_2.0.npy', '.npy')
    args['topology_classes'] = args['topology_classes'].replace('_2.0.npy', '.npy')
    args['n_neutral_labels'] = 18

# Define checkpoint update function
def update_checkpoint(processed_path):
    processed_images.add(processed_path)
    checkpoint_data = {'processed_images': list(processed_images)}
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not update checkpoint file: {str(e)}")

# Register checkpoint function for SynthSeg
checkpoint_function = update_checkpoint

# run prediction if there are pending images
if not pending_images:
    print("No images to process. All images have already been processed.")
    sys.exit(0)

# Run prediction with the modified predict function
predict(path_images=pending_images,
        path_segmentations=segmentation_paths,
        path_model_segmentation=args['path_model_segmentation'],
        labels_segmentation=args['labels_segmentation'],
        robust=args['robust'],
        fast=args['fast'],
        v1=args['v1'],
        do_parcellation=args['parc'],
        n_neutral_labels=args['n_neutral_labels'],
        names_segmentation=args['names_segmentation_labels'],
        labels_denoiser=args['labels_denoiser'],
        path_posteriors=posteriors_paths,
        path_resampled=resampled_paths,
        path_volumes=volumes_paths,
        path_model_parcellation=args['path_model_parcellation'],
        labels_parcellation=args['labels_parcellation'],
        names_parcellation=args['names_parcellation_labels'],
        path_model_qc=args['path_model_qc'],
        labels_qc=args['labels_qc'],
        path_qc_scores=qc_paths,
        names_qc=args['names_qc_labels'],
        cropping=args['crop'],
        topology_classes=args['topology_classes'],
        ct=args['ct'],
        checkpoint_function=checkpoint_function)

print("\nProcessing complete!")
print(f"Total images processed: {len(processed_images)}")
print(f"Checkpoint saved to: {checkpoint_file}")