# SynthSeg

> **Note about this fork**
>
> This repository is a fork of [Photo-SynthSeg](https://github.com/MGH-LEMoN/Photo-SynthSeg), which itself is a fork of the original [SynthSeg](https://github.com/BBillot/SynthSeg). This version adds the following enhancements:
> 
> 1. **Modern Python and TensorFlow Support**: Compatibility with Python 3.11 and TensorFlow 2.15
> 2. **CSV-based Batch Processing**: Process multiple images defined in a CSV file with a single command
> 3. **Checkpointing**: Automatically resume processing from where it left off if interrupted

SynthSeg, is the first deep learning tool for segmentation of brain scans of
any contrast and resolution. SynthSeg works out-of-the-box without any retraining, and is robust to:
- any contrast
- any resolution up to 10mm slice spacing
- a wide array of populations: from young and healthy to ageing and diseased
- scans with or without preprocessing: bias field correction, skull stripping, normalisation, etc.
- white matter lesions.
\
\
![Generation examples](data/README_figures/segmentations.png)


\
SynthSeg was first presented for the automated segmentation of brain scans of any contrast and resolution.

**SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining** \
B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias \
Medical Image Analysis (2023) \
[ [article](https://www.sciencedirect.com/science/article/pii/S1361841523000506) | [arxiv](https://arxiv.org/abs/2107.09559) | [bibtex](bibtex.bib) ]
\
\
Then, we extended it to work on heterogeneous clinical scans, and to perform cortical parcellation and automated 
quality control.

**Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets** \
B. Billot, M. Colin, Y. Cheng, S.E. Arnold, S. Das, J.E. Iglesias \
PNAS (2023) \
[ [article](https://www.pnas.org/doi/full/10.1073/pnas.2216399120#bibliography) | [arxiv](https://arxiv.org/abs/2203.01969) | [bibtex](bibtex.bib) ]

\
Here, we distribute our model to enable users to run SynthSeg on their own data. We emphasise that 
predictions are always given at 1mm isotropic resolution (regardless of the input resolution). The code can be run on
the GPU (~15s per scan) or on the CPU (~1min).


----------------

### New features and updates

\
15/05/2025: **Added support for Python 3.11 and TensorFlow 2.15 + CSV processing with checkpointing!** 🚀 \
This fork now supports modern Python and TensorFlow environments, making it easier to integrate with current machine learning workflows. We've also added a new CSV-based batch processing system with automatic checkpointing to handle large datasets more efficiently. See the [Processing images from a CSV file](#processing-images-from-a-csv-file) section for more details.

\
01/03/2023: **The papers for SynthSeg and SynthSeg 2.0 are out! :open_book: :open_book:** \
After a long review process for SynthSeg (Medical Image Analysis), and a much faster one for SynthSeg 2.0 (PNAS), both
papers have been accepted nearly at the same time ! See the references above, or in the citation section.

\
04/10/2022: **SynthSeg is available with Matlab!** :star: \
We are delighted that Matlab 2022b (and onwards) now includes SynthSeg in its Medical Image
Toolbox. They have a [documented example](https://www.mathworks.com/help/medical-imaging/ug/Brain-MRI-Segmentation-Using-Trained-3-D-U-Net.html)
on how to use it. But, to simplify things, we wrote our own Matlab wrapper, which you can call in one single line. 
Just download [this zip file](https://liveuclac-my.sharepoint.com/:u:/g/personal/rmappmb_ucl_ac_uk/EctEe3hOP8dDh1hYHlFS_rUBo80yFg7MQY5WnagHlWcS6A?e=e8bK0f),
uncompress it, open Matlab, and type `help SynthSeg` for instructions.

\
29/06/2022: **SynthSeg 2.0 is out !** :v: \
In addition to whole-brain segmentation, it now also performs **Cortical parcellation, automated QC, and intracranial 
volume (ICV) estimation** (see figure below). Also, most of these features are compatible with SynthSeg 1.0. (see table).
\
\
![new features](data/README_figures/new_features.png)

![table versions](data/README_figures/table_versions.png)

\
01/03/2022: **Robust version** :hammer: \
SynthSeg sometimes falters on scans with low signal-to-noise ratio, or with very low tissue contrast. For this reason, 
we developed a new model for increased robustness, named "SynthSeg-robust". You can use this mode when SynthSeg gives 
results like in the figure below:
\
\
![Robust](data/README_figures/robust.png)

\
29/10/2021: **SynthSeg is now available on the dev version of
[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) !!** :tada: \
See [here](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg) on how to use it.

----------------

### Try it in one command !

Once all the python packages are installed (see below), you can simply test SynthSeg on your own data with:
```
python ./scripts/commands/SynthSeg_predict.py --i <input> --o <output> [--parc --robust --ct --vol <vol> --qc <qc> --post <post> --resample <resample>]
```


where:
- `<input>` path to a scan to segment, or to a folder. This can also be the path to a text file, where each line is the
path of an image to segment.
- `<output>` path where the output segmentations will be saved. This must be the same type as `<input>` (i.e., the path 
to a file, a folder, or a text file where each line is the path to an output segmentation).
- `--parc` (optional) to perform cortical parcellation in addition to whole-brain segmentation.
- `--robust` (optional) to use the variant for increased robustness (e.g., when analysing clinical data with large space
spacing). This can be slower than the other model.
- `--ct` (optional) use on CT scans in Hounsfield scale. It clips intensities to [0, 80].
- `<vol>` (optional) path to a CSV file where the volumes (in mm<sup>3</sup>) of all segmented regions will be saved for all scans 
(e.g. /path/to/volumes.csv). If `<input>` is a text file, so must be `<vol>`, for which each line is the path to a 
different CSV file corresponding to one subject only.
- `<qc>` (optional) path to a CSV file where QC scores will be saved. The same formatting requirements as `<vol>` apply.
- `<post>` (optional) path where the posteriors, given as soft probability maps, will be saved (same formatting 
requirements as for `<output>`).
- `<resample>` (optional) SynthSeg segmentations are always given at 1mm isotropic resolution. Hence, 
images are always resampled internally to this resolution (except if they are already at 1mm resolution). 
Use this flag to save the resampled images (same formatting requirements as for `<output>`).

Additional optional flags are also available:
- `--cpu`: (optional) to enforce the code to run on the CPU, even if a GPU is available.
- `--threads`: (optional) number of threads to be used by Tensorflow (default uses one core). Increase it to decrease 
the runtime when using the CPU version.
- `--crop`: (optional) to crop the input images to a given shape before segmentation. This must be divisible by 32.
Images are cropped around their centre, and their segmentations are given at the original size. It can be given as a 
single (i.e., `--crop 160`), or several integers (i.e, `--crop 160 128 192`, ordered in RAS coordinates). By default the
whole image is processed. Use this flag for faster analysis or to fit in your GPU.
- `--fast`: (optional) to disable some operations for faster prediction (twice as fast, but slightly less accurate). 
This doesn't apply when the --robust flag is used.
- `--v1`: (optional) to run the first version of SynthSeg (SynthSeg 1.0, updated 29/06/2022).


**IMPORTANT:** SynthSeg always give results at 1mm isotropic resolution, regardless of the input. However, this can 
cause some viewers to not correctly overlay segmentations on their corresponding images. In this case, you can use the
`--resample` flag to obtain a resampled image that lives in the same space as the segmentation, such that they can be 
visualised together with any viewer.

The complete list of segmented structures is available in [labels table.txt](data/labels%20table.txt) along with their
corresponding values. This table also details the order in which the posteriors maps are sorted.


----------------

### Processing images from a CSV file

This fork adds a new command for processing images listed in a CSV file with built-in checkpointing, which is especially useful for large datasets:

```bash
python ./scripts/commands/SynthSeg_dataset.py --csv <csv_file> [--path_column <column_name>] [--checkpoint <checkpoint_file>] [options]
```

**Required arguments:**
- `<csv_file>`: Path to a CSV file containing paths to the images that need to be processed

**Optional arguments:**
- `--path_column <column_name>`: Name of the column in the CSV file that contains image paths (default: "path")
- `--checkpoint <checkpoint_file>`: Path to store/load checkpoint data (default: "synthseg_checkpoint.json")
- All standard SynthSeg options are supported (--parc, --robust, --fast, etc.)

**Example usage:**
```bash
python ./scripts/commands/SynthSeg_dataset.py --csv my_dataset.csv --path_column image_path --robust --parc --qc --threads 4
```

**Checkpoint functionality:**
The script automatically keeps track of processed images and maintains a checkpoint file. If the process is interrupted, you can simply run the same command again, and it will resume from where it left off, skipping already processed images. The checkpoint file is a JSON file containing a list of all processed image paths.

**Output files:**
For each input image path in the CSV, the script generates:
- A segmentation file (with suffix "_segm.nii.gz" by default)
- A volumes CSV file (with suffix "_volumes.csv" by default)
- Optional posteriors, resampled images, and QC scores based on the provided flags

**Custom suffixes:**
You can customize the output file suffixes using these options:
- `--suffix_segm`: Suffix for segmentation outputs (default: "_segm") 
- `--suffix_vol`: Suffix for volume CSV files (default: "_volumes.csv")
- `--suffix_post`: Suffix for posterior outputs (default: "_posteriors")
- `--suffix_resample`: Suffix for resampled outputs (default: "_resampled")
- `--suffix_qc`: Suffix for QC outputs (default: "_qc.csv")

----------------

### Installation

1. Clone this repository.

2. Create a virtual environment with Python 3.11 and install the required packages:

```bash
# Conda approach using the requirements file
conda create -n synthseg python=3.11 -y
conda activate synthseg
pip install -r requirements.txt
```

3. Go to this link [UCL dropbox](https://liveuclac-my.sharepoint.com/:f:/g/personal/rmappmb_ucl_ac_uk/EtlNnulBSUtAvOP6S99KcAIBYzze7jTPsmFk2_iHqKDjEw?e=rBP0RO), and download the missing models. Then simply copy them to [models](models).

4. CUDA and cuDNN dependencies will be installed automatically with the TensorFlow package. If you encounter GPU-related issues, make sure your NVIDIA drivers are up to date.

That's it ! You're now ready to use SynthSeg ! :tada:


----------------

### How does it work ?

In short, we train a network with synthetic images sampled on the fly from a generative model based on the forward
model of Bayesian segmentation. Crucially, we adopt a domain randomisation strategy where we fully randomise the 
generation parameters which are drawn at each minibatch from uninformative uniform priors. By exposing the network to 
extremely variable input data, we force it to learn domain-agnostic features. As a result, SynthSeg is able to readily 
segment real scans of any target domain, without retraining or fine-tuning. 

The following figure first illustrates the workflow of a training iteration, and then provides an overview of the 
different steps of the generative model:
\
\
![Overview](data/README_figures/overview.png)
\
\
Finally we show additional examples of the synthesised images along with an overlay of their target segmentations:
\
\
![Training data](data/README_figures/training_data.png)
\
\
If you are interested to learn more about SynthSeg, you can read the associated publication (see below), and watch this
presentation, which was given at MIDL 2020 for a related article on a preliminary version of SynthSeg (robustness to
MR contrast but not resolution).
\
\
[![Talk SynthSeg](data/README_figures/youtube_link.png)](https://www.youtube.com/watch?v=Bfp3cILSKZg&t=1s)


----------------

### Train your own model

This repository contains all the code and data necessary to train, validate, and test your own network. Importantly, the
proposed method only requires a set of anatomical segmentations to be trained (no images), which we include in 
[data](data/training_label_maps). While the provided functions are thoroughly documented, we highly recommend to start 
with the following tutorials:

- [1-generation_visualisation](scripts/tutorials/1-generation_visualisation.py): This very simple script shows examples
of the synthetic images used to train SynthSeg.

- [2-generation_explained](scripts/tutorials/2-generation_explained.py): This second script describes all the parameters
used to control the generative model. We advise you to thoroughly follow this tutorial, as it is essential to understand
how the synthetic data is formed before you start training your own models.

- [3-training](scripts/tutorials/3-training.py): This scripts re-uses the parameters explained in the previous tutorial
and focuses on the learning/architecture parameters. The script here is the very one we used to train SynthSeg !

- [4-prediction](scripts/tutorials/4-prediction.py): This scripts shows how to make predictions, once the network has 
been trained.

- [5-generation_advanced](scripts/tutorials/5-generation_advanced.py): Here we detail more advanced generation options, 
in the case of training a version of SynthSeg that is specific to a given contrast and/or resolution (although these
types of variants were shown to be outperformed by the SynthSeg model trained in the 3rd tutorial).

- [6-intensity_estimation](scripts/tutorials/6-intensity_estimation.py): This script shows how to estimate the 
Gaussian priors of the GMM when training a contrast-specific version of SynthSeg.

- [7-synthseg+](scripts/tutorials/7-synthseg+.py): Finally, we show how the robust version of SynthSeg was 
trained.

These tutorials cover a lot of materials and will enable you to train your own SynthSeg model. Moreover, even more 
detailed information is provided in the docstrings of all functions, so don't hesitate to have a look at these !


----------------

### Content

- [SynthSeg](SynthSeg): this is the main folder containing the generative model and training function:

  - [labels_to_image_model.py](SynthSeg/labels_to_image_model.py): contains the generative model for MRI scans.
  
  - [brain_generator.py](SynthSeg/brain_generator.py): contains the class `BrainGenerator`, which is a wrapper around 
  `labels_to_image_model`. New images can simply be generated by instantiating an object of this class, and call the 
  method `generate_image()`.
  
  - [training.py](SynthSeg/training.py): contains code to train the segmentation network (with explanations for all 
  training parameters). This function also shows how to integrate the generative model in a training setting.
  
  - [predict.py](SynthSeg/predict.py): prediction and testing.
  
  - [predict_synthseg.py](SynthSeg/predict_synthseg.py): main prediction function used by the command-line scripts.

- [scripts/commands](scripts/commands): contains command-line tools for using SynthSeg:

  - [SynthSeg_predict.py](scripts/commands/SynthSeg_predict.py): main script for segmenting individual images or folders.
  
  - [SynthSeg_dataset.py](scripts/commands/SynthSeg_dataset.py): new script for processing images from a CSV file with checkpointing.
   
  - [validate.py](SynthSeg/validate.py): includes code for validation (which has to be done offline on real images).
 
- [models](models): this is where you will find the trained model for SynthSeg.
 
- [data](data): this folder contains some examples of brain label maps if you wish to train your own SynthSeg model.
 
- [script](scripts): contains tutorials as well as scripts to launch trainings and testings from a terminal.

- [ext](ext): includes external packages, especially the *lab2im* package, and a modified version of *neuron*.


----------------

### Citation/Contact

This code is under [Apache 2.0](LICENSE.txt) licensing. 

- If you use the **cortical parcellation**, **automated QC**, or **robust version**, please cite the following paper:

**Robust machine learning segmentation for large-scale analysisof heterogeneous clinical brain MRI datasets** \
B. Billot, M. Colin, Y. Cheng, S.E. Arnold, S. Das, J.E. Iglesias \
PNAS (2023) \
[ [article](https://www.pnas.org/doi/full/10.1073/pnas.2216399120#bibliography) | [arxiv](https://arxiv.org/abs/2203.01969) | [bibtex](bibtex.bib) ]


- Otherwise, please cite:

**SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining** \
B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias \
Medical Image Analysis (2023) \
[ [article](https://www.sciencedirect.com/science/article/pii/S1361841523000506) | [arxiv](https://arxiv.org/abs/2107.09559) | [bibtex](bibtex.bib) ]

If you have any question regarding the usage of this code, or any suggestions to improve it, please raise an issue or 
contact us at: bbillot@mit.edu
