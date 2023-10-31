# lighthloc - A light and fast implementation of hloc with TensorRT

## Installation

`hloc` requires Python >=3.7 and PyTorch >=1.1. Installing the package locally pulls the other dependencies:

```bash
git clone --recursive https://github.com/hugoycj/lighthloc/
cd lighthloc/
python setup.py develop
```

## How to use
### Data preparation
Place your images in the images/ folder. The structure of your data should be as follows:
```
-data_001
    -images
-data_002
    -images
-data_003
    -images
```
Each separate data folder houses its own images folder.

### Processing
We have a convient shell command similar to `ns-process-data` in `nerfstudio`:
```
Usage: hloc-process-data [OPTIONS]

Options:
  --data TEXT                     Path to data directory
  --match-type [exhaustive|sequential|retrival]
                                  Type of matching to perform (default:
                                  retrival)
  --feature-type [superpoint_inloc|superpoint_aachen]
                                  Type of feature extraction (default:
                                  superpoint_inloc)
  --matcher-type [lightglue|lightglue_trt|superglue]
                                  Type of feature matching (default:
                                  lightglue)
  --help                          Show this message and exit.
```

To achieve the best results in SFM (Structure from Motion), we recommend using the following command:
```
hloc-process-data --data ${INPUT_DIR} --feature-type superpoint_aachen --match-type exhaustive --matcher-type superglue
```
This combination of feature type, match type, and matcher type typically yields high-quality results in terms of accuracy.

If you are seeking a balance between speed and accuracy, you can try the following command:
```
hloc-process-data --data ${INPUT_DIR} --feature-type superpoint_aachen --match-type retrival --matcher-type lightglue
```
This configuration is optimized for faster processing while still maintaining satisfactory accuracy.

However, if your input data is extracted from a sequential video and your primary concern is speed, we recommend the following command:

If your input data is extracted from a sequential video and want to get a fast results, we recommand you to use
```
hloc-process-data --data ${INPUT_DIR} --feature-type superpoint_aachen --match-type sequential --matcher-type lightglue
```
Using the "sequential" feature type can help expedite the processing time while still providing reasonable results.

By following these recommendations, you can optimize your SFM pipeline based on your specific needs, whether you prioritize accuracy, speed, or a balance between the two.

*An experimental version of TensorRT-accelerated LightGlue has been provided. However, the overall efficiency is currently poor due to the time required to transfer intermediate data from the GPU to the CPU and then convert it to ONNX tensor. There is potential for improvement in the future, which could boost efficiency.*