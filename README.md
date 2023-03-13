# AI Image Enhancing

## Overview

This mockup is based on [CppFlow](https://github.com/serizba/cppflow) that uses [Tensorflow C API](https://www.tensorflow.org/install/lang_c) to run pre-trained models from [Real-ESRGAN project](https://github.com/xinntao/Real-ESRGAN).

## Prerequisites
- [Setup Tensorflow for Visual Studio](http://iamsurya.com/using-libtensorflow-dlls-in-a-visual-studio-project/) (install it using at least TF 2.6)
- [Setup OpenCV for Visual Studio](https://www.youtube.com/watch?v=unSce_GPwto)

No need to install [CppFlow](https://github.com/serizba/cppflow) as it's already integrated inside the project.

## Build

The project needs C++17 to build.

Steps:
- Open .sln file in Visual Studio
- Set `Release` and `x64` as build options
- Build the solution

### Models

Pre-trained models come from [Real-ESRGAN project](https://github.com/xinntao/Real-ESRGAN).
Models are located in the `models/` directory. The project needs `.pb` file (Tensorflow models) in order to work.

Get inputs and outputs info from the model, you need to install Tensorflow first:
```bash
saved_model_cli show --dir models/regular --tag_set serve --signature_def serving_default
```

## Sources

Model: https://tfhub.dev/captain-pool/esrgan-tf2/1
