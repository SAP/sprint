# Artifact Appendix

Paper title: **SPRINT: Scalable, Secure & Private Inference for Transformers**

Requested Badge(s):
  - [x] **Available**
  - [ ] **Functional**
  - [ ] **Reproduced**

## Description

This artifact relates to the paper **"SPRINT: Scalable, Secure & Private Inference for Transformers"**.

This artifact provides the complete implementation of SPRINT, a scalable framework that integrates differential privacy (DP) fine-tuning with multi-party computation (MPC) inference for transformer-based models. The artifact directly supports the paper's contributions by providing:

1. **DP Fine-tuning Pipeline**: Implementation of differentially private fine-tuning using Opacus, including DP-specific optimizations like parameter-efficient fine-tuning (LoRA) and noise-aware optimizers (DP-AdamBC).

2. **MPC Inference Framework**: CrypTen-based secure inference implementation with MPC optimizations including cleartext public parameters and efficient approximations of non-linear functions (GELU, etc.).

3. **Experimental Reproduction**: Complete experimental setup to reproduce the paper's results on GLUE benchmark tasks with RoBERTa models, demonstrating ~1.6× faster MPC inference than SHAFT and <1 percentage point accuracy gap compared to cleartext inference.

4. **Modular Architecture**: The `sprint_core` module provides extensible components for configuration management, model creation, data loading, training orchestration, and inference execution, enabling future research in privacy-preserving machine learning.

The artifact enables researchers to reproduce the paper's experimental results and extend the framework for novel applications in secure and private transformer inference.

### Security/Privacy Issues and Ethical Concerns

This artifact does not intentionally disable security mechanisms or run vulnerable code. However, it relies on research-grade libraries for secure and private machine learning, which have the following caveats:

- **private-transformers**: The codebase is not production-grade. For example, cryptographically secure PRNGs are recommended for sampling noise in differential privacy, but the current implementation uses standard PRNGs for performance reasons. 
- **CrypTen**: This library is intended for research and is not production-ready. It may lack some security hardening and should not be used for sensitive deployments.
- **General**: The artifact does not collect, store, or process real user data. All experiments use public datasets or synthetic data. No user study or personal data is included.

**Ethical Review**: No user study or human subject data is included, so no IRB process was required.

**Recommendation**: Users should not deploy this code in production or on sensitive data without further security review and hardening.


## Environment 

### Accessibility

The artifact is publicly accessible via the following persistent repository:

https://github.com/SAP/sprint

### Set up the environment

- **Python Version**: Tested with Python 3.9.23 (The setup script installs this version if not present)
- **Hardware**: All experiments can be run on CPU, but GPU with CUDA support is recommended for larger models and datasets
- **Operating System**: Tested on macOS and debian-based Linux distributions (e.g., Ubuntu) with apt package manager.

#### Dependencies and Installation
The repository includes a `setup.sh` script that automates the environment setup, including creating a virtual environment and installing dependencies. 

1. Clone the repository:
```bash
   git clone https://github.com/SAP/sprint.git
   cd sprint
```

2. Make setup script executable and run it:
- It is possible to run the setup script installing python either system-wide and creating a virtual environment, or installing python via conda (using the ``--conda`` flag).
```bash
   chmod +x setup.sh
   ./setup.sh --conda  # for conda installation
```

3. Activate the conda environment:
```bash
   conda activate sprint
```
or the virtual environment:
```bash
   source sprint_env/bin/activate
```

4. Set environment variable for sprint path (in the following, the command in run in the root of the cloned repo):
```bash
   export SPRINT_PATH=$(pwd)
```

Alternatively, a **manual setup process** is provided below (for linux). First with system-wide python installation and virtual environment, and then with conda (replacing step 2 and 3).

**Manual Setup Process with system-wide Python and Virtual Environment**

2. Install python version 3.9 (e.g. in linux via apt)
```bash
   sudo apt-get install python3.9 python3.9-venv python3.9-dev
```
This code may fail since latest os versions do not have python3.9 available in default repositiories (e.g., in Ubuntu22.04). IN that case you may need to run the following commands: 
```bash
   sudo apt-get update
   sudo apt-get install -y software-properties-common
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt-get update
   sudo apt-get install python3.9 python3.9-venv python3.9-dev
```

3. Setup and activate a virtual environment
```bash
   python3.9 -m venv sprint_env
   source sprint_env/bin/activate
```

4. Install dependencies:
```bash
   SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -r requirements.txt
```

*NOTE: `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` since CrypTen requirements include sklearn. Alternatively, you can download CrypTen from source and modify the requirements file (i.e., replacing `sklearn` with `scikit-learn`).*

5. **Modify `autograd_grad_sample.py` in the `private_transformers library`**:
   - The expected path with the virtual environment is `sprint_env/lib/python3.9/site-packages/private_transformers/` (it may vary depending on the OS and python version). 
   - You need to add `register_full_backward_hook` in line 97 of `autograd_grad_sample.py` (instead of `register_backward_hook`, which does not support layers with multiple autograd nodes like LoRALayers). The modified line changes from `handles.append(layer.register_backward_hook(this_backward))` to `handles.append(layer.register_full_backward_hook(this_backward))`.

6. Set environment variable for sprint path (in the following, the command in run in the root of the cloned repo):
```bash
   export SPRINT_PATH=$(pwd)
```


**Manual Setup Process with Conda**
2. Install conda (if not already installed). You can use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for a minimal installation.

3. Create and activate a conda environment with python 3.9:
```bash
   conda create -n sprint python=3.9
   conda activate sprint
```

4. Install dependencies:
```bash
   SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -r requirements.txt
```

*NOTE: `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` since CrypTen requirements include sklearn. Alternatively, you can download CrypTen from source and modify the requirements file (i.e., replacing `sklearn` with `scikit-learn`).*

5. **Modify `autograd_grad_sample.py` in the `private_transformers library`**:
   - The expected path with the virtual environment is `sprint_env/lib/python3.9/site-packages/private_transformers/` (it may vary depending on the OS and python version). 
   - You need to add `register_full_backward_hook` in line 97 of `autograd_grad_sample.py` (instead of `register_backward_hook`, which does not support layers with multiple autograd nodes like LoRALayers). The modified line changes from `handles.append(layer.register_backward_hook(this_backward))` to `handles.append(layer.register_full_backward_hook(this_backward))`.

6. Set environment variable for sprint path (in the following, the command in run in the root of the cloned repo):
```bash
   export SPRINT_PATH=$(pwd)
```


#### Setup Verification

To verify that everything is set up correctly:

1. **Swith to the `src` folder**
```bash
   cd src
```

2. **Test Data Loading**: Download and tokenize a dataset:
```bash
   python tokenize_dataset.py --dataset sst2 --model_type roberta
```
   This should create tokenized data in `$SPRINT_PATH/data/tokenized_dataset/roberta/sst2/`.

3. **Test DP Fine-tuning**: Run a small fine-tuning experiment:
```bash
   python run_dp_finetuning.py --config fine-tuning_example_cpu.yaml
```
This verifies the DP training pipeline is working correctly. The config file `$SPRINT_PATH/src/configs/fine-tuning_example_cpu.yaml` uses the CPU for fine-tuning. If you have a GPU with CUDA support, you can use the config file `$SPRINT_PATH/src/configs/fine-tuning_example_cuda.yaml`.

*Expected runtime*: ~6 hours on NVIDIA A10G (g5.xlarge AWS instance) for RoBERTa-base on SST-2 dataset. For larger datasets (e.g., MNLI) the runtime can be more than 1 day. On CPU (e.g., c6.xlarge AWS instance) the runtime is ~2x longer.

4. **Test Inference**: Run inference with CrypTen:
   
```bash
python run_inference.py --config inference_example.yaml --crypten_config crypten_inference_config.yaml
```

*Note: this examples works with a non fine-tuned roberta-base model. The model_name can be replaced with the name of a fine-tuned model.*

*Expected runtime (MPC inference with CrypTen)*: ~15 minutes on NVIDIA A10G (g5.xlarge AWS instance) for RoBERTa-base on SST-2 dataset in MPC. Batched inference is ~2x faster than non-batched inference. On CPU (e.g., c6.xlarge AWS instance) the runtime is ~1.5x longer. 

*Expected runtime (cleartext inference with PyTorch)*: ~1 minute on NVIDIA A10G (g5.xlarge AWS instance) for RoBERTa-base on SST-2 dataset. On CPU (e.g., c6.xlarge AWS instance) the runtime is ~2x longer.

For more details, see the [README](README.md) in the repository.

## Notes on Reusability 
The scope of this repository is to create a general framework that integrates differential privacy (DP) fine-tuning and multi-party computation (MPC) inference for transformer-based models. The overall goal is not only to reproduce our research results but to foster future research and development in privacy-preserving machine learning by providing a modular, extensible foundation.

Here we list some examples of how this artifact can be adapted and extended:

- **Different Models**: The modular architecture supports various transformer architectures beyond RoBERTa and BERT, including newer models like DeBERTa, GPT-like models, or custom transformer variants. This requires adding modeling files for both cleartext and MPC variants in `src/modeling/models`.
- **Novel Datasets**: The data loading framework can be extended to handle additional NLP tasks beyond GLUE benchmark tasks, including custom datasets for domain-specific applications.
- **Non-linear Function Approximations**: Researchers can experiment with different MPC-friendly approximations for activation functions (e.g., polynomial approximations for GELU, ReLU variants) by adding new activation modules to `src/modeling/activations`.
- **DP Techniques**: The framework supports experimentation with different noise mechanisms, clipping strategies, and privacy accounting methods beyond the current DP-SGD implementation, thanks to the integration with Opacus, by changing accounting or noise type configurations.

The modular design enables researchers to replace individual components (optimizers, activation functions, privacy mechanisms) without modifying the entire pipeline, facilitating systematic evaluation of privacy-utility trade-offs in secure transformer inference.
