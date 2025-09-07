[![REUSE status](https://api.reuse.software/badge/github.com/SAP/sprint)](https://api.reuse.software/info/github.com/SAP/sprint)

# SPRINT: Scalable, Secure & Differentially Private Inference for Transformers

## About this project

SPRINT is a scalable framework for differentially private (DP) fine-tuning and inference via multiparty computation (MPC) of transformer-based models. SPRINT is built on top of PyTorch, Opacus for DP fine-tuning and CrypTen for MPC inference.

Repository for the paper "SPRINT: Scalable, Secure & Private Inference for Transformers"

## Abstract

Machine learning as a service (MLaaS) enables deploying models for inference on cloud servers, offering scalable infrastructure and resource management.
However, MLaaS exposes user queries and model parameters to servers. 
To guarantee confidentiality of queries and model parameters, multi-party computation (MPC) ensures secure inference by distributing data and computations across multiple service providers. MPC eliminates single points of failure, mitigates provider breaches and ensures confidentiality beyond legal agreements.
Additionally, models can memorize and leak training data. 
To mitigate privacy concerns, differential privacy (DP) provides a formal privacy guarantee for training data, which can satisfied by injecting carefully calibrated noise into gradients during training.
However, naive combinations of DP and MPC amplify accuracy loss due to DP noise and MPC approximations, and incur high computational and communication overhead due to cryptographic operations.

We present **SPRINT**, the first scalable solution for efficient MPC inference on DP fine-tuned models with high accuracy.
**SPRINT** fine-tunes public pre-trained models on private data using DP, and integrates DP-specific optimizations, e.g., parameter-efficient fine-tuning and noise-aware optimizers, with MPC optimizations, e.g., cleartext public parameters and efficient approximations of non-linear functions.
We evaluate **SPRINT** on the GLUE benchmark with RoBERTa, achieving 1.6× faster MPC than the state-of-the-art non-DP solution (SHAFT). Notably, **SPRINT** maintains high accuracy during MPC inference, with $<1$ percentage point gap compared to its cleartext accuracy.

## Repository Structure
The repository is organized as follows:
- `src/`: Contains the source code for the project.
  - `run_dp_finetuning.py`: Refactored script for fine-tuning models with differential privacy.
  - `run_inference.py`: Refactored script for model inference (cleartext and MPC).
  - `sprint_core/`: Core modular components for SPRINT experiments.
    - `config_manager.py`: Configuration management and validation.
    - `model_factory.py`: Model creation and LoRA integration.
    - `data_loaders.py`: Data loading and tokenization utilities.
    - `training_manager.py`: Training orchestration and DP integration.
    - `inference_manager.py`: Inference execution and overflow handling.
    - `experiment_runner.py`: End-to-end experiment orchestration.
    - `multiprocess_launcher.py`: Multi-process execution for MPC.
  - `configs/`: Contains YAML configuration files for different experimental settings.
    - `crypten_inference_config.yaml`: Configuration file for CrypTen.
  - `modeling/`: Contains the CrypTen modeling of BERT and RoBERTa.
    - `models/`: Model implementations (clear and encrypted versions).
    - `lora/`: LoRA (Low-Rank Adaptation) implementation and utilities.
    - `activations/`: Custom activation functions for MPC.
    - `optimizers/`: DP-specific optimizers (e.g., DP-AdamBC).
  - `aws/`: Contains the refactored scripts for running MPC experiments on AWS.
   - `aws_launcher_refactored.py`: Modified AWS instance launcher from crypten.
   - `aws_mpc_inference.sh`: Script for running MPC inference on AWS.
- `data/`: Contains the datasets and models.
  - `models/`: Contains the fine-tuned models.
  - `finetuning/`: Contains the results of the fine-tuning experiments (for each dataset).
  - `inference/`: Contains the results of the MPC inference.
    - `accuracy/`: Inference accuracy results (for each dataset, encrypted and not-encrypted inference).
    - `runtime/`: Runtime and communication profiling results (from aws experiments)
- `aws/`: Contains the refactored scripts for running MPC experiments on AWS.
  - `run_mpc_experiments.py`: Main AWS MPC experiment runner.
  - `aws_launcher_refactored.py`: Improved AWS instance launcher.

## Requirements and Setup
- **Python Version**: Tested with Python 3.8.17
- **Hardware**: All experiments can be run on CPU, but GPU with CUDA support is recommended for larger models and datasets
- **Operating System**: Tested on macOS and Linux.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SAP/sprint.git
   cd sprint
   ```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
    ```

3. **Important Setup Notes**:
   - You may need to set `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` since CrypTen requirements include sklearn. Alternatively, you can download CrypTen from source and modify the requirements file.
   - **Critical Modification Required**: Add `register_full_backward_hook` in line 98 of `autograd_grad_sample.py` of the private-transformers library (`register_backward_hook` does not support layers with multiple autograd nodes like LoRALayers).

## Run DP Fine-tuning
1. Download and tokenize the dataset (for example sst2):
    ```bash
    cd src
    python tokenize_dataset.py --dataset sst2 --model_type roberta
    ```
This will download the dataset and save it in the `data` folder. The tokenized dataset will be saved in `data/tokenized_dataset/roberta/sst2/`.

2. Configure the fine-tuning parameters in `src/configs/fine-tuning_example_cuda.yaml` (or create your own config file). 

3. Run the fine-tuning script:
   ```bash
   cd src
   python run_dp_finetuning.py --config fine-tuning_example_cuda.yaml
   ```
   The fine-tuned model will be saved in the `data/models/` folder. The results (loss, validation accuracy) will be saved in the `data/finetuning/` folder. 

## Run Inference
The inference can run in cleartext or in MPC. The cleartext mode is used for local evaluation and debugging, while the MPC mode is used for secure inference. The MPC inference can be run locally on two different processes (to evaluate MPC accuracy) or on AWS with multiple machines (to evaluate the communication and overhead).

During inference, there is the possibility to apply logits capping (if not applied during fine-tuning), or to apply a different capping threshold. The capping threshold is set in the `src/configs/config_cuda0.yaml` file during fine-tuning (default value is 50.0).

The ouput of the inference process can be saved in a log file via >> `$log_file` (e.g. `log.txt`).


### Run local inference (cleartext or in MPC)
```bash
cd src
python run_inference.py --config inference_example.yaml
```


## AWS evaluation for runtime and communication overhead
The inference on different AWS machines can be run with the script in `src/aws/` folder via:
```bash
./aws_mpc_inference.sh
```

This bash scripts runs on toy data. The script used for inference is the same as for local inference, with a different config files. For the example we use `aws_inference_config.yaml`.
The data will be saved on the aws machine in the `aws-launcher-tmp` folder.
The AWS machines need to be configured with the same environment as the local machine.


## Third-parties components

The code for DP-AdamBC in `src/modeling/optimizers` has been adapted from https://github.com/ubc-systopia/DP-AdamBC.

The modeling for BERT and RoBERTA model ,in `src/modeling/models` folder, has been adapted for SPRINT fine-tuning and MPC inference from the Transformer library (https://github.com/huggingface/transformers).

The code for LoRA has been adapted from LoRA repository (https://github.com/microsoft/LoRA).

The launchers for MPC inference on AWS in the `src/aws` folder have been adapted from the CrypTen repository (https://github.com/facebookresearch/CrypTen).


### Security/Privacy Issues and Ethical Concerns

This artifact does not intentionally disable security mechanisms or run vulnerable code. However, it relies on research-grade libraries for secure and private machine learning, which have the following caveats:

- **private-transformers**: The codebase is not production-grade. For example, cryptographically secure PRNGs are recommended for sampling noise in differential privacy, but the current implementation uses standard PRNGs for performance reasons. 
- **CrypTen**: This library is intended for research and is not production-ready. It may lack some security hardening and should not be used for sensitive deployments.
- **General**: The artifact does not collect, store, or process real user data. All experiments use public datasets or synthetic data. No user study or personal data is included.

**Ethical Review**: No user study or human subject data is included, so no IRB process was required.

**Recommendation**: Users should not deploy this code in production or on sensitive data without further security review and hardening.


## Notes on Reusability (Encouraged for all badges)
The scope of this repository is to create a general framework that integrates differential privacy (DP) fine-tuning and multi-party computation (MPC) inference for transformer-based models. The overall goal is not only to reproduce our research results but to foster future research and development in privacy-preserving machine learning by providing a modular, extensible foundation.

Here we list some examples of how this artifact can be adapted and extended:

- **Different Models**: The modular architecture supports various transformer architectures beyond RoBERTa and BERT, including newer models like DeBERTa, GPT-like models, or custom transformer variants. This requires adding modeling files for both cleartext and MPC variants in `src/modeling/models`.
- **Novel Datasets**: The data loading framework can be extended to handle additional NLP tasks beyond GLUE benchmark tasks, including custom datasets for domain-specific applications.
- **Non-linear Function Approximations**: Researchers can experiment with different MPC-friendly approximations for activation functions (e.g., polynomial approximations for GELU, ReLU variants) by adding new activation modules to `src/modeling/activations`.
- **DP Techniques**: The framework supports experimentation with different noise mechanisms, clipping strategies, and privacy accounting methods beyond the current DP-SGD implementation, thanks to the integration with Opacus, by changing accounting or noise type configurations.

The modular design enables researchers to replace individual components (optimizers, activation functions, privacy mechanisms) without modifying the entire pipeline, facilitating systematic evaluation of privacy-utility trade-offs in secure transformer inference.

## Support, Feedback, Contributing

This project is open to feature requests/suggestions, bug reports etc. via [GitHub issues](https://github.com/SAP/sprint/issues). Contribution and feedback are encouraged and always welcome. For more information about how to contribute, the project structure, as well as additional contribution information, see our [Contribution Guidelines](CONTRIBUTING.md).

## Security / Disclosure
If you find any bug that may be a security problem, please follow our instructions at [in our security policy](https://github.com/SAP/sprint/security/policy) on how to report it. Please do not create GitHub issues for security-related doubts or problems.

## Code of Conduct

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone. By participating in this project, you agree to abide by its [Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md) at all times.

## Licensing

Copyright 2025 SAP SE or an SAP affiliate company and sprint contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP/sprint).

