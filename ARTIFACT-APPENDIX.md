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

- **Python Version**: Tested with Python 3.8.17
- **Hardware**: All experiments can be run on CPU, but GPU with CUDA support is recommended for larger models and datasets
- **Operating System**: Tested on macOS and Linux.

#### Dependencies and Installation

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

#### Setup Verification

To verify that everything is set up correctly:

1. **Test Data Loading**: Download and tokenize a dataset:
   ```bash
   cd src
   python tokenize_dataset.py --dataset sst2 --model_type roberta
   ```
   This should create tokenized data in `data/tokenized_dataset/roberta/sst2/`.

2. **Test DP Fine-tuning**: Run a small fine-tuning experiment:
   ```bash
   cd src
   python run_dp_finetuning.py --config configs/fine-tuning_example_cuda.yaml
   ```
   This verifies the DP training pipeline is working correctly.

3. **Test Inference**: Ensure you can load a configuration file and run inference:
   ```bash
   cd src
   python run_inference.py --config configs/inference_example.yaml
   ```


## Notes on Reusability 
The scope of this repository is to create a general framework that integrates differential privacy (DP) fine-tuning and multi-party computation (MPC) inference for transformer-based models. The overall goal is not only to reproduce our research results but to foster future research and development in privacy-preserving machine learning by providing a modular, extensible foundation.

Here we list some examples of how this artifact can be adapted and extended:

- **Different Models**: The modular architecture supports various transformer architectures beyond RoBERTa and BERT, including newer models like DeBERTa, GPT-like models, or custom transformer variants. This requires adding modeling files for both cleartext and MPC variants in `src/modeling/models`.
- **Novel Datasets**: The data loading framework can be extended to handle additional NLP tasks beyond GLUE benchmark tasks, including custom datasets for domain-specific applications.
- **Non-linear Function Approximations**: Researchers can experiment with different MPC-friendly approximations for activation functions (e.g., polynomial approximations for GELU, ReLU variants) by adding new activation modules to `src/modeling/activations`.
- **DP Techniques**: The framework supports experimentation with different noise mechanisms, clipping strategies, and privacy accounting methods beyond the current DP-SGD implementation, thanks to the integration with Opacus, by changing accounting or noise type configurations.

The modular design enables researchers to replace individual components (optimizers, activation functions, privacy mechanisms) without modifying the entire pipeline, facilitating systematic evaluation of privacy-utility trade-offs in secure transformer inference.
