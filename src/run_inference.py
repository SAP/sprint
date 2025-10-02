# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

"""
Refactored SPRINT inference script with clean architecture.

This script replaces the original run_inference.py with a cleaner, more modular approach
following the same patterns as the refactored fine-tuning code.
"""

import argparse
from sprint_core import InferenceConfigManager, InferenceManager, MultiProcessLauncher


def run_inference(args):
    """Main inference function that can be run in single or multi-process mode."""
    # Create configuration from YAML file
    config = InferenceConfigManager().load_from_yaml(config_file=args.config)

    # Create and setup inference manager
    inference_manager = InferenceManager(config)
    
    try:
        # Setup environment (CrypTen initialization, device assignment)
        inference_manager.setup_environment(crypten_config_path=args.crypten_config)
        
        # Load model and data
        inference_manager.load_model()
        inference_manager.load_data()
        
        # Run inference
        if config.profile:
            print("Running profiled inference...")
            results = inference_manager.run_inference_profiled()
        else:
            results = inference_manager.run_inference()

        return results
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


def main():
    """Main entry point for SPRINT inference."""
    parser = argparse.ArgumentParser(description="SPRINT Model Inference")
    
    # Configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML inference configuration file"
    )

    parser.add_argument(
        "--crypten_config",
        type=str,
        required=True,
        help="Path to CrypTen configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration to determine execution mode
        config = InferenceConfigManager().load_from_yaml(config_file=args.config)

        if config.encrypted and config.world_size > 1 and config.multiprocess:
            # Multi-process encrypted inference
            launcher = MultiProcessLauncher(config.world_size, run_inference, args)
            launcher.start()
            launcher.join()
            launcher.terminate()
        else:
            # Single-process inference 
            run_inference(args)
            print("Inference completed successfully!")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
