"""
Simplified main script for SPRINT fine-tuning.

This script replaces the original dp_finetuning.py with a cleaner, more modular approach.
"""

import argparse
from sprint_core import ConfigManager, ExperimentRunner


def main():
    """Main entry point for SPRINT fine-tuning experiments."""
    parser = argparse.ArgumentParser(description="SPRINT Fine-tuning")
    parser.add_argument(
        "--config", 
        type=str,
        default=None,
        help="Path to the configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_from_yaml(args.config)
        
        # Create and run experiments
        experiment_runner = ExperimentRunner(config)
        experiment_runner.run_all_experiments(
            device=config.device,
            dataset_name=config.dataset_name
        )
        
        print("All experiments completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
