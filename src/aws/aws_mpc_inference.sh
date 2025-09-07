export SPRINT_PATH=$HOME/sprint


python $SPRINT_PATH/src/aws/aws_launcher.py \
    --ssh_key_file=$HOME/aws_utils/aws_keys.pem \
    --instances=i-068e800fa2d5f4407,i-055fd64aa89b93226 \
    --regions=eu-central-1 \
    --master_port=5000 \
    --src_folder $SPRINT_PATH/src/ \
    run_inference.py \
        --config aws_inference_config.yaml \
        --base_path /home/ubuntu/aws-launcher-tmp/configs

        #--mode inference \
        #--examples 10 \
        #--batch_size 1 \
        #--epochs 2 \
        #--clip_threshold 0.1 \
        #--lr 0.001 \
        #--model roberta-base \
        #--softmax softmax_max \
        #--hidden_act bolt_gelu_batched \
        #--optimizer abam_bc \
        #--epsilon 10 \
        #--device cpu \
        #--subsampling mock \
        #--clear_embedding \
        #--clear_non_trainable \
        #--dataset toy \
        #--val_batch_size 128 \
        #--verbose \
        #--logits_cap_attention 50.0 \
        #--embeddings_cap 50.0 \
        #--ffa_lora \
        #--lora_rank 16 \
        #--lora_alpha 16 \
        #--lora_layers "query key value dense" \
        #--profile
        
        
        #--lora_layers "query value dense" \
        #--lora_layers "query value attention.output.dense" \

        #--profile
        #--logits_cap_classifier 10.0 \
        #--embeddings_cap 10.0 \
        #--ffa_lora \
        #--dynamic_padding \
        #--verbose \
    #--data_folder $CUSTOM_PATH/tokenized_datasets/sst2/ \

# Default parameters
# --mode train
# --epochs 5
# --batch_size 10
# --examples 50
# --lr 0.1
# --clip_threshold 1.0
# --epsilon 10
# --subsampling poisson
# --model bert-base
# --device cpu
# --optimizer dpsgd
# --softmax softmax
# --hidden_act relu


# Optimizations
# --lora (specify lor rank)
# --ffa_lora (specify lora rank)
# --clear_embedding
# --clear_non_trainable

# --optimizers:
#   - dpzero
#   - dpsgd

#--instances=i-04ecba924d307c0f8,i-0d12eca82c2870028 \
#--instances=i-0842fcf0970ad3d12,i-01ecdc27c44adcf66 \

#--instances=i-091b05bc78f5d2fe5,i-0cab0559ad8749752 \
#--device cuda:0 \

#--mode train

# g3s.xlarge (GPU) instances
#--instances=i-0ec5e9fd3739c0c3c,i-049c1908cb39cb54e,i-0c827c8ddc6ac4630 \
#--instances=i-0ec5e9fd3739c0c3c,i-049c1908cb39cb54e \

# CPU instances t2.xlarge
#--instances=i-0561e9597acb1eda5,i-0451cb56b75da5912 \


# SVM example
#--aux_files=$CRYPTEN_PATH/examples/mpc_linear_svm/mpc_linear_svm.py,$CONFIG_PATH \
#    $CRYPTEN_PATH/examples/mpc_linear_svm/launcher.py \
#        --features 50 \
#        --examples 100 \
#        --epochs 50 \
#        --lr 0.5 \
#        --skip_plaintext

# BERT
#--aux_files $CUSTOM_PATH/modeling_bert.py,$CUSTOM_PATH/utils_lora.py,$CUSTOM_PATH/dpsgd_trainer.py,$CUSTOM_PATH/tokenized_cola/notebooks/transformers_test/tokenized_cola/cola_train_dataset.pkl,$CUSTOM_PATH/eval_utils.py,$CUSTOM_PATH/aws/debug_config.yaml,$CUSTOM_PATH/aws/mpc_bert_training.py \
#   $CUSTOM_PATH/aws/bert_launcher.py \
#       --epochs 1\
#       --batch_size 10 \
#       --examples 100 \
#       --lr 0.01 \
#       --epsilon 10 \
#       --clip_threshold 1.0 \