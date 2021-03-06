# Used for testing full ResNet-50 on image classification tasks
resnet_var_dict = { \
'resnet_model/batch_normalization/': 'res_net/res_block/bn_1/', \
'resnet_model/batch_normalization_1/': 'res_net/res_block/bn_2/', \
'resnet_model/batch_normalization_2/': 'res_net/res_block/bn_3/', \
'resnet_model/batch_normalization_3/': 'res_net/res_block_1/bn_1/', \
'resnet_model/batch_normalization_4/': 'res_net/res_block_1/bn_2/', \
'resnet_model/batch_normalization_5/': 'res_net/res_block_1/bn_3/', \
'resnet_model/batch_normalization_6/': 'res_net/res_block_2/bn_1/', \
'resnet_model/batch_normalization_7/': 'res_net/res_block_2/bn_2/', \
'resnet_model/batch_normalization_8/': 'res_net/res_block_2/bn_3/', \
'resnet_model/batch_normalization_9/': 'res_net/res_block_3/bn_1/', \
'resnet_model/batch_normalization_10/': 'res_net/res_block_3/bn_2/', \
'resnet_model/batch_normalization_11/': 'res_net/res_block_3/bn_3/', \
'resnet_model/batch_normalization_12/': 'res_net/res_block_4/bn_1/', \
'resnet_model/batch_normalization_13/': 'res_net/res_block_4/bn_2/', \
'resnet_model/batch_normalization_14/': 'res_net/res_block_4/bn_3/', \
'resnet_model/batch_normalization_15/': 'res_net/res_block_5/bn_1/', \
'resnet_model/batch_normalization_16/': 'res_net/res_block_5/bn_2/', \
'resnet_model/batch_normalization_17/': 'res_net/res_block_5/bn_3/', \
'resnet_model/batch_normalization_18/': 'res_net/res_block_6/bn_1/', \
'resnet_model/batch_normalization_19/': 'res_net/res_block_6/bn_2/', \
'resnet_model/batch_normalization_20/': 'res_net/res_block_6/bn_3/', \
'resnet_model/batch_normalization_21/': 'res_net/res_block_7/bn_1/', \
'resnet_model/batch_normalization_22/': 'res_net/res_block_7/bn_2/', \
'resnet_model/batch_normalization_23/': 'res_net/res_block_7/bn_3/', \
'resnet_model/batch_normalization_24/': 'res_net/res_block_8/bn_1/', \
'resnet_model/batch_normalization_25/': 'res_net/res_block_8/bn_2/', \
'resnet_model/batch_normalization_26/': 'res_net/res_block_8/bn_3/', \
'resnet_model/batch_normalization_27/': 'res_net/res_block_9/bn_1/', \
'resnet_model/batch_normalization_28/': 'res_net/res_block_9/bn_2/', \
'resnet_model/batch_normalization_29/': 'res_net/res_block_9/bn_3/', \
'resnet_model/batch_normalization_30/': 'res_net/res_block_10/bn_1/', \
'resnet_model/batch_normalization_31/': 'res_net/res_block_10/bn_2/', \
'resnet_model/batch_normalization_32/': 'res_net/res_block_10/bn_3/', \
'resnet_model/batch_normalization_33/': 'res_net/res_block_11/bn_1/', \
'resnet_model/batch_normalization_34/': 'res_net/res_block_11/bn_2/', \
'resnet_model/batch_normalization_35/': 'res_net/res_block_11/bn_3/', \
'resnet_model/batch_normalization_36/': 'res_net/res_block_12/bn_1/', \
'resnet_model/batch_normalization_37/': 'res_net/res_block_12/bn_2/', \
'resnet_model/batch_normalization_38/': 'res_net/res_block_12/bn_3/', \
'resnet_model/batch_normalization_39/': 'res_net/res_block_13/bn_1/', \
'resnet_model/batch_normalization_40/': 'res_net/res_block_13/bn_2/', \
'resnet_model/batch_normalization_41/': 'res_net/res_block_13/bn_3/', \
'resnet_model/batch_normalization_42/': 'res_net/res_block_14/bn_1/', \
'resnet_model/batch_normalization_43/': 'res_net/res_block_14/bn_2/', \
'resnet_model/batch_normalization_44/': 'res_net/res_block_14/bn_3/', \
'resnet_model/batch_normalization_45/': 'res_net/res_block_15/bn_1/', \
'resnet_model/batch_normalization_46/': 'res_net/res_block_15/bn_2/', \
'resnet_model/batch_normalization_47/': 'res_net/res_block_15/bn_3/', \
'resnet_model/batch_normalization_48/': 'res_net/bn_final/', \
'resnet_model/conv2d/': 'res_net/conv_init/', \
'resnet_model/conv2d_1/': 'res_net/res_block/conv_0/', \
'resnet_model/conv2d_2/': 'res_net/res_block/conv_1/', \
'resnet_model/conv2d_3/': 'res_net/res_block/conv_2/', \
'resnet_model/conv2d_4/': 'res_net/res_block/conv_3/', \
'resnet_model/conv2d_5/': 'res_net/res_block_1/conv_1/', \
'resnet_model/conv2d_6/': 'res_net/res_block_1/conv_2/', \
'resnet_model/conv2d_7/': 'res_net/res_block_1/conv_3/', \
'resnet_model/conv2d_8/': 'res_net/res_block_2/conv_1/', \
'resnet_model/conv2d_9/': 'res_net/res_block_2/conv_2/', \
'resnet_model/conv2d_10/': 'res_net/res_block_2/conv_3/', \
'resnet_model/conv2d_11/': 'res_net/res_block_3/conv_0/', \
'resnet_model/conv2d_12/': 'res_net/res_block_3/conv_1/', \
'resnet_model/conv2d_13/': 'res_net/res_block_3/conv_2/', \
'resnet_model/conv2d_14/': 'res_net/res_block_3/conv_3/', \
'resnet_model/conv2d_15/': 'res_net/res_block_4/conv_1/', \
'resnet_model/conv2d_16/': 'res_net/res_block_4/conv_2/', \
'resnet_model/conv2d_17/': 'res_net/res_block_4/conv_3/', \
'resnet_model/conv2d_18/': 'res_net/res_block_5/conv_1/', \
'resnet_model/conv2d_19/': 'res_net/res_block_5/conv_2/', \
'resnet_model/conv2d_20/': 'res_net/res_block_5/conv_3/', \
'resnet_model/conv2d_21/': 'res_net/res_block_6/conv_1/', \
'resnet_model/conv2d_22/': 'res_net/res_block_6/conv_2/', \
'resnet_model/conv2d_23/': 'res_net/res_block_6/conv_3/', \
'resnet_model/conv2d_24/': 'res_net/res_block_7/conv_0/', \
'resnet_model/conv2d_25/': 'res_net/res_block_7/conv_1/', \
'resnet_model/conv2d_26/': 'res_net/res_block_7/conv_2/', \
'resnet_model/conv2d_27/': 'res_net/res_block_7/conv_3/', \
'resnet_model/conv2d_28/': 'res_net/res_block_8/conv_1/', \
'resnet_model/conv2d_29/': 'res_net/res_block_8/conv_2/', \
'resnet_model/conv2d_30/': 'res_net/res_block_8/conv_3/', \
'resnet_model/conv2d_31/': 'res_net/res_block_9/conv_1/', \
'resnet_model/conv2d_32/': 'res_net/res_block_9/conv_2/', \
'resnet_model/conv2d_33/': 'res_net/res_block_9/conv_3/', \
'resnet_model/conv2d_34/': 'res_net/res_block_10/conv_1/', \
'resnet_model/conv2d_35/': 'res_net/res_block_10/conv_2/', \
'resnet_model/conv2d_36/': 'res_net/res_block_10/conv_3/', \
'resnet_model/conv2d_37/': 'res_net/res_block_11/conv_1/', \
'resnet_model/conv2d_38/': 'res_net/res_block_11/conv_2/', \
'resnet_model/conv2d_39/': 'res_net/res_block_11/conv_3/', \
'resnet_model/conv2d_40/': 'res_net/res_block_12/conv_1/', \
'resnet_model/conv2d_41/': 'res_net/res_block_12/conv_2/', \
'resnet_model/conv2d_42/': 'res_net/res_block_12/conv_3/', \
'resnet_model/conv2d_43/': 'res_net/res_block_13/conv_0/', \
'resnet_model/conv2d_44/': 'res_net/res_block_13/conv_1/', \
'resnet_model/conv2d_45/': 'res_net/res_block_13/conv_2/', \
'resnet_model/conv2d_46/': 'res_net/res_block_13/conv_3/', \
'resnet_model/conv2d_47/': 'res_net/res_block_14/conv_1/', \
'resnet_model/conv2d_48/': 'res_net/res_block_14/conv_2/', \
'resnet_model/conv2d_49/': 'res_net/res_block_14/conv_3/', \
'resnet_model/conv2d_50/': 'res_net/res_block_15/conv_1/', \
'resnet_model/conv2d_51/': 'res_net/res_block_15/conv_2/', \
'resnet_model/conv2d_52/': 'res_net/res_block_15/conv_3/', \
'resnet_model/dense/': 'res_net/fc/', \
}

# For initializing pre-trained weights in a ResNet-50-C4 Faster R-CNN implementation
resnet_C4_var_dict = { \
'resnet_model/batch_normalization/': 'res_net/res_block/bn_1/', \
'resnet_model/batch_normalization_1/': 'res_net/res_block/bn_2/', \
'resnet_model/batch_normalization_2/': 'res_net/res_block/bn_3/', \
'resnet_model/batch_normalization_3/': 'res_net/res_block_1/bn_1/', \
'resnet_model/batch_normalization_4/': 'res_net/res_block_1/bn_2/', \
'resnet_model/batch_normalization_5/': 'res_net/res_block_1/bn_3/', \
'resnet_model/batch_normalization_6/': 'res_net/res_block_2/bn_1/', \
'resnet_model/batch_normalization_7/': 'res_net/res_block_2/bn_2/', \
'resnet_model/batch_normalization_8/': 'res_net/res_block_2/bn_3/', \
'resnet_model/batch_normalization_9/': 'res_net/res_block_3/bn_1/', \
'resnet_model/batch_normalization_10/': 'res_net/res_block_3/bn_2/', \
'resnet_model/batch_normalization_11/': 'res_net/res_block_3/bn_3/', \
'resnet_model/batch_normalization_12/': 'res_net/res_block_4/bn_1/', \
'resnet_model/batch_normalization_13/': 'res_net/res_block_4/bn_2/', \
'resnet_model/batch_normalization_14/': 'res_net/res_block_4/bn_3/', \
'resnet_model/batch_normalization_15/': 'res_net/res_block_5/bn_1/', \
'resnet_model/batch_normalization_16/': 'res_net/res_block_5/bn_2/', \
'resnet_model/batch_normalization_17/': 'res_net/res_block_5/bn_3/', \
'resnet_model/batch_normalization_18/': 'res_net/res_block_6/bn_1/', \
'resnet_model/batch_normalization_19/': 'res_net/res_block_6/bn_2/', \
'resnet_model/batch_normalization_20/': 'res_net/res_block_6/bn_3/', \
'resnet_model/batch_normalization_21/': 'res_net/res_block_7/bn_1/', \
'resnet_model/batch_normalization_22/': 'res_net/res_block_7/bn_2/', \
'resnet_model/batch_normalization_23/': 'res_net/res_block_7/bn_3/', \
'resnet_model/batch_normalization_24/': 'res_net/res_block_8/bn_1/', \
'resnet_model/batch_normalization_25/': 'res_net/res_block_8/bn_2/', \
'resnet_model/batch_normalization_26/': 'res_net/res_block_8/bn_3/', \
'resnet_model/batch_normalization_27/': 'res_net/res_block_9/bn_1/', \
'resnet_model/batch_normalization_28/': 'res_net/res_block_9/bn_2/', \
'resnet_model/batch_normalization_29/': 'res_net/res_block_9/bn_3/', \
'resnet_model/batch_normalization_30/': 'res_net/res_block_10/bn_1/', \
'resnet_model/batch_normalization_31/': 'res_net/res_block_10/bn_2/', \
'resnet_model/batch_normalization_32/': 'res_net/res_block_10/bn_3/', \
'resnet_model/batch_normalization_33/': 'res_net/res_block_11/bn_1/', \
'resnet_model/batch_normalization_34/': 'res_net/res_block_11/bn_2/', \
'resnet_model/batch_normalization_35/': 'res_net/res_block_11/bn_3/', \
'resnet_model/batch_normalization_36/': 'res_net/res_block_12/bn_1/', \
'resnet_model/batch_normalization_37/': 'res_net/res_block_12/bn_2/', \
'resnet_model/batch_normalization_38/': 'res_net/res_block_12/bn_3/', \
'resnet_model/batch_normalization_39/': 'roi_head/time_distributed/res_block/bn_1/', \
'resnet_model/batch_normalization_40/': 'roi_head/time_distributed/res_block/bn_2/', \
'resnet_model/batch_normalization_41/': 'roi_head/time_distributed/res_block/bn_3/', \
'resnet_model/batch_normalization_42/': 'roi_head/time_distributed_1/res_block/bn_1/', \
'resnet_model/batch_normalization_43/': 'roi_head/time_distributed_1/res_block/bn_2/', \
'resnet_model/batch_normalization_44/': 'roi_head/time_distributed_1/res_block/bn_3/', \
'resnet_model/batch_normalization_45/': 'roi_head/time_distributed_2/res_block/bn_1/', \
'resnet_model/batch_normalization_46/': 'roi_head/time_distributed_2/res_block/bn_2/', \
'resnet_model/batch_normalization_47/': 'roi_head/time_distributed_2/res_block/bn_3/', \
'resnet_model/batch_normalization_48/': 'roi_head/time_distributed_3/bn_final/', \
'resnet_model/conv2d/': 'res_net/conv_init/', \
'resnet_model/conv2d_1/': 'res_net/res_block/conv_0/', \
'resnet_model/conv2d_2/': 'res_net/res_block/conv_1/', \
'resnet_model/conv2d_3/': 'res_net/res_block/conv_2/', \
'resnet_model/conv2d_4/': 'res_net/res_block/conv_3/', \
'resnet_model/conv2d_5/': 'res_net/res_block_1/conv_1/', \
'resnet_model/conv2d_6/': 'res_net/res_block_1/conv_2/', \
'resnet_model/conv2d_7/': 'res_net/res_block_1/conv_3/', \
'resnet_model/conv2d_8/': 'res_net/res_block_2/conv_1/', \
'resnet_model/conv2d_9/': 'res_net/res_block_2/conv_2/', \
'resnet_model/conv2d_10/': 'res_net/res_block_2/conv_3/', \
'resnet_model/conv2d_11/': 'res_net/res_block_3/conv_0/', \
'resnet_model/conv2d_12/': 'res_net/res_block_3/conv_1/', \
'resnet_model/conv2d_13/': 'res_net/res_block_3/conv_2/', \
'resnet_model/conv2d_14/': 'res_net/res_block_3/conv_3/', \
'resnet_model/conv2d_15/': 'res_net/res_block_4/conv_1/', \
'resnet_model/conv2d_16/': 'res_net/res_block_4/conv_2/', \
'resnet_model/conv2d_17/': 'res_net/res_block_4/conv_3/', \
'resnet_model/conv2d_18/': 'res_net/res_block_5/conv_1/', \
'resnet_model/conv2d_19/': 'res_net/res_block_5/conv_2/', \
'resnet_model/conv2d_20/': 'res_net/res_block_5/conv_3/', \
'resnet_model/conv2d_21/': 'res_net/res_block_6/conv_1/', \
'resnet_model/conv2d_22/': 'res_net/res_block_6/conv_2/', \
'resnet_model/conv2d_23/': 'res_net/res_block_6/conv_3/', \
'resnet_model/conv2d_24/': 'res_net/res_block_7/conv_0/', \
'resnet_model/conv2d_25/': 'res_net/res_block_7/conv_1/', \
'resnet_model/conv2d_26/': 'res_net/res_block_7/conv_2/', \
'resnet_model/conv2d_27/': 'res_net/res_block_7/conv_3/', \
'resnet_model/conv2d_28/': 'res_net/res_block_8/conv_1/', \
'resnet_model/conv2d_29/': 'res_net/res_block_8/conv_2/', \
'resnet_model/conv2d_30/': 'res_net/res_block_8/conv_3/', \
'resnet_model/conv2d_31/': 'res_net/res_block_9/conv_1/', \
'resnet_model/conv2d_32/': 'res_net/res_block_9/conv_2/', \
'resnet_model/conv2d_33/': 'res_net/res_block_9/conv_3/', \
'resnet_model/conv2d_34/': 'res_net/res_block_10/conv_1/', \
'resnet_model/conv2d_35/': 'res_net/res_block_10/conv_2/', \
'resnet_model/conv2d_36/': 'res_net/res_block_10/conv_3/', \
'resnet_model/conv2d_37/': 'res_net/res_block_11/conv_1/', \
'resnet_model/conv2d_38/': 'res_net/res_block_11/conv_2/', \
'resnet_model/conv2d_39/': 'res_net/res_block_11/conv_3/', \
'resnet_model/conv2d_40/': 'res_net/res_block_12/conv_1/', \
'resnet_model/conv2d_41/': 'res_net/res_block_12/conv_2/', \
'resnet_model/conv2d_42/': 'res_net/res_block_12/conv_3/', \
'resnet_model/conv2d_43/': 'roi_head/time_distributed/res_block/conv_0/', \
'resnet_model/conv2d_44/': 'roi_head/time_distributed/res_block/conv_1/', \
'resnet_model/conv2d_45/': 'roi_head/time_distributed/res_block/conv_2/', \
'resnet_model/conv2d_46/': 'roi_head/time_distributed/res_block/conv_3/', \
'resnet_model/conv2d_47/': 'roi_head/time_distributed_1/res_block/conv_1/', \
'resnet_model/conv2d_48/': 'roi_head/time_distributed_1/res_block/conv_2/', \
'resnet_model/conv2d_49/': 'roi_head/time_distributed_1/res_block/conv_3/', \
'resnet_model/conv2d_50/': 'roi_head/time_distributed_2/res_block/conv_1/', \
'resnet_model/conv2d_51/': 'roi_head/time_distributed_2/res_block/conv_2/', \
'resnet_model/conv2d_52/': 'roi_head/time_distributed_2/res_block/conv_3/', \
}
