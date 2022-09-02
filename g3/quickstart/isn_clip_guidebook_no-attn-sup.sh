config=image_clip_clues
name=resnet50_image_clip_clues_no-attn-sup
text_labels_file=""

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m classification.train.train_classification \
    --config ./config/resnet50_${config}.yml \
    model_params.name=${name} \
    model_params.text_labels_file=${text_labels_file} \
    seed=132 \

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m classification.train.train_classification \
    --config ./config/resnet50_${config}.yml \
    model_params.name=${name} \
    model_params.text_labels_file=${text_labels_file} \
    seed=131 \

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m classification.train.train_classification \
    --config ./config/resnet50_${config}.yml \
    model_params.name=${name} \
    model_params.text_labels_file=${text_labels_file} \
    seed=130 \

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m classification.train.train_classification \
    --config ./config/resnet50_${config}.yml \
    model_params.name=${name} \
    model_params.text_labels_file=${text_labels_file} \
    seed=129 \

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m classification.train.train_classification \
	--config ./config/resnet50_${config}.yml \
	model_params.name=${name} \
    model_params.text_labels_file=${text_labels_file} \
	seed=128 \