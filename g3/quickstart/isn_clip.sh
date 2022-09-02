config=image_and_clip
name=resnet50_image_and_clip

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m classification.train.train_classification \
    --config ./config/resnet50_${config}.yml \
    model_params.name=${name} \
    seed=132 \

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m classification.train.train_classification \
    --config ./config/resnet50_${config}.yml \
    model_params.name=${name} \
    seed=131 \

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m classification.train.train_classification \
    --config ./config/resnet50_${config}.yml \
    model_params.name=${name} \
    seed=130 \

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m classification.train.train_classification \
    --config ./config/resnet50_${config}.yml \
    model_params.name=${name} \
    seed=129 \

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m classification.train.train_classification \
	--config ./config/resnet50_${config}.yml \
	model_params.name=${name} \
	seed=128 \