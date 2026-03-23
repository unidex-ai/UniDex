if [ ! -d "pretrained" ]; then
    mkdir pretrained
fi

if [ ! -d "pretrained/uni3d" ]; then
    mkdir pretrained/uni3d
fi

if [ -d "$HOME/.cache/modelscope/hub/models/BAAI/Uni3D/" ]; then
    ORIGINAL_DIR=$HOME/.cache/modelscope/hub/models/BAAI/Uni3D/
elif [ -d "$HOME/.cache/huggingface/hub/models--BAAI--Uni3D" ]; then
    ORIGINAL_DIR=$HOME/.cache/huggingface/hub/models--BAAI--Uni3D/
else
    echo "Model directory not found."
    exit 1
fi

cp $ORIGINAL_DIR/modelzoo/uni3d-ti/model.pt pretrained/uni3d/model-ti.pt
cp $ORIGINAL_DIR/modelzoo/uni3d-s/model.pt pretrained/uni3d/model-s.pt
cp $ORIGINAL_DIR/modelzoo/uni3d-g/model.pt pretrained/uni3d/model-g.pt
cp $ORIGINAL_DIR/modelzoo/uni3d-l/model.pt pretrained/uni3d/model-l.pt
cp $ORIGINAL_DIR/modelzoo/uni3d-b/model.pt pretrained/uni3d/model-b.pt

python -c "
import torch
import os

for model_size in ['ti', 's', 'g', 'l', 'b']:
    model_path = f'pretrained/uni3d/model-{model_size}.pt'
    if os.path.exists(model_path):
        model = torch.load(model_path)
        model = model['module']
        model = {k.replace('point_encoder.', ''): v for k,v in model.items() if 'point_encoder.' in k}
        torch.save(model, model_path)
    else:
        print(f'Model {model_size} not found at {model_path}.')
"