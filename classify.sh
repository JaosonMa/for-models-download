cd ..
CUDA_VISIBLE_DEVICES=0 python3 classify.py \
  --model_file=./data/flowers_2/models/all/infer/my_freeze.pb \
  --image_file=./bat-file/2_1_25_3_1.jpg \
  --label_file=./data/flowers_2/labels.txt
