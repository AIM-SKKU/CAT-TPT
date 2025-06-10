#!/bin/bash

data_root='/path/to/your/data/root'
coop_weight='/path/to/pretrained/coop/weight.pth'
testsets=$1
arch=ViT-B/16
ctx_init=a_photo_of_a

python ./cattpt_classiification.py --tva_root ${data_root} --test_sets ${testsets} \
-a ${arch} --gpu 0 \
--tpt --ctx_init ${ctx_init} --load ${coop_weight}
