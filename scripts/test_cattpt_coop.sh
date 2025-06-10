#!/bin/bash

#data_root='/path/to/your/data/root'
#coop_weight='/path/to/pretrained/coop/weight.pth'

data_root='/home/datasets/TPT/generated_images/palavra-attributr-stdiff-16shots/'
coop_weight='/home/datasets/TPT/coop_imagenet_16shots_4backbones/to_gdrive/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed3/prompt_learner/model.pth.tar-50'
testsets=A/R/V/K/I
arch=ViT-B/16
bs=21
ctx_init=a_photo_of_a
tta_steps=4

###
description=attribute
aug_mode=cattpt
set_size=4
s1=0.8
s2=0.3


python ./cattpt_classification.py --tva_root ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 2 \
--tpt --ctx_init ${ctx_init} --load ${coop_weight} \
--aug_mode ${aug_mode} --description ${description} --set_size ${set_size} \
--selection_cosine ${s1} --selection_selfentro ${s2} --tta_steps ${tta_steps}












