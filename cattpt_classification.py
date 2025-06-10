import argparse
from tqdm import tqdm
import time

from copy import deepcopy

from PIL import Image

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import random

import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models
from clip.custom_clip_cat import get_coop
from clip.cocoop_cat import get_cocoop

from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

import cattpt_utils
import json


import wandb
from datetime import datetime

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))



descriptor_dict = {
    "flower102": './attribute_description/attribute_flower.json',
    "food101": './attribute_description/attribute_food101.json',
    "dtd": './attribute_description/attribute_dtd.json',
    "pets": './attribute_description/attribute_pets.json',
    "sun397": './attribute_description/attribute_sun.json',
    "caltech101":'./attribute_description/attribute_caltech.json',
    "ucf101": './attribute_description/attribute_ucf.json',
    "cars": './attribute_description/attribute_car.json',
    "eurosat": './attribute_description/attribute_eurosat.json',
    "aircraft": './attribute_description/attribute_aircraft.json',
    "imagenet": './attribute_description/attribute_imagenet.json'
}



def Generate_attribution_descriptions(args, classname,attribute_path):
    set_size = args.set_size
    word_to_add = "*"


    with open(attribute_path, 'r') as fp:
        att_descriptions = json.load(fp)

    # attributes Prompt
    attribute_prompt = []
    for key in att_descriptions:
        if len(att_descriptions[key]) > set_size:
            chosen_attributes = random.sample(att_descriptions[key], args.set_size)
        else:
            chosen_attributes = att_descriptions[key]
        attribute_prompt.extend(chosen_attributes)

    att_descriptions = [
        args.before_text + word_to_add + args.between_text + "which is " + chosen_prompt + args.after_text for
        chosen_prompt in random.sample(attribute_prompt, set_size)]


    attribute_description = {}
    for name in classname:
        attribute_description[name] = [attribute.replace("*", name) for attribute in att_descriptions]
    return attribute_description


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner.ctx.copy_(pretrained_ctx)
                model.prompt_learner.ctx_init_state = pretrained_ctx
        model_state = None

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)

    print("=> Model created: visual backbone {}".format(args.arch))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        if args.load is not None:
            optimizer = None
            optim_state = None
        else:
            trainable_param = model.prompt_generator.parameters()
            optimizer = torch.optim.AdamW(trainable_param, args.lr)
            optim_state = deepcopy(optimizer.state_dict())
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}

    for set_id in datasets:

        if set_id in fewshot_datasets:
            classnames = eval("{}_classes".format(set_id.lower()))
            attbutr_dir = set_id
        else:
            classnames = imagenet_classes
            attbutr_dir = 'imagenet'

        # # # attribute driven
        attribute_path = descriptor_dict[attbutr_dir.lower()]
        gpt_descriptions = Generate_attribution_descriptions(args, classnames, attribute_path)


        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])

            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size, augmix=len(set_id) > 1)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        print("evaluating: {}".format(set_id))
        # # reset the model
        # # Reset classnames of custom CLIP model
        if len(set_id) > 1:
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all

        class_des_label = [i for i in range(len(classnames)) for _ in range(args.set_size)]
        class_des_label = torch.tensor(class_des_label).cuda(args.gpu)

        classnames_des = []
        for i in range(len(classnames)):
            classnames_i = classnames[i]
            for des in gpt_descriptions[classnames_i]:
                classnames_des.append(des)

        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames_des, class_des_label, len(classnames), args.arch)  # 25/04/07
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        else:
            model.reset_classnames(classnames_des, class_des_label, len(classnames), args.arch)# 25/04/07


        val_dataset = cattpt_utils.get_data_loader(set_id, data_transform, args)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batchsize, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        results[set_id] = test_time_adapt_eval(set_id, val_loader, model, model_state, optimizer, optim_state, scaler, args)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}/ @3 {}/ @10 {}".format(set_id, results[set_id][0],
                                                                                results[set_id][1], results[set_id][2],
                                                                                results[set_id][3]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))



    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))


    print("\t\t [set_id] \t\t Top-1 acc.")
    for id in results.keys():
        print("{}".format(id), end=" ")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id].item()), end=" ")
    print("\n")


def test_time_adapt_eval(set_id,val_loader, model, model_state, optimizer, optim_state, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)


    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop:  # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()


    date = datetime.now().strftime("%b%d_%H-%M-%S")
    group_name = f"{args.arch}_{set_id}_{date}"
    run_name = f"CAT-TPT_{set_id}_{args.arch}"
    run = wandb.init(project="CAT-TPT-IJCV", config=args, group=group_name, name=run_name)

    print("############## Starting ##############")
    start_time = time.time()
    batch_end = time.time()
    for i, (images, target) in enumerate(tqdm(val_loader, desc='Processed test images: ')):
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)
        if args.tpt:
            images = torch.cat(images, dim=0)

        # reset the tunable prompt to its initial state
        if not args.cocoop:  # no need to reset cocoop because it's fixed
            if args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()
            optimizer.load_state_dict(optim_state)
            cattpt_utils.test_time_tuning_tpt(model, images, optimizer, scaler, args)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            pgen_ctx = cattpt_utils.test_time_tuning_tpt(model, (image_feature, pgen_ctx), optimizer, scaler, args)


        # The actual inference goes here
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    output = model(image)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - batch_end)

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

        wandb.log({"Averaged test accuracy": top1.avg}, commit=True)

    # measure elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    wandb.log({"Elapsed time": f"{elapsed_time:.2f} seconds"}, commit=True)
    wandb.log({f"Acc": top1.avg})
    run.finish()
    progress.display_summary()

    return top1.avg


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='CAT-TPT')
    # parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    # parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    # parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    # parser.add_argument('-b', '--batch-size', default=21, type=int, metavar='N')
    # parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,  metavar='LR', help='initial learning rate', dest='lr')
    # parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency (default: 10)')
    # parser.add_argument('--gpu', default=6, type=int, help='GPU id to use.')
    # parser.add_argument('--tpt', action='store_true', default=True, help='run test-time prompt tuning')
    # parser.add_argument('--tta_steps', default=4, type=int, help='test-time-adapt steps')
    # parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    # parser.add_argument('--ctx_init', default='a_photo_of_a', type=str, help='init tunable prompts')
    # parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--aug_mode', default='cattpt', type=str)
    # parser.add_argument('--selection_cosine', default=0.8, type=float, help='confidence selection percentile')
    # parser.add_argument('--selection_selfentro', default=0.3, type=float, help='confidence selection percentile')
    #
    # parser.add_argument('--tva_root', type=str, help='SD generate img root', default='./datasets/generated_images)
    # parser.add_argument('--test_sets', type=str, default='R', help='test dataset (multiple datasets split by slash A/R/V/K/I)')
    # parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    # parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    #
    # ### Attributes description
    # parser.add_argument("--attribute_descriptor", type=str, default='./attribute_description/attribute_imagenet.json')
    # parser.add_argument("--category_name_inclusion", type=str, default='prepend')  # 'append' 'prepend'
    # parser.add_argument("--before_text", type=str, default="")
    # parser.add_argument("--label_before_text", type=str, default="")
    # parser.add_argument("--between_text", type=str, default=', ')
    # parser.add_argument("--after_text", type=str, default='')
    # parser.add_argument("--label_after_text", type=str, default='')
    # parser.add_argument("--apply_descriptor_modification", type=str, default=True)
    # parser.add_argument("--unmodify", type=str, default=True)
    # parser.add_argument("--set_size", type=int, default=4, help="description set size")



    ###　实验
    parser = argparse.ArgumentParser(description='CAT-TPT')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=21, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,  metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--tpt', action='store_true', default=True, help='run test-time prompt tuning')
    parser.add_argument('--tta_steps', default=4, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default='a_photo_of_a', type=str, help='init tunable prompts')
    parser.add_argument('--aug_mode', default='cattpt')
    parser.add_argument('--seed', type=int, default=0)
    # # # # # ###　　CAT-TPT　###
    # parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    # parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    #
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    # # parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')

    # # ###　　cat-tpt &　coop　###
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")

    # parser.add_argument('--load', default='/home/datasets/TPT/coop_imagenet_16shots_4backbones/to_gdrive/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed3/prompt_learner/model.pth.tar-50', type=str, help='path to a pre-trained coop/cocoop')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    # parser.add_argument('--load', default='/home/datasets/TPT/coop_imagenet_16shots_4backbones/to_gdrive/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed3/prompt_learner/model.pth.tar-50', type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')


    # ###　　cat-tpt　＆　CoCoop　####
    # parser.add_argument('--cocoop', action='store_true', default=True, help="use cocoop's output as prompt initialization")
    # # parser.add_argument('--load', default='/home/datasets/TPT/Cocoop_imagenet_16shots/xd_rn50_c4_ep10_batch1_ctxv1_seed3/prompt_learner/model.pth.tar-10', type=str, help='path to a pre-trained coop/cocoop')
    # # parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    # parser.add_argument('--load', default='/root/datasets/TPT/Cocoop_imagenet_16shots/xd_vit_b16_c4_ep10_batch1_ctxv1_seed3/prompt_learner/model.pth.tar-10', type=str, help='path to a pre-trained coop/cocoop')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')


    parser.add_argument('--tva_root', type=str, help='SD generate img root', default='/home/datasets/TPT/generated_images/palavra-attributr-stdiff-16shots/')
    parser.add_argument('--test_sets', type=str, default='A', help='test dataset (multiple datasets split by slash A/R/V/K/I)')
    parser.add_argument('--gpu', default=5, type=int, help='GPU id to use.')
    parser.add_argument('--selection_cosine', default=0.8, type=float, help='confidence selection percentile')
    parser.add_argument('--selection_selfentro', default=0.3, type=float, help='confidence selection percentile')



    ### Attributes description
    parser.add_argument("--attribute_descriptor", type=str, default='./attribute_description/attribute_imagenet.json')
    parser.add_argument("--category_name_inclusion", type=str, default='prepend')  # 'append' 'prepend'
    parser.add_argument("--before_text", type=str, default="")
    parser.add_argument("--label_before_text", type=str, default="")
    parser.add_argument("--between_text", type=str, default=', ')
    parser.add_argument("--after_text", type=str, default='')
    parser.add_argument("--label_after_text", type=str, default='')
    parser.add_argument("--apply_descriptor_modification", type=str, default=True)
    parser.add_argument("--unmodify", type=str, default=True)
    parser.add_argument("--set_size", type=int, default=4, help="description set size")

    main()