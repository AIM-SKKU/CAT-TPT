import torch
import clip
import numpy as np
import sys
sys.path.append("../")
from utils.non_nv import encode_text_with_learnt_tokens
from utils.deep_set_clf import D as deep_set
import torch.nn.functional as F
from dataclasses import dataclass
from simple_parsing import ArgumentParser
import time
import os
from utils.nv import TextVisualMap, TextVisualMapAbl, natural_prompt_multi,imagenet_templates
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import shutil
from diffusers import AutoPipelineForImage2Image
import json

from data.fewshot_generation import build_fewshot_dataset, fewshot_datasets,ID_to_DIRNAME,path_dict

# emb_dim: int = 512
#　VIT-L-14
emb_dim: int = 768
num_tokens = 77

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
    "i":'./attribute_description/attribute_imagenet.json',
    "a": './attribute_description/attribute_imagenet.json',
    "k": './attribute_description/attribute_imagenet.json',
    "r": './attribute_description/attribute_imagenet.json',
    "v":'./attribute_description/attribute_imagenet.json',
    "c":'./attribute_description/attribute_imagenet.json'

}
@dataclass
class HParams:
    """Set of options for the training of a Model."""

    no_of_new_tokens: int = 1
    is_prompt_multi: bool = True #Use prompt augmentations:    {"This is a photo of a *", "This photo contains a *",..,} etc.
    set_size: int = 20
    deep_set_d_dim: int = 4096
    dropout: float = 0.25
    natural_prompt_default: str = "This is a photo of a *"
    pooling_type: str = "mean"

    between_text = ', '
    before_text = ""
    after_text = ''
    category_name_inclusion = 'prepend'


    set_id = 'A'
    data_dir = "./datasets/TPT/domain_shift_datastes/imagenet-a/imagenet-a/"
    save_dir = "./datasets/TPT/generated_images/"
    checkpoint = "./checkpoints/set_model_k16.pt"


    # #### Imagenet-C
    # ### corruption = gaussian_noise/shot_noise/impulse_noise/defocus_blur/glass_blur/motion_blur/zoom_blur/frost/snow/fog/brightness/contrast/elastic_transform/pixelate/jpeg_compression
    # set_id = 'C'
    # corruption = 'jpeg_compression'
    # level=5
    # data_dir = "./datasets/TPT/imagenet-c/"
    # save_dir = "./datasets/TPT/generated_images/"
    # checkpoint = "./checkpoints/set_model_k16.pt"


class Dataset_ImageNetR(Dataset):
    def __init__(self, root, transform, encode_image,device):
        self.root = root
        self.transform = transform
        self.folders = os.listdir(self.root)
        self.folders.sort()
        self.images = []
        self.device = device
        self.encode_image = encode_image
        self.tform = transforms.Compose([transforms.ToTensor(), transforms.Resize((500, 500),),])

        ### randomly choose 5/all images for each folder
        for folder in self.folders:
            if not os.path.isdir(os.path.join(self.root, folder)):
                continue
            class_images = os.listdir(os.path.join(self.root, folder))
            class_images = list(map(lambda x: os.path.join(folder, x), class_images))
            random.shuffle(class_images)
            class_image = class_images[0:5]
            self.images = self.images + class_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx]))).unsqueeze(0).to(self.device)
        image_embeddings = self.encode_image(image)

        init_image = self.tform(Image.open(os.path.join(self.root, self.images[idx])).convert('RGB'))
        return self.images[idx], image_embeddings, init_image


def Generate_attribution_descriptions(args):
    if args.hparams.is_prompt_multi:
        chosen_prompt = np.random.randint(len(natural_prompt_multi))
        word_to_add = natural_prompt_multi[chosen_prompt]
    else:
        word_to_add = args.hparams.natural_prompt_default

    json_path = descriptor_dict[args.hparams.set_id.lower()]
    with open(json_path, 'r') as fp:
        att_descriptions = json.load(fp)

    # attributes Prompt
    attribute_prompt = []
    for key in att_descriptions:
        attribute_prompt.extend(att_descriptions[key])

    chosen_prompt = random.sample(attribute_prompt, args.hparams.set_size)
    prompt_des = [args.hparams.before_text + word_to_add + args.hparams.between_text + "which is " + chosen_prompt[i] + args.hparams.after_text for i in range(args.hparams.set_size)]

    ids = args.hparams.set_size//4
    return prompt_des[0:ids], prompt_des[ids:ids*2], prompt_des[ids*2:ids*3], prompt_des[ids*3::]


def main():

    parser = ArgumentParser()
    parser.add_arguments(HParams, dest="hparams")
    args, unknown = parser.parse_known_args()
    print("")
    print("args",args)

    deep_set_d_dim = args.hparams.deep_set_d_dim
    dropout = args.hparams.dropout

    #Deep set model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model,_,preprocess= clip.load("ViT-L/14", device=device)
    #Add personalized text encoder method to CLIP
    funcType = type(model.encode_text)
    model.encode_text_with_learnt_tokens = funcType(encode_text_with_learnt_tokens, model)

    model.eval()

    #Deep set model
    set_model = deep_set(deep_set_d_dim, x_dim=emb_dim, out_dim=args.hparams.no_of_new_tokens*emb_dim, pool=args.hparams.pooling_type, dropout=dropout)
    set_model = set_model.to(device)
    checkpoint = args.hparams.checkpoint
    set_model.load_state_dict(torch.load(checkpoint, map_location=device))
    set_model.eval()

    ## data dir
    data_dir = args.hparams.data_dir

    pipe = AutoPipelineForImage2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.enable_model_cpu_offload()

    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    data_augment_preprocess = Compose([
        Resize(300, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    # dataset = Dataset_ImageNetR(data_dir, data_augment_preprocess, model.encode_image)

    if args.hparams.set_id in fewshot_datasets:
        data_root = os.path.join(data_dir, ID_to_DIRNAME[args.hparams.set_id.lower()])
        dataset = build_fewshot_dataset(args.hparams.set_id, data_root, data_augment_preprocess,  encode_image=model.encode_image, mode='test', device=device)
        if args.hparams.set_id == 'Aircraft':
            data_dir = os.path.join(data_root, 'images')
        else:
            path_suffix, _ = path_dict[args.hparams.set_id.lower()]
            data_dir = os.path.join(data_root, path_suffix)
        save_dir = os.path.join(args.hparams.save_dir, ID_to_DIRNAME[args.hparams.set_id.lower()])
        os.makedirs(save_dir, exist_ok=True)
    elif args.hparams.set_id =='C':
        data_dir = os.path.join(data_dir, args.hparams.corruption, str(args.hparams.level))
        dataset = Dataset_ImageNetR(data_dir, data_augment_preprocess, model.encode_image, device=device)
        save_dir = os.path.join(args.hparams.save_dir, ID_to_DIRNAME[args.hparams.set_id], args.hparams.corruption, str(args.hparams.level))
        os.makedirs(save_dir, exist_ok=True)
    else:
        dataset = Dataset_ImageNetR(data_dir, data_augment_preprocess, model.encode_image, device=device)
        save_dir = os.path.join(args.hparams.save_dir, ID_to_DIRNAME[args.hparams.set_id])
        os.makedirs(save_dir, exist_ok=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False)



    print("############## Starting Generation ##############")
    start_time = time.time()
    ### use general attribute
    no_of_new_tokens = args.hparams.no_of_new_tokens
    with torch.no_grad():
        for count, (image_locations, image_embeddings, init_image) in enumerate(dataloader):
            print(f'{count} / {len(dataloader)}, {image_locations[0]}.')

            for image_lo in image_locations:
                os.makedirs(os.path.join(save_dir, os.path.dirname(image_lo)), exist_ok=True)
                source_path = os.path.join(data_dir, image_lo)
                dist_path = os.path.join(save_dir, image_lo)

                if not os.path.exists(dist_path):
                    shutil.copyfile(source_path, dist_path)
                    with open(os.path.join(save_dir, 'selected_data_list.txt'), 'a+') as f:
                        f.write(dist_path + '\n')


            for index in range(len(init_image)):
                # without classname using "*"
                label_descriptors = Generate_attribution_descriptions(args)
                with torch.no_grad():
                    image_features = F.normalize(image_embeddings[index].squeeze(1).float(), dim=-1)
                    out_features = set_model(image_features)
                    estimated_tokens = out_features.reshape((out_features.shape[0], no_of_new_tokens, emb_dim))
                    # estimated_tokens = out_features.reshape((no_of_new_tokens, emb_dim))
                    base_token = None
                    asterix_token = clip.tokenize(["*"]).to(device)[0][1]


                    idx = 0
                    for descriptors in label_descriptors:
                        # print(descriptors)
                        natural_prompt_asterix_embeddings = torch.tensor(()).to(device)
                        estimated_tokens = estimated_tokens.repeat(len(descriptors), 1, 1)
                        text = clip.tokenize(descriptors).to(device)
                        natural_prompt_embeddings = model.encode_text_with_learnt_tokens(text, asterix_token, estimated_tokens, base_token, is_token_emb=True)
                        natural_prompt_asterix_embeddings = torch.cat((natural_prompt_asterix_embeddings, natural_prompt_embeddings), dim=0)
                        generated_images = pipe(prompt_embeds=natural_prompt_asterix_embeddings, image=init_image[index]).images

                        for i in range(len(generated_images)):
                            format = image_locations[index].split('.')[-1]
                            image_format = '.' + format
                            generated_images[i].save(os.path.join(save_dir, image_locations[index].split(image_format)[0] + '_' + str(idx+i) + '.jpg'))
                        idx = idx+i+1
    # measure elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()


