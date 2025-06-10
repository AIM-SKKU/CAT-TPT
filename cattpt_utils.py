import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask, \
    imagenet_r_wnids, all_wnids

# # imagenet_r
imagenet_r_fold = ['n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677']



fewshot_datasets = ['DTD', 'Flower102', 'Food101', 'Cars', 'SUN397',
                    'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat']
ID_to_DIRNAME={
    'I': 'imageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sk',
    'R': 'imagenet-r',
    'V': 'imagenetv2',
    'flower102': 'cross_datasets_1k_gpt_att/oxford_flowers',
    'dtd': 'cross_datasets_1k_gpt_att/dtd',
    'pets': 'cross_datasets_1k_gpt_att/oxford_pets',
    'cars': 'cross_datasets_1k_gpt_att/stanford_cars',
    'ucf101': 'cross_datasets_1k_gpt_att/ucf101',
    'caltech101': 'cross_datasets_1k_gpt_att/caltech-101',
    'food101': 'cross_datasets_1k_gpt_att/food-101',
    'sun397': 'cross_datasets_1k_gpt_att/sun397',
    'aircraft': 'cross_datasets_1k_gpt_att/fgvc_aircraft',
    'eurosat': 'cross_datasets_1k_gpt_att/eurosat',
    'C': 'imagenet-c'
}
path_dict = {
    "flower102": ["jpg", "data/data_splits/split_zhou_OxfordFlowers.json"],
    "food101": ["images", "data/data_splits/split_zhou_Food101.json"],
    "dtd": ["images", "data/data_splits/split_zhou_DescribableTextures.json"],
    "pets": ["images", "data/data_splits/split_zhou_OxfordPets.json"],
    "sun397": ["SUN397", "data/data_splits/split_zhou_SUN397.json"],
    "caltech101": ["101_ObjectCategories", "data/data_splits/split_zhou_Caltech101.json"],
    "ucf101": ["UCF-101-midframes", "data/data_splits/split_zhou_UCF101.json"],
    "cars": ["", "data/data_splits/split_zhou_StanfordCars.json"],
    "eurosat": ["2750", "data/data_splits/split_zhou_EuroSAT.json"]
}

# original view_num=63, generated images number by diffusion model
class tvaData(Dataset):
    def __init__(self, data_root, img_list, label_list, trainsform, augmode='cattpt', diffu_ratio=0.5, view_num=20):
        self.data_root = data_root
        self.img_list = img_list
        self.label_list = label_list
        self.trainsform = trainsform
        self.augmode = augmode
        self.diffu_ratio = diffu_ratio
        self.view_num = view_num

    def __getitem__(self, item):
        if self.augmode == 'tpt':
            img = Image.open(os.path.join(self.data_root,self.img_list[item])).convert('RGB')
            imgs = self.trainsform[0](img)
            return imgs, self.label_list[item]
        elif self.augmode == 'cattpt':
            # imgs + diffu 20 + aug 21
            img_path = os.path.join(self.data_root, self.img_list[item])
            img = Image.open(img_path).convert('RGB')
            imgs = self.trainsform[0](img)  # imgs[ori，augx21(self.trainsform[0].n_view)]

            diffu_img = []  # 20 diffu imgs
            for i in range(self.view_num):
                image_format = '.' + ((self.img_list[item]).split('.'))[-1]
                leng = len(image_format)
                # img_i = Image.open(os.path.join(img_path[:-leng] + '_' + str(i) + image_format)).convert('RGB')
                img_i = Image.open(os.path.join(img_path[:-leng] + '_' + str(i) + '.jpg')).convert('RGB')
                img_i = self.trainsform[1](img_i)
                diffu_img.append(img_i)

            return imgs + diffu_img, self.label_list[item]
            # [ori, tpt, diffu]
            # [ori, tpt, diffu]
        else:
            print('self.augmode error! tpt or cattpt')
            exit()

    def __len__(self):
        return len(self.img_list)


def get_data_loader(set_id, data_transform, args=None):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    data_transform_TVA = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        normalize,
    ])

    if set_id == 'A':
        imagenet_dataset_fold = [all_wnids[k] for k in imagenet_a_mask]
    elif set_id== 'R':
        imagenet_dataset_fold = imagenet_r_fold
    elif set_id == 'K':
        imagenet_dataset_fold = all_wnids
    elif set_id == 'V':
        imagenet_dataset_fold = [str(k) for k in imagenet_v_mask]
    elif set_id == 'I':
        imagenet_dataset_fold = all_wnids
    elif set_id == 'C':
        imagenet_dataset_fold = all_wnids
    elif set_id in fewshot_datasets:
        if set_id == 'Aircraft':
            # path = os.path.join(args.hparams.data_dir, ID_to_DIRNAME[args.hparams.set_id.lower()])
            path = "/home/datasets/TPT/cross_datasets/fgvc_aircraft/"

            with open(os.path.join(path, "variants.txt"), 'r') as fp:
                cname = [l.replace("\n", "") for l in fp.readlines()]

            with open(os.path.join(path, 'images_variant_{:s}.txt'.format(args.dataset_mode)), 'r') as fp:
                lines = [s.replace("\n", "") for s in fp.readlines()]
                label_dict = {}
                for l in lines:
                    ls = l.split(" ")
                    img = ls[0]
                    label = " ".join(ls[1:])
                    image = "{}.jpg".format(img)
                    label_dict[image] = cname.index(label)

        else:
            path_suffix, json_path = path_dict[set_id.lower()]
            with open(json_path) as fp:
                splits = json.load(fp)
                samples = splits[args.dataset_mode]
                label_dict={}
                for s in samples:
                    label_dict[s[0]] = s[1]



    img_list = []
    label_list = []

    if set_id in fewshot_datasets:
        data_root = os.path.join(args.tva_root, ID_to_DIRNAME[set_id.lower()])
        with open(os.path.join(data_root, 'selected_data_list.txt'), 'r') as f:
            for line in f.readlines():
                read_img = line.strip()
                img_list.append(read_img)
                read_img = read_img.replace(data_root+"/", "")
                label_list.append(label_dict[read_img])
    else:
        if set_id =='C':
            data_root = os.path.join(args.tva_root, ID_to_DIRNAME[set_id], args.corruption, args.level)
        else:
            data_root = os.path.join(args.tva_root, ID_to_DIRNAME[set_id])
        with open(os.path.join(data_root, 'selected_data_list.txt'), 'r') as f:
            for line in f.readlines():
                read_img = line.strip()
                img_list.append(read_img)
                label_list.append(imagenet_dataset_fold.index(read_img.split('/')[-2]))

    tpt_dataset = tvaData(data_root=data_root, img_list=img_list, label_list=label_list, \
                            trainsform=[data_transform, data_transform_TVA], augmode=args.aug_mode,
                            view_num=args.batch_size - 1)

    return tpt_dataset


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def select_confident_samples_cosine(logits, selection_cosine, selection_selfentro):
    cosine_distan = torch.nn.functional.cosine_similarity(logits[0].unsqueeze(0), logits[1:], dim=1)  # [N-1]
    idx_cosine = torch.topk(cosine_distan, k=int(cosine_distan.size(0) * selection_cosine), largest=True).indices + 1

    logits_cos = logits[idx_cosine]  # Choose first selection_cosine aug+diff
    logits = torch.cat((logits[0, :].unsqueeze(0), logits_cos), dim=0)  # cat(original image logits, choose aug+diff logits)

    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    # idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * selection_selfentro)]
    idx = torch.topk(batch_entropy, k=int(batch_entropy.size(0) * selection_selfentro), largest=False).indices
    selected_idx = torch.cat([torch.tensor([0], device=logits.device), idx_cosine])[idx]

    return logits[idx], selected_idx

def test_time_tuning_tpt(model, inputs, optimizer, scaler, args):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)

    selected_idx = None
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs)

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples_cosine(output, args.selection_cosine, args.selection_selfentro)

            loss = avg_entropy(output)

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

    if args.cocoop:
        return pgen_ctx

    return