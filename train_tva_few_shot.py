from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import wandb
import os
import clip
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.deep_set_clf import D as deep_set
from dataclasses import dataclass
from utils.non_nv import encode_text_with_learnt_tokens
from simple_parsing import ArgumentParser
import torch.nn.functional as F

import time
from torch.nn import CosineSimilarity, MSELoss
from torch.utils.data import DataLoader
from data.fewshot_generation import CustomTextDataset

natural_prompt_multi = [
    "This is a photo of a * ",
    "This photo contains a * ",
    "A photo of a * ",
    "This is an illustrations of a * ",
    "This illustrations contains a * ",
    "An illustrations of a * ",
    "This is a sketch of a * ",
    "This sketch contains a * ",
    "A sketch of a * ",
    "This is a diagram of a * ",
    "This diagram contains a * ",
    "A diagram of a * ",
    "A * ",
    "We see a * ",
    ]

@dataclass
class HParams:
    """Set of options for the training of a Model."""
    lr: float = 1e-4 #learning rate
    lr_text_vis: float = 1e-4 #learning rate text_vis
    epochs: int = 1000
    batch_size: int = 64 #input batch size for training
    no_of_new_tokens: int = 1
    is_learn_token: bool = True

    is_prompt_multi: bool = True #Use prompt augmentations:    {"This is a photo of a *", "This photo contains a *",..,} etc.
    is_save_models: bool = True
    is_learn_prefix: bool = False  #Use also coarse grained textual prompt
    is_learn_prefix_also_image: bool = False #dependant on is_learn_prefix
    is_gt_object_loss: bool = True  #Use gt (l2) loss
    coeff_gt_object_loss: float = 1 #Coefficent for the gt (l2) loss
    coeff_cycle_loss: float = 1 #Coefficent for the cycle loss
    contrastive_temp: float = 0.25
    shot_size: int = 16
    deep_set_d_dim: int = 4096
    dropout: float = 0.5
    project_name: str = "FSL_clip"
    loss_str: str = "contrastive" #cycle loss type
    save_model_name: str = "first_save"
    natural_prompt_default: str = "This is a photo of a *"

    pooling_type: str = "mean" #deepset pooling

    #Set transfomer parameters below, currently deprecated
    is_set_transformer: bool = False
    st_num_outputs: int = 1
    st_num_inds: int = 32
    st_dim_hidden: int = 128
    st_num_heads: int = 4
    st_ln: bool = False
    is_multi_set_size: bool = False #Variable set input size, currently deprecated


    ##### using ViT-L-14 backbone for diffusion models
    lable_img_dict_path: str = "./ImageNet/lable_img_dict_k16.npz"
    visual_features_path: str = "./ImageNet/visual_features_k16.npz"
    emb_dim = 768
    save_path = "./checkpoints/"

def save_trained_model(save_path, model_name, args, trained_model, timestr):
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path , "%s_%s"%(model_name,timestr) )
    args_path = os.path.join(save_path, "%s_%s"%("args",timestr) )
    np.save(args_path,args.hparams)
    torch.save(trained_model.state_dict(), model_path)



def get_clip_text_acc(natural_prompt_embeddings, natural_prompt_asterix_embeddings):
    #Calculate accuracy
    pred = torch.argmax(torch.mm(natural_prompt_embeddings.float(), natural_prompt_asterix_embeddings.t().contiguous().float()), dim = 1)
    gt = np.arange(len(pred))
    correct_num = (len(np.where(pred.cpu().numpy() == gt)[0]))
    print("acc %d / %d"%(correct_num, len(pred)))
    return correct_num, len(pred)

def run_epoch(args, epoch, dataloader,  optimizer, scheduler, model, set_model, device, img_features_dict, is_train):
    #run a single training epoch

    chosen_prompt = np.random.randint(len(natural_prompt_multi))
    natural_prompt = natural_prompt_multi[chosen_prompt]

    asterix_token = clip.tokenize(["*"]).to(device)[0][1]
    no_of_new_tokens = args.hparams.no_of_new_tokens
    criterion = contrastive_loss

    for batch_num, sample in enumerate(dataloader):
        ####### Image input
        visual_features_list = []
        target_labels, chosen_img_id = sample

        for i, chosen_img_id_inst in enumerate(chosen_img_id):
            img_features_inst = [img_features_dict[chosen_img_id_inst_i] for chosen_img_id_inst_i in chosen_img_id_inst]
            img_features_inst = torch.from_numpy(np.asarray(img_features_inst)).cuda()
            img_features_inst = F.normalize(img_features_inst, dim=-1).squeeze(dim=1)
            visual_features_list.append(img_features_inst[:,0,:])
        visual_features = torch.stack(visual_features_list, 1) #[200,5,512]

        total_loss, total_num = 0.0, 0
        optimizer.zero_grad()

        gt_object_total_loss = 0.0
        all_correct = 0
        all_pred = 0

        # Use visual_features to get estimated object tokens
        token_features = visual_features.float()
        token_features = F.normalize(token_features, dim=-1)#[200,5,512]
        out_features = set_model(token_features) #[200,512]

        estimated_tokens = out_features.reshape((out_features.shape[0],no_of_new_tokens,args.hparams.emb_dim)) #[200,1, 512]

        #Find target text_features
        with torch.no_grad():
            natural_prompt_labels = [natural_prompt[:-2] + str(local_label) for local_label in (target_labels)]
            text = clip.tokenize(natural_prompt_labels).to(device)
            natural_prompt_embeddings = model.encode_text(text)
            natural_prompt_embeddings /= natural_prompt_embeddings.norm(dim=-1, keepdim=True)
            natural_prompt_embeddings = F.normalize(natural_prompt_embeddings, dim=-1)

        #Use estimated object tokens to get embeddings with text_asterix
        natural_prompt_asterix = [natural_prompt for _ in (target_labels)]
        text = clip.tokenize(natural_prompt_asterix).to(device)
        base_token = None
        natural_prompt_asterix_embeddings = model.encode_text_with_learnt_tokens(text, asterix_token, estimated_tokens, base_token)

        natural_prompt_asterix_embeddings = F.normalize(natural_prompt_asterix_embeddings, dim=-1)#[200,512]

        #Optimize for embeddings with text_asterix to reconstruct text_features
        if args.hparams.loss_str == "contrastive":
            loss = criterion(natural_prompt_embeddings, natural_prompt_asterix_embeddings, args.hparams.contrastive_temp)
        if args.hparams.loss_str == "cosine":
            loss = cosine_loss(natural_prompt_embeddings, natural_prompt_asterix_embeddings)
        total_num += natural_prompt_embeddings.size(0)
        total_loss += loss.item() * natural_prompt_embeddings.size(0)

        if args.hparams.is_gt_object_loss:
            text = clip.tokenize(target_labels).to(device)
            gt_tokens = model.token_embedding(text)
            gt_object_loss = l2_norm_loss(gt_tokens[:,[1],:], estimated_tokens)
            gt_object_total_loss += gt_object_loss.item() * args.hparams.batch_size

        # both embedding-level and token level alignment
        loss = args.hparams.coeff_cycle_loss*loss + args.hparams.coeff_gt_object_loss*gt_object_loss

        if is_train:
            loss.backward()
            optimizer.step()

        # accumulate accuracy for eval phase
        if not is_train:
            correct_batch, pred_batch = get_clip_text_acc(natural_prompt_embeddings, natural_prompt_asterix_embeddings)
            all_correct += correct_batch
            all_pred += pred_batch

    scheduler.step()

    avg_loss = total_loss / total_num
    avg_gt_object_loss = gt_object_total_loss / total_num if args.hparams.is_gt_object_loss else 0.0

    if is_train:
        print(f"ep {epoch} train loss {avg_loss:.5f}")
    else:
        print(f"\nep {epoch} test  loss {avg_loss:.5f}")
        acc = all_correct / all_pred if all_pred > 0 else 0.0
        wandb.log({
            'test/loss': avg_loss,
            'test/accuracy': acc,
            'train gt_object_loss': avg_gt_object_loss
        }, step=epoch, sync=False)
        args.hparams.test_accuracy = acc



def cosine_loss(out_1, out_2):
    # No license needed, can be moved to nv.py?
    cos = CosineSimilarity(dim=1, eps=1e-6)
    loss = -cos(out_1, out_2).mean()

    return loss

def l2_norm_loss(out_1, out_2):
    # No license needed, can be moved to nv.py?
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    loss = MSELoss()
    output = loss(out_1,out_2)
    return output





def contrastive_loss(v1, v2, temperature = 0.25):
    v1 = F.normalize(v1, dim=1)
    v2 = F.normalize(v2, dim=1)

    numerator = torch.exp(torch.diag(torch.inner(v1,v2))/temperature)
    numerator = torch.cat((numerator,numerator), 0)
    joint_vector = torch.cat((v1,v2), 0)
    pairs_product = torch.exp(torch.mm(joint_vector,joint_vector.t()) / temperature)
    denominator = torch.sum(pairs_product - pairs_product*torch.eye(joint_vector.shape[0]).cuda(), 0)

    loss = -torch.mean(torch.log(numerator/denominator))

    return loss

def main():

    parser = ArgumentParser()
    parser.add_arguments(HParams, dest="hparams")
    args, unknown = parser.parse_known_args()
    print("")
    print("args",args)

    wandb.init(project=args.hparams.project_name, config=args.hparams) #I can name the run here

    batch_size = args.hparams.batch_size
    deep_set_d_dim = args.hparams.deep_set_d_dim
    dropout = args.hparams.dropout
    shot_size = args.hparams.shot_size
    learning_rate = args.hparams.lr
    epochs = args.hparams.epochs

    #Deep set model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, emb_dim,preprocess = clip.load("ViT-B/32", device=device)
    # ViT-L/14
    model, emb_dim, preprocess = clip.load("ViT-L/14", device=device)
    #Add personalized text encoder method to CLIP
    funcType = type(model.encode_text)
    model.encode_text_with_learnt_tokens = funcType(encode_text_with_learnt_tokens, model)

    model.eval()

    #Deep set model
    set_model = deep_set(deep_set_d_dim, x_dim=emb_dim, out_dim=args.hparams.no_of_new_tokens*emb_dim, pool=args.hparams.pooling_type, dropout=dropout)
    set_model = set_model.to(device)

    #Load data
    lable_img_dict = np.load(args.hparams.lable_img_dict_path, allow_pickle=True)
    lable_img_dict = lable_img_dict['arr_0'].tolist()
    classname_key_vals = list(lable_img_dict.keys())

    train_dataset = CustomTextDataset(classname_key_vals, lable_img_dict, shot_size, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CustomTextDataset(classname_key_vals, lable_img_dict, shot_size, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    #Load image feautres
    data_img_features = np.load(args.hparams.visual_features_path, allow_pickle=True)
    data_img_features_dict = data_img_features['arr_0'].tolist()


    #Optimization
    optimizer = optim.Adam(set_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.2)

    # # Parameters
    num_params = sum(p.numel() for p in set_model.parameters() if p.requires_grad)
    print(f"number of params: {str(num_params / 1000 ** 2) + 'M'}")

    print("############## Starting ##############")
    start_time = time.time()

    #Training
    best_score = 0.0
    for epoch in range(epochs):
        run_epoch(args, epoch, train_dataloader, optimizer, scheduler, model, set_model, device, data_img_features_dict, is_train=True)

        if (epoch+1)%100==0:
            with torch.no_grad():
                model.eval()
                run_epoch(args, epoch, test_dataloader, optimizer, scheduler, model, set_model, device, data_img_features_dict, is_train=False)
            if args.hparams.test_accuracy >best_score:
                best_score=args.hparams.test_accuracy
                os.makedirs(args.hparams.save_path, exist_ok=True)
                torch.save(set_model.state_dict(), os.path.join(args.hparams.save_path, "set_model_xxxx.pt"))

    # measure elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    main()


