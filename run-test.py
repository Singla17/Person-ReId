# Evaluation 
# Acknowledgement: the code is based on Siddhant Kapil's repo on LA-Transformer

import os
import faiss
import numpy as np

import argparse
import timm
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import LATransformerTest as LATransformer
from utils import get_id
from metrics import rank1, rank5, calc_ap


def parse_args():
    """
    Returns the arguments parsed from command line
    """
    parser = argparse.ArgumentParser(description='Getting various paths')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--model_path', type=str, default='result', required=True, \
                                                        help="Path for the model weights")
    parser.add_argument('-c', '--num_classes', type=int, default=62, required=True, \
                                                        help="Number of classes in trainset")
        
    args = parser.parse_args()
    return args

def extract_feature(model,dataloaders,device):
    """
    Used to get the features/representation of queryset and galleryset
    """
    features =  torch.FloatTensor()
    count = 0
    idx = 0
    for index,data in enumerate(dataloaders):
        img, label = data    
        img, label = img.to(device), label.to(device)

        output = model(img) # (B, D, H, W) --> B: batch size, HxWxD: feature volume size

        n, c, h, w = img.size()
        
        count += n
        features = torch.cat((features, output.detach().cpu()), 0)
        idx += 1
    return features

def search(index,query: str, k=1):
    """
    to do a similarity based search using faiss library
    """
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    return top_k

if __name__ == "__main__":
    """
    Setting up input and output paths
    """
    args = parse_args()
    inp_path = args.inp_path
    wts_path = args.model_path
    num_classes = args.num_classes
    
    """
    Checks the availibility of a GPU
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {}".format(device))


    """
    Load saved model
    """
    vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    vit_base= vit_base.to(device)

    # Create La-Transformer
    num_la_blocks = 14
    blocks = 12
    int_dim = 768
    lmbd =8
    model = LATransformer(vit_base, lmbd,num_classes,num_la_blocks,blocks,int_dim).to(device)
    model.load_state_dict(torch.load(wts_path), strict=False)
    model.eval()
    
    batch_size = 1
    
    ## Data Loader for query and gallery
    
    transform_query_list = [
            transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_gallery_list = [
            transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    
    data_transforms = {
            'query': transforms.Compose( transform_query_list ),
            'gallery': transforms.Compose(transform_gallery_list),
        }
    
    
    image_datasets = {}
    
    image_datasets['query'] = datasets.ImageFolder(os.path.join(inp_path, 'query').replace("\\","/"),
                                              data_transforms['query'])
    image_datasets['gallery'] = datasets.ImageFolder(os.path.join(inp_path, 'gallery').replace("\\","/"),
                                              data_transforms['gallery'])
    query_loader = DataLoader(dataset = image_datasets['query'], batch_size=batch_size, shuffle=False )
    gallery_loader = DataLoader(dataset = image_datasets['gallery'], batch_size=batch_size, shuffle=False)
    
    # Extract Query Features
    
    query_feature= extract_feature(model,query_loader,device)
    
    # Extract Gallery Features
    
    gallery_feature = extract_feature(model,gallery_loader,device)
    
    # Retrieve labels
    
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs
    
    gallery_cam,gallery_label = get_id(gallery_path)
    query_cam,query_label = get_id(query_path)
    
    ## Concat Averaged GELTs
    
    concatenated_query_vectors = []
    for idx,query in enumerate(query_feature):
        fnorm = torch.norm(query, p=2, dim=1, keepdim=True)*np.sqrt(14)
        query_norm = query.div(fnorm.expand_as(query))
        concatenated_query_vectors.append(query_norm.view((-1)))
    
    concatenated_gallery_vectors = []
    for idx,gallery in enumerate(gallery_feature):
        fnorm = torch.norm(gallery, p=2, dim=1, keepdim=True)*np.sqrt(14)
        gallery_norm = gallery.div(fnorm.expand_as(gallery))
        concatenated_gallery_vectors.append(gallery_norm.view((-1)))
      
    
    ## Calculate Similarity using FAISS
    index = faiss.IndexIDMap(faiss.IndexFlatIP((14*768)))
    index.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]),np.array(gallery_label).astype(np.int64))
    
    ## Evaluate 
    rank1_score = 0
    rank5_score = 0
    ap = 0
    count = 0
    for query, label in zip(concatenated_query_vectors, query_label):
        count += 1
        label = label
        output = search(index,query, k=10)
        rank1_score += rank1(label, output) 
        rank5_score += rank5(label, output) 
        print("Correct: {}, Total: {}, Incorrect: {}".format(rank1_score, count, count-rank1_score))
        ap += calc_ap(label, output)
    
    print("Rank1: %.3f, Rank5: %.3f, mAP: %.3f"%(rank1_score/len(query_feature), 
                                                 rank5_score/len(query_feature), 
                                                 ap/len(query_feature)))    



