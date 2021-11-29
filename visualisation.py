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
from model import LATransformerTest_Pooldualsum
from utils import get_id
from torchvision.utils import save_image


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
    parser.add_argument('-m', '--is_model_baseline', type=bool, default=True, required=True, \
                                                        help="To choose which model to invoke baseline/improved") 
    
    parser.add_argument('-v', '--vis_path', type=str, default=True, required=True, \
                                                        help="Path to store images") 
    args = parser.parse_args()
    return args

def extract_feature(model,dataloaders,device):
    """
    Used to get the features/representation of queryset and galleryset
    """
    features =  torch.FloatTensor()
    count = 0
    idx = 0
    images= {}
    for index,data in enumerate(dataloaders):
        img, label = data    
        img, label = img.to(device), label.to(device)

        output = model(img) # (B, D, H, W) --> B: batch size, HxWxD: feature volume size

        n, c, h, w = img.size()
        
        count += n
        features = torch.cat((features, output.detach().cpu()), 0)
        idx += 1
        images[index]=img
    return features,images

def extract_feature_flip(model,dataloaders,device):
    """
    Used to get the features/representation of queryset and galleryset
    """
    features =  torch.FloatTensor()
    count = 0
    idx = 0
    images = {}
    for index,data in enumerate(dataloaders):
        img, label = data    
        img, label = img.to(device), label.to(device)

        img_flip= torch.flip(img,[3])
        output = model(img) # (B, D, H, W) --> B: batch size, HxWxD: feature volume size
        output_flip = model(img_flip)
        output = (output+output_flip)/2

        n, c, h, w = img.size()
        
        count += n
        features = torch.cat((features, output.detach().cpu()), 0)
        idx += 1
        images[index]=img
    return features,images

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
    model_type = args.is_model_baseline
    vis_path = args.vis_path
    
    """
    Checks the availibility of a GPU
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {}".format(device))

    use_gpu=False
    if device=='cuda':
        use_gpu=True
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
    
    if model_type:
        model = LATransformer(vit_base, lmbd,num_classes,num_la_blocks,blocks,int_dim).to(device)
        model.load_state_dict(torch.load(wts_path), strict=False)
        model.eval()
    else:
        model = LATransformerTest_Pooldualsum(vit_base, lmbd,num_classes,num_la_blocks,blocks,int_dim).to(device)
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
    
    
    
    if model_type:
    
        # Extract Query Features
        query_feature,query_images= extract_feature(model,query_loader,device)
        
        # Extract Gallery Features 
        gallery_feature,gallery_images = extract_feature(model,gallery_loader,device)
    
    else:
        # Extract Query Features
        query_feature,query_images= extract_feature_flip(model,query_loader,device)
        
        # Extract Gallery Features
        gallery_feature,gallery_images = extract_feature_flip(model,gallery_loader,device)
        
    
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
    index = (faiss.IndexFlatIP((14*768)))
    index.add(np.array([t.numpy() for t in concatenated_gallery_vectors]))
    
    index2 = faiss.IndexIDMap(faiss.IndexFlatIP((14*768)))
    index2.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]),np.array(gallery_label).astype(np.int64))
    
    ## Evaluate
    mod_str = "improved"
    if model_type:
        mod_str = "baseline"
    
    itr_no = 0
    pred_img= None
    for query, label in zip(concatenated_query_vectors, query_label):
        label = label
        encoded_query = query.unsqueeze(dim=0).numpy()
        output = index.search(encoded_query,10)
        output2 = index2.search(encoded_query,10)
        label_pred = output2[1][0][0]
        if label_pred != label:
            list_preds = output[1]
            pred_img = gallery_images[list_preds[0][0]]
            pred_img2 = gallery_images[list_preds[0][1]]
            pred_img3 = gallery_images[list_preds[0][2]]
            pred_img4 = gallery_images[list_preds[0][3]]
            pred_img5 = gallery_images[list_preds[0][4]]
            inp_img = query_images[itr_no]
            save_image(inp_img[0], vis_path+'/query_'+mod_str+'_'+str(itr_no)+'.png')
            save_image(pred_img[0], vis_path+'/gallery_'+mod_str+'_1_'+str(itr_no)+'.png')
            save_image(pred_img2[0], vis_path+'/gallery_'+mod_str+'_2_'+str(itr_no)+'.png')
            save_image(pred_img3[0], vis_path+'/gallery_'+mod_str+'_3_'+str(itr_no)+'.png')
            save_image(pred_img4[0], vis_path+'/gallery_'+mod_str+'_4_'+str(itr_no)+'.png')
            save_image(pred_img5[0], vis_path+'/gallery_'+mod_str+'_5_'+str(itr_no)+'.png')
        itr_no +=1



