import torch
import torch.nn as nn
from torch.nn import init

def weights_init_kaiming(m):
    """
    Initialization of weights of the model layers 
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    """
    Initialization of the classifier head wts of the model
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
        
class ClassBlock(nn.Module):
    """
    loclly aware network structure
    """
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x
        
class LATransformer(nn.Module):
    """
    The main model architecture
    Here the "model" param in __init__ is the ViT backbone
    """
    def __init__(self, model, lmbd, class_num,part,num_blocks,int_dim ):
        super(LATransformer, self).__init__()
        self.class_num = class_num
        self.part = part # We cut the pool5 to sqrt(N) parts
        self.num_blocks = num_blocks
        self.model = model
        self.model.head.requires_grad_ = False 
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,int_dim))
        self.dropout = nn.Dropout(p=0.5)
        self.lmbd = lmbd
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(int_dim, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        

    def forward(self,x):
        
        # Divide input image into patch embeddings and add position embeddings
        x = self.model.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.pos_embed)
        
        # Feed forward through transformer blocks
        for i in range(self.num_blocks):
            x = self.model.blocks[i](x)
        x = self.model.norm(x)
        
        # extract the cls token
        cls_token_out = x[:, 0].unsqueeze(1)
        
        # Average pool
        x = self.avgpool(x[:, 1:])
        
        # Add global cls token to each local token 
        for i in range(self.part):
            out = torch.mul(x[:, i, :], self.lmbd)
            x[:,i,:] = torch.div(torch.add(cls_token_out.squeeze(),out), 1+self.lmbd)
        
        # Locally aware network
        part = {}
        predict = {}
        for i in range(self.part):
            part[i] = x[:,i,:]
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])
        return predict

class LATransformer_AMS(nn.Module):
    """
    The main model architecture
    Here the "model" param in __init__ is the ViT backbone
    """
    def __init__(self, model, lmbd, class_num,part,num_blocks,int_dim ):
        super(LATransformer_AMS, self).__init__()
        self.class_num = class_num
        self.part = part # We cut the pool5 to sqrt(N) parts
        self.num_blocks = num_blocks
        self.model = model
        self.model.head.requires_grad_ = False 
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,int_dim))
        self.dropout = nn.Dropout(p=0.5)
        self.lmbd = lmbd
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(int_dim, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256,return_f=True))

        

    def forward(self,x):
        
        # Divide input image into patch embeddings and add position embeddings
        x = self.model.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.pos_embed)
        
        # Feed forward through transformer blocks
        for i in range(self.num_blocks):
            x = self.model.blocks[i](x)
        x = self.model.norm(x)
        
        # extract the cls token
        cls_token_out = x[:, 0].unsqueeze(1)
        
        # Average pool
        x = self.avgpool(x[:, 1:])
        
        # Add global cls token to each local token 
        for i in range(self.part):
            out = torch.mul(x[:, i, :], self.lmbd)
            x[:,i,:] = torch.div(torch.add(cls_token_out.squeeze(),out), 1+self.lmbd)
        
        # Locally aware network
        part = {}
        predict = {}
        pred_classes = {}
        for i in range(self.part):
            part[i] = x[:,i,:]
            name = 'classifier'+str(i)
            c = getattr(self,name)
            temp = c(part[i])
            predict[i] = temp[1]
            pred_classes[i] = temp[0]
        return (predict,pred_classes)

class LATransformerTest(nn.Module):
    """
    This architecture excludes the loacally classifier ensemble
    It helps in getting the feature during inference time which are then fed to faiss
    """
    def __init__(self, model, lmbd, class_num,part,num_blocks,int_dim ):
        super(LATransformerTest, self).__init__()
        
        self.class_num = class_num
        self.part = part # We cut the pool5 to sqrt(N) parts
        self.num_blocks = num_blocks
        self.model = model
        self.model.head.requires_grad_ = False 
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,int_dim))
        self.dropout = nn.Dropout(p=0.5)
        self.lmbd = lmbd
#         for i in range(self.part):
#             name = 'classifier'+str(i)
#             setattr(self, name, ClassBlock(768, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        

    def forward(self,x):
        
        # Divide input image into patch embeddings and add position embeddings
        x = self.model.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.pos_embed)
        
        # Feed forward through transformer blocks
        for i in range(self.num_blocks):
            x = self.model.blocks[i](x)
        x = self.model.norm(x)
        
        # Average pool
        x = self.avgpool(x[:, 1:])
        
        # Add global cls token to each local token 
#         for i in range(self.part):
#             out = torch.mul(x[:, i, :], self.lmbd)
#             x[:,i,:] = torch.div(torch.add(cls_token_out.squeeze(),out), 1+self.lmbd)

        return x.cpu()
    
"""
device = 'cpu'
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
vit_base= vit_base.to(device)
vit_base.eval()
num_la_blocks = 14
blocks = 12
int_dim = 768
num_classes = 62
model = LATransformer_AMS(vit_base, lmbd,num_classes,num_la_blocks,blocks,int_dim).to(device)
op = model(inp)
"""