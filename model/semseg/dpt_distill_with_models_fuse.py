import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from model.backbone.dinov2 import DINOv2
from model.util.blocks import FeatureFusionBlock, _make_scratch
from transformers import SamProcessor
from model.semseg.clip_modifie.clip import load as load_clip
# from transformers import CLIPModel
from torchvision.transforms import ToPILImage
from PIL import Image
def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv(path_1)
        
        return out

class feature_translator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(feature_translator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, student_feature):
        return self.layer(student_feature)

class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=21,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
        size=518,
        hidden_layer_dim_for_translator=1024
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23], 
            'giant': [9, 19, 29, 39]
        }
        
        self.encoder_size = encoder_size
        self.backbone = DINOv2(model_name=encoder_size)
        self.embed_dim = self.backbone.embed_dim
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
        #dino teacher
        self.Dino_teacher_model = DINOv2(model_name='base')
        state_dict = torch.load('./pretrained/dinov2_base.pth')
        self.Dino_teacher_model.load_state_dict(state_dict)
        self.Dino_teacher_dim = self.Dino_teacher_model.embed_dim
        # clip teacher
        self.clip_model, _ = load_clip("ViT-L/14", device="cuda")
        # frozen teacher model
        for param in self.Dino_teacher_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.parameters():
            param.requires_grad = False  
        self.patch_num = size // 14
        # feature translator
        self.feature_translator_for_dino = feature_translator(self.embed_dim, self.Dino_teacher_dim, hidden_layer_dim_for_translator)
        self.feature_translator_for_clip = feature_translator(self.embed_dim, 1024, hidden_layer_dim_for_translator)
        # self.feature_translator_for_sam = feature_translator(self.embed_dim, self.sam_model_dim, hidden_layer_dim_for_translator)
        self.project_dino = nn.Linear(self.Dino_teacher_dim, self.embed_dim)
        self.project_clip = nn.Linear(1024, self.embed_dim)
        

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def forward(self, x, compute_loss=False):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )   
        student_feature = features[-1] 
        x_clip = F.interpolate(x, (224, 224), mode='bilinear', align_corners=True)
        student_clip_feature = self.backbone.get_intermediate_layers(
            x_clip, self.intermediate_layer_idx[self.encoder_size]
        )[-1]
        ## translate feature
        student_dino_feature = self.feature_translator_for_dino(student_feature)
        student_clip_feature = self.feature_translator_for_clip(student_clip_feature)
        B = x.shape[0]
        student_clip_map = student_clip_feature.permute(0, 2, 1).reshape(B, 1024, 16, 16)  # [B, 1024, 16, 16]
        student_clip_map_resized = F.interpolate(student_clip_map, size=(patch_h, patch_w), mode='bilinear', align_corners=True)  # → [B, 1024, 37, 37]
        student_clip_aligned = student_clip_map_resized.reshape(B, 1024, -1).permute(0, 2, 1)  # [B, 1369, 1024]
        student_dino_feature_aligned = self.project_dino(student_dino_feature)      # [B, 1369, 768]
        student_clip_aligned = self.project_clip(student_clip_aligned)              # [B, 1369, 768]
        fused_feature = 0.8 * student_feature +0.1 * (student_dino_feature_aligned + student_clip_aligned)
        features = list(features)  
        features[-1] = fused_feature
        features = tuple(features)
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        if compute_loss:
            with torch.no_grad():
                dino_features = self.Dino_teacher_model.get_intermediate_layers(
                    x, self.intermediate_layer_idx['base']
                )
                dino_feature = dino_features[-1]
                clip_features = self.clip_model.encode_image(x_clip).float() 
            # student_sam_feature = self.feature_translator_for_sam(student_feature)
            loss_1 = F.mse_loss(student_dino_feature, dino_feature)
            loss_2 = F.mse_loss(student_clip_feature, clip_features)
            # loss_3 = F.mse_loss(student_sam_feature, sam_features)
            loss =(loss_1 + loss_2)/2
            return out, loss
        return out