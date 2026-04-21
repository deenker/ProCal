from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector

import PIL
import os
import torch
import torch.nn.functional as F
import torchvision
import json
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

def _convert_to_rgb(image):
    return image.convert('RGB')

@DETECTORS.register_module()
class FViT(TwoStageDetector):
    def simple_test(self, img, img_metas, proposals=None, rescale=False, bnk_embeds=None, pen_backbone=None, clip_preprocess=None):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        mlvl_feats = self.backbone(img)
        if self.with_neck:
            x = self.neck(mlvl_feats[:-1])
        else:
            x = mlvl_feats[:-1]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        ######################################################################
        ### penalty
        img_id = int(img_metas[0]['filename'].split('/')[-1].split('.')[0])
        baseline = False
        if args.load_proposal_prior or args.baseline:
            if args.baseline:
                PENALTY = None
            else: 
                json_folder = f'./{args.prior_folder}/{img_id}/'
                
                with open(f'{json_folder}/proposal_prior.json', 'r') as f:
                    PENALTY = json.load(f)
                for k, v in PENALTY.items():
                    PENALTY[k] = torch.tensor(v).cuda()
        else:
            img = PIL.Image.open(img_metas[0]['filename']).convert('RGB')
            # img_input = clip_preprocess(img).unsqueeze(0).cuda()
            clip_preprocess_ = torchvision.transforms.Compose([
                    _convert_to_rgb,
                    ToTensor(),
                    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                ])
            img_input = clip_preprocess_(img).unsqueeze(0).cuda()
            crop_imgs = torchvision.ops.roi_pool(img_input, [proposal_list[0][:, :4]], output_size=(224, 224))
            crop_feats = pen_backbone.encode_image(crop_imgs)
            crop_feats = F.normalize(crop_feats, dim=-1, p=2)     
            
            bnk_sim = crop_feats @ bnk_embeds['bnk_embed'].T
            bnk_sim4 = crop_feats @ bnk_embeds['bnk_embed4'].T
            bg_bnk_sim = crop_feats @ bnk_embeds['bg_bnk_embed'].T
            
            PENALTY = dict()
            PENALTY['bnk_sim'] = bnk_sim
            PENALTY['bnk_sim4'] = bnk_sim4
            PENALTY['bg_bnk_sim'] = bg_bnk_sim
            
            if args.save_proposal_prior:
                json_folder = f'./{args.prior_folder}/{img_id}/'
                if os.path.exists(json_folder) == False:
                    os.makedirs(json_folder)
                with open(f'{json_folder}/proposal_prior.json', 'w') as f:
                    json.dump({k: v.cpu().tolist() for k, v in PENALTY.items()}, f, indent=4)

            return None
        

        ######################################################################
        
        res = self.roi_head.simple_test(x,
                                        proposal_list,
                                        img_metas,
                                        vlm_feat=mlvl_feats[-1],
                                        rescale=rescale, PENALTY=PENALTY)

        

        return res

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_captions=None,
                      gt_embeds=None,
                      **kwargs):
        res_feats = self.backbone(img)
        if self.with_neck:
            x = self.neck(res_feats[:-1])
        else:
            x = res_feats[:-1]

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 res_feats=res_feats,
                                                 gt_captions=gt_captions,
                                                 gt_embeds=gt_embeds,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
