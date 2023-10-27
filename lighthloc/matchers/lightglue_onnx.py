from ..utils.base_model import BaseModel
from .modules.lightglue_onnx import LightGlue as LightGlue_
# from lightglue import LightGlue as LightGlue_

class LightGlue(BaseModel):
    default_conf = {
        'features': 'superpoint',
        'depth_confidence': 0.95,
        'width_confidence': 0.99,
    }
    required_inputs = [
        'image0', 'keypoints0', 'descriptors0',
        'image1', 'keypoints1', 'descriptors1',
    ]

    def _init(self, conf):
        self.net = LightGlue_(conf.pop('features'), **conf)

    def _forward(self, data):
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
                
        desc0 = data['descriptors0'].transpose(-1, -2)
        desc1 = data['descriptors1'].transpose(-1, -2)

        m0, m1, mscores0, mscores1, matches, mscores = self.net(kpts0.contiguous(), kpts1.contiguous(), desc0, desc1)

        return {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "matches": matches,
            "scores": mscores,
        }