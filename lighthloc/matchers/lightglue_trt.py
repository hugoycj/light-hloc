from ..utils.base_model import BaseModel
import onnxruntime as ort
import torch
import time

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
        sess_options = ort.SessionOptions()
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        trt = True
        if trt:
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "weights/cache",
                    },
                )
            ] + providers
            
        self.net = ort.InferenceSession(
            'lighthloc/matchers/superpoint_lightglue.onnx', sess_options=sess_options, providers=providers
        )
        
    def _forward(self, data):
        kpts0, kpts1 = data["keypoints0"].cpu().numpy(), data["keypoints1"].cpu().numpy()
                
        desc0 = data['descriptors0'].transpose(-1, -2).cpu().numpy()
        desc1 = data['descriptors1'].transpose(-1, -2).cpu().numpy()

        m0, m1, mscores0, mscores1 = self.net.run(
                None,
                {
                    "kpts0": kpts0,
                    "kpts1": kpts1,
                    "desc0": desc0,
                    "desc1": desc1,
                },
            )
        end_time = time.time()


        return {
            "matches0": torch.from_numpy(m0),
            "matches1": torch.from_numpy(m1),
            "matching_scores0": torch.from_numpy(mscores0),
            "matching_scores1": torch.from_numpy(mscores1),
        }