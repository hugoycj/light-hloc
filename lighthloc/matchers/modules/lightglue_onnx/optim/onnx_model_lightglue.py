from logging import getLogger
from typing import Optional

from onnx import ModelProto
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.fusion_transpose import FusionTranspose
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel

from .fusion_attention_lightglue import FusionAttentionLightGlue

logger = getLogger(__name__)


class LightGlueOnnxModel(BertOnnxModel):
    def __init__(
        self,
        model: ModelProto,
        num_heads: int = 0,
        hidden_size: int = 0,
    ):
        super().__init__(model, num_heads, hidden_size)

    def preprocess(self):
        self.remove_useless_div()

    def postprocess(self):
        self.prune_graph()
        self.remove_unused_constant()

    def remove_useless_div(self):
        """Remove Div by 1"""
        div_nodes = [node for node in self.nodes() if node.op_type == "Div"]

        nodes_to_remove = []
        for div in div_nodes:
            if self.find_constant_input(div, 1.0) == 1:
                nodes_to_remove.append(div)

        for node in nodes_to_remove:
            self.replace_input_of_all_nodes(node.output[0], node.input[0])

        if nodes_to_remove:
            self.remove_nodes(nodes_to_remove)
            logger.info("Removed %d Div nodes", len(nodes_to_remove))

    def merge_adjacent_transpose(self):
        fusion_transpose = FusionTranspose(self)
        fusion_transpose.apply()

    def fuse_multi_head_attention(self, options: Optional[FusionOptions] = None):
        # Self Attention
        enable_packed_qkv = (options is None) or options.enable_packed_qkv
        self_attention_fusion = FusionAttentionLightGlue(
            self,
            self.hidden_size,
            self.num_heads,
            is_cross_attention=False,
            enable_packed_qkv=enable_packed_qkv,
            enable_packed_kv=False,
        )
        self_attention_fusion.apply()

        # Cross Attention
        enable_packed_kv = (options is None) or options.enable_packed_kv
        cross_attention_fusion = FusionAttentionLightGlue(
            self,
            self.hidden_size,
            self.num_heads,
            is_cross_attention=True,
            enable_packed_qkv=False,
            enable_packed_kv=enable_packed_kv,
        )
        cross_attention_fusion.apply()

    def optimize(self, options: Optional[FusionOptions] = None):
        if (options is not None) and not options.enable_shape_inference:
            self.disable_shape_inference()

        self.utils.remove_identity_nodes()

        # Remove cast nodes that having same data type of input and output based on symbolic shape inference.
        self.utils.remove_useless_cast_nodes()

        if (options is None) or options.enable_layer_norm:
            self.fuse_layer_norm()

        if (options is None) or options.enable_gelu:
            self.fuse_gelu()

        self.preprocess()

        self.fuse_reshape()

        if (options is None) or options.enable_attention:
            self.fuse_multi_head_attention(options)

        self.fuse_shape()

        # Remove reshape nodes that having same shape of input and output based on symbolic shape inference.
        self.utils.remove_useless_reshape_nodes()

        if options is None or options.enable_nhwc_conv:
            self.merge_adjacent_transpose()

        self.postprocess()

        logger.info(f"opset version: {self.get_opset_version()}")

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        ops = [
            "MultiHeadAttention",
            "LayerNormalization",
            "Gelu",
        ]
        for op in ops:
            nodes = self.get_nodes_by_op_type(op)
            op_count[op] = len(nodes)

        logger.info(f"Optimized operators:{op_count}")
        return op_count
