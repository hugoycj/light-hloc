from logging import getLogger
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from onnx import NodeProto, TensorProto, helper
from onnxruntime.transformers.fusion_base import Fusion
from onnxruntime.transformers.onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionAttentionLightGlue(Fusion):
    """
    Fuse Attention subgraph of LightGlue into one MultiHeadAttention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        is_cross_attention: bool,
        enable_packed_qkv: bool,
        enable_packed_kv: bool,
    ):
        super().__init__(model, "MultiHeadAttention", ["LayerNormalization"])
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.is_cross_attention = is_cross_attention
        self.enable_packed_qkv = enable_packed_qkv
        self.enable_packed_kv = enable_packed_kv

    def fuse(self, normalize_node: NodeProto, input_name_to_nodes, output_name_to_node):
        match = self.match_sdpa(normalize_node)
        if match is None:
            return

        is_self_attention = not self.is_cross_attention
        if is_self_attention:
            reshape_qkv, transpose_qkv, matmul_q, matmul_k, matmul_v = match
        else:
            reshape_qkv, transpose_qkv, add_q, add_k, add_v = match

        attention_last_node = reshape_qkv

        if is_self_attention:
            new_node = self.create_self_attention_node(
                matmul_q,
                matmul_k,
                matmul_v,
                self.hidden_size,
                self.num_heads,
                output=attention_last_node.output[0],
            )
        else:  # cross-attn
            new_node = self.create_cross_attention_node(
                add_q,
                add_k,
                add_v,
                self.hidden_size,
                self.num_heads,
                output=attention_last_node.output[0],
            )

        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
        self.nodes_to_add.append(new_node)

        self.nodes_to_remove.extend([attention_last_node, transpose_qkv])

        # Use prune graph to remove nodes since they are shared by all attention nodes.
        self.prune_graph = True

    def match_sdpa(self, node: NodeProto) -> Optional[Tuple[NodeProto, ...]]:
        is_self_attention = not self.is_cross_attention
        self.model: OnnxModel
        nodes_before_sdpa = self.model.match_parent_path(
            node,
            [
                "Add",
                "MatMul",
                "Concat",
                "Add",
                "MatMul",
                "Reshape",
                "Transpose",
                "MatMul",
            ],
            [None, None, None, 1, None, None, None, None],
        )

        if nodes_before_sdpa is None:
            return

        (*_, reshape_qkv, transpose_qkv, matmul_qkv) = nodes_before_sdpa

        assert "inner_attn" in matmul_qkv.name
        if is_self_attention and "self_attn" not in matmul_qkv.name:
            return
        if self.is_cross_attention and "cross_attn" not in matmul_qkv.name:
            return

        # V nodes
        if is_self_attention:
            # Due to how LightGlue applies positional encoding in self-attention,
            # the *matmul* nodes for Q, K, V are different from usual.
            # matmul_v is actually Gather, whereas matmul_q and matmul_k are Add.
            v_nodes = self.model.match_parent_path(matmul_qkv, ["Gather"], [None])
            if v_nodes is None:
                logger.debug("fuse_attention: failed to match v path")
                return
            (matmul_v,) = v_nodes
        else:  # cross-attn
            # LightGlue uses bias in the input projection.
            v_nodes = self.model.match_parent_path(
                matmul_qkv, ["Transpose", "Reshape", "Add"], [1, 0, 0]
            )
            if v_nodes is None:
                logger.debug("fuse_attention: failed to match v path")
                return None
            (_, _, add_v) = v_nodes

        # QK nodes
        qk_nodes = self.model.match_parent_path(
            matmul_qkv, ["Softmax", "MatMul"], [0, 0]
        )
        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return
        (_softmax_qk, matmul_qk) = qk_nodes

        # Q nodes
        if is_self_attention:
            q_nodes = self.model.match_parent_path(matmul_qk, ["Mul", "Add"], [0, None])
            if q_nodes is None:
                logger.debug("fuse_attention: failed to match q path")
                return
            (mul_q, matmul_q) = q_nodes
        else:  # cross-attn
            q_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Mul", "Transpose", "Reshape", "Add"],
                [0, None, 0, 0],
            )
            if q_nodes is None:
                logger.debug("fuse_attention: failed to match q path")
                return None
            (mul_q, _transpose_q, reshape_q, add_q) = q_nodes

        # K nodes
        if is_self_attention:
            k_nodes = self.model.match_parent_path(
                matmul_qk, ["Mul", "Transpose", "Add"], [1, None, None]
            )
            if k_nodes is None:
                logger.debug("fuse_attention: failed to match k path")
                return
            (_mul_k, _, matmul_k) = k_nodes
        else:  # cross-attn
            k_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Mul", "Transpose", "Reshape", "Add"],
                [1, None, 0, 0],
            )
            if k_nodes is None:
                logger.debug("fuse_attention: failed to match k path")
                return None
            (_mul_k, _, _, add_k) = k_nodes

        # Final check using scale
        if is_self_attention:
            mul_q_nodes = self.model.match_parent_path(
                mul_q,
                ["Sqrt", "Div", "Sqrt", "Cast", "Slice", "Shape", "Add"],
                [None, 0, 1, 0, 0, 0, 0],
            )
            if mul_q_nodes is None or mul_q_nodes[-1] != matmul_q:
                logger.debug("fuse_attention: failed to match mul_q path")
                return
        else:  # cross-attn
            mul_q_nodes = self.model.match_parent_path(
                mul_q,
                [
                    "Sqrt",
                    "Div",
                    "Sqrt",
                    "Cast",
                    "Slice",
                    "Shape",
                    "Transpose",
                    "Reshape",
                ],
                [None, 0, 1, 0, 0, 0, 0, 0],
            )
            if mul_q_nodes is None or mul_q_nodes[-1] != reshape_q:
                logger.debug("fuse_attention: failed to match mul_q path")
                return

        if is_self_attention:
            return reshape_qkv, transpose_qkv, matmul_q, matmul_k, matmul_v
        else:
            return reshape_qkv, transpose_qkv, add_q, add_k, add_v

    def create_self_attention_node(
        self,
        matmul_q: NodeProto,
        matmul_k: NodeProto,
        matmul_v: NodeProto,
        hidden_size: int,
        num_heads: int,
        output: str,
    ) -> NodeProto:
        # all_inputs are (B, N, S, H)
        if self.enable_packed_qkv:
            unsqueeze_q_node_name = self.model.create_node_name("Unsqueeze")
            unsqueeze_k_node_name = self.model.create_node_name("Unsqueeze")
            unsqueeze_v_node_name = self.model.create_node_name("Unsqueeze")
            for n in (
                unsqueeze_q_node_name,
                unsqueeze_k_node_name,
                unsqueeze_v_node_name,
            ):
                self.add_initializer(
                    name=n + "_axis",
                    data_type=TensorProto.INT64,
                    dims=[1],
                    vals=[-1],
                    raw=False,
                )
            unsqueeze_q_node = helper.make_node(
                "Unsqueeze",
                inputs=[matmul_q.output[0], unsqueeze_q_node_name + "_axis"],
                outputs=[unsqueeze_q_node_name + "_out"],
                name=unsqueeze_q_node_name,
            )
            self.node_name_to_graph_name[unsqueeze_q_node.name] = self.this_graph_name
            unsqueeze_k_node = helper.make_node(
                "Unsqueeze",
                inputs=[matmul_k.output[0], unsqueeze_k_node_name + "_axis"],
                outputs=[unsqueeze_k_node_name + "_out"],
                name=unsqueeze_k_node_name,
            )
            self.node_name_to_graph_name[unsqueeze_k_node.name] = self.this_graph_name
            unsqueeze_v_node = helper.make_node(
                "Unsqueeze",
                inputs=[matmul_v.output[0], unsqueeze_v_node_name + "_axis"],
                outputs=[unsqueeze_v_node_name + "_out"],
                name=unsqueeze_v_node_name,
            )
            self.node_name_to_graph_name[unsqueeze_v_node.name] = self.this_graph_name

            concat_node_name = self.model.create_node_name("Concat")
            concat_node = helper.make_node(
                "Concat",
                inputs=[
                    unsqueeze_q_node_name + "_out",
                    unsqueeze_k_node_name + "_out",
                    unsqueeze_v_node_name + "_out",
                ],
                outputs=[concat_node_name + "_out"],
                name=concat_node_name,
                axis=-1,
            )
            self.node_name_to_graph_name[concat_node.name] = self.this_graph_name

            # concat_output is (B, N, S, H, 3)
            # packed QKV needs (B, S, N, 3, H)
            transpose_node_name = self.model.create_node_name("Transpose")
            transpose_node = helper.make_node(
                "Transpose",
                inputs=[concat_node_name + "_out"],
                outputs=[transpose_node_name + "_out"],
                name=transpose_node_name,
                perm=[0, 2, 1, 4, 3],
            )
            self.node_name_to_graph_name[transpose_node.name] = self.this_graph_name

            self.nodes_to_add.extend(
                [
                    unsqueeze_q_node,
                    unsqueeze_k_node,
                    unsqueeze_v_node,
                    concat_node,
                    transpose_node,
                ]
            )

            attention_inputs = [transpose_node_name + "_out"]

            attention_node_name = self.model.create_node_name("MultiHeadAttention")
            attention_node = helper.make_node(
                "MultiHeadAttention",
                inputs=attention_inputs,
                outputs=[output],
                name=attention_node_name,
                domain="com.microsoft",
                num_heads=num_heads,
            )

            return attention_node
        else:  # Not packed
            raise NotImplementedError("Unpacked QKV self-attention not implemented.")

    def create_cross_attention_node(
        self,
        add_q: NodeProto,
        add_k: NodeProto,
        add_v: NodeProto,
        hidden_size: int,
        num_heads: int,
        output: str,
    ) -> NodeProto:
        # all_inputs are (B, S, NH)
        attention_node_name = self.model.create_node_name("MultiHeadAttention")

        attention_inputs = [add_q.output[0], add_k.output[0], add_v.output[0]]

        attention_node = helper.make_node(
            "MultiHeadAttention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
            domain="com.microsoft",
            num_heads=num_heads,
        )
        return attention_node

    def add_initializer(
        self,
        name: str,
        data_type: int,
        dims: Sequence[int],
        vals: Any,
        raw: bool = True,
    ):
        if raw:
            np_type = helper.tensor_dtype_to_np_dtype(data_type)
            if not isinstance(vals, np.ndarray):
                bytes = np.array(vals, dtype=np_type).tobytes()
            else:
                bytes = vals.astype(np_type).tobytes()
            tensor = helper.make_tensor(
                name=name,
                data_type=data_type,
                dims=dims,
                vals=bytes,
                raw=True,
            )
        else:
            tensor = helper.make_tensor(
                name=name,
                data_type=data_type,
                dims=dims,
                vals=vals,
                raw=False,
            )

        self.model.add_initializer(tensor, self.this_graph_name)
        return tensor
