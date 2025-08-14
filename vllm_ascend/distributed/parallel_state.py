from typing import Optional

import torch
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)

# Currently, mc2 op need their own group coordinator.
_MC2: Optional[GroupCoordinator] = None
_MLP_TP: Optional[GroupCoordinator] = None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2

def get_mlp_tp_group() -> GroupCoordinator:
    assert _MLP_TP is not None, ("mlp group is not initialized")
    return _MLP_TP


def model_parallel_initialized():
    return (_MC2 is not None)


def init_ascend_model_parallel(
    expert_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    backend: Optional[str] = None,
    data_parallel_size: int = 1,
):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    num_expert_parallel_groups = world_size // expert_parallel_size

    global _MC2
    group_ranks = []
    for i in range(num_expert_parallel_groups):
        ranks = list(range(i, world_size, num_expert_parallel_groups))
        group_ranks.append(ranks)

    _MC2 = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="mc2")
    
    global _MLP_TP
    assert _MLP_TP is None, ("mlp tensor model parallel group is already initialized")
    
    # Initialize MLP TP group only when ENABLE_MLP_TP is true
    import vllm_ascend.envs as ascend_envs
    enable_mlp_tp = bool(ascend_envs.ENABLE_MLP_TP)
    
    if enable_mlp_tp:
        # When MLP TP is enabled, mlp_tp_size follows data_parallel_size
        mlp_tp = data_parallel_size
        
        # Add logging for MLP TP initialization
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"MLP TP enabled: mlp_tp_size={mlp_tp}, data_parallel_size={data_parallel_size}")
        
        all_ranks_mlp_head = torch.arange(world_size).reshape(
            -1, mlp_tp, pipeline_parallel_size, 1)  # noqa
        group_ranks = all_ranks_mlp_head.view(-1, mlp_tp).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        
        # message queue broadcaster is only used in tensor model parallel group
        _MLP_TP = init_model_parallel_group(group_ranks,
                                                get_world_group().local_rank,
                                                backend,
                                                group_name="mlp_tp")
        logger.info(f"MLP TP group initialized successfully with {len(group_ranks)} groups")
    else:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("MLP TP disabled: using standard tensor parallel for MLP layers")

def get_lm_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_lm_tp_group().world_size

def get_lm_tensor_model_parallel_rank():
    """Return world size for the tensor model parallel group."""
    return get_lm_tp_group().rank_in_group

def get_mlp_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_mlp_tp_group().world_size

def get_mlp_tensor_model_parallel_rank():
    """Return world size for the tensor model parallel group."""
    return get_mlp_tp_group().rank_in_group


def destroy_ascend_model_parallel():
    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None
