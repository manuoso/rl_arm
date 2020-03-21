import numpy as np

from cpprb.experimental import ReplayBuffer, PrioritizedReplayBuffer


# ----------------------------------------------------------------------------------------------------
def get_default_rb_dict(size, obs_shape, act_shape):
    return {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {
                "shape": obs_shape},
            "next_obs": {
                "shape": obs_shape},
            "act": {
                "shape": act_shape},
            "rew": {},
            "done": {}}}

# ----------------------------------------------------------------------------------------------------
def get_replay_buffer(obs_shape, act_shape, mem_capacity, use_prioritized_rb=False, on_policy=False, discrete=False):

    kwargs = get_default_rb_dict(mem_capacity, obs_shape, act_shape)

    # on-policy policy
    if on_policy:
        kwargs["size"] = mem_capacity
        kwargs["env_dict"].pop("next_obs")
        kwargs["env_dict"].pop("rew")
        # TODO: Remove done. Currently cannot remove because of cpprb implementation
        # kwargs["env_dict"].pop("done")
        kwargs["env_dict"]["logp"] = {}
        kwargs["env_dict"]["ret"] = {}
        kwargs["env_dict"]["adv"] = {}
        if discrete:
            kwargs["env_dict"]["act"]["dtype"] = np.int32
        return ReplayBuffer(**kwargs)
    
    if len(obs_shape) == 3:
        kwargs["env_dict"]["obs"]["dtype"] = np.ubyte
        kwargs["env_dict"]["next_obs"]["dtype"] = np.ubyte

    # prioritized
    if use_prioritized_rb:
        return PrioritizedReplayBuffer(**kwargs)

    return ReplayBuffer(**kwargs)
