import numpy as np
import bittensor as bt
from typing import List
from loguru import logger


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False

    # Filter out non validator permit.
    if metagraph.validator_permit[uid]:

        # Filter out miners who are validators
        if metagraph.S[uid] >= vpermit_tao_limit:
            return False
        
        # Filter out uid without IP.
        if metagraph.neurons[uid].axon_info.ip == '0.0.0.0':
            return False
    # Available otherwise.
    return True

async def get_top_miner_uids(metagraph: "bt.metagraph.Metagraph",
                             blacklisted_axon_ids: List[int] = [],
                             top_rate: float = 1,
                             vpermit_tao_limit: int = 4096) -> [np.int64]:

    """Returns the available top miner UID from the metagraph.
    Args:
        metagraph (bt.metagraph.Metagraph): Metagraph object
        top_rate (float): The fraction of top nodes to consider based on stake. Defaults to 1.
        vpermit_tao_limit (int): Validator permit tao limit
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        top_miner_uid (np.int64): The top miner UID.
    """

    try:
        miner_candidate_uids = []
        for uid in range(metagraph.n.item()):
            uid_is_available = check_uid_availability(
                metagraph, uid, vpermit_tao_limit
            )

            if uid_is_available:
                if uid not in blacklisted_axon_ids:
                    miner_candidate_uids.append(uid)

        ips = []
        miner_ip_filtered_uids = []
        for uid in miner_candidate_uids:
            if metagraph.axons[uid].ip not in ips:
                ips.append(metagraph.axons[uid].ip)
                miner_ip_filtered_uids.append(uid)

        # Consider both of incentive and trust score
        values = [(uid, metagraph.I[uid] * metagraph.trust[uid]) for uid in miner_ip_filtered_uids]
        sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
        top_rate_num_items = max(1, int(top_rate * len(miner_ip_filtered_uids)))
        top_miner_uids = np.array([uid for uid, _ in sorted_values[:top_rate_num_items]])
        return top_miner_uids
    except Exception as e:
        logger.error(f"Failed to get top miner uids", error = {'exception_type': e.__class__.__name__,'exception_message': str(e),'exception_args': e.args})
        return None
