from sklearn.cluster import DBSCAN
import numpy as np
from loguru import logger


def dbscan(request, eps=10):
    token_usages_completion = [token_usage['completion_tokens'] for token_usage in request.token_usages]
    token_usages_prompt = [token_usage['prompt_tokens'] for token_usage in request.token_usages]
    token_usages_total = [token_usage['total_tokens'] for token_usage in request.token_usages]

    # TODO: use logger instead
    logger.info(f'token_usages_completion {token_usages_completion}')
    logger.info(f'token_usages_prompt {token_usages_prompt}')
    logger.info(f'token_usages_total {token_usages_total}')

    completion_index = _dbscan(token_usages_completion, eps)
    logger.info(f'completion_index {completion_index}')
    prompt_index = _dbscan(token_usages_prompt, eps)
    logger.info(f'prompt_index {prompt_index}')
    total_index = _dbscan(token_usages_total, eps)
    logger.info(f'total_index {total_index}')

    final_index = completion_index * prompt_index * total_index
    logger.info(f'----------------------------------{final_index}')

    # return np.array(entries)[final_index].tolist(), np.array(entries)[~final_index].tolist()
    return [
        (request.miner_hotkeys[i], request.query_start_times[i], request.execution_times[i], request.token_usages[i])
        for i in range(len(final_index)) if final_index[i] == True]


def _dbscan(data, eps):
    # Example data
    data = np.array(data)

    # Reshape data for clustering
    data = data.reshape(-1, 1)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=1)
    clusters = dbscan.fit_predict(data)

    # Find the largest cluster
    unique, counts = np.unique(clusters, return_counts=True)
    largest_cluster_index = unique[np.argmax(counts)]

    return np.array(clusters == largest_cluster_index)
