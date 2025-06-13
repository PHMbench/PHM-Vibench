from .prototypical_network import task as PrototypicalNetworkTask
from .knn_feature import task as KNNFeatureTask
from .matching_network import task as MatchingNetworkTask
from .finetuning import task as FinetuningTask

# You can add other FewShot tasks here as they are implemented
# from .another_fewshot_method import task as AnotherFewShotTask

__all__ = [
    'PrototypicalNetworkTask',
    'KNNFeatureTask',
    'MatchingNetworkTask',
    'FinetuningTask'
    # 'AnotherFewShotTask'
]