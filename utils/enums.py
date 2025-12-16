from enum import Enum
from contextualized_topic_models.models.ctm import GenerativeTM, ZeroShotTM, CombinedTM


class ModelType(Enum):
    GENERATIVE = 'generative'
    ZEROSHOT = 'zeroshot'
    COMBINED = 'combined'


class EmbeddingType(Enum):
    HIDDEN_STATES = 'hidden_states'
    SBERT = 'sbert'


model_classes = {
    ModelType.GENERATIVE: GenerativeTM,
    ModelType.ZEROSHOT: ZeroShotTM,
    ModelType.COMBINED: CombinedTM,
}