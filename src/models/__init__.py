from .bilstm          import BiLSTMClassifier
from .tcn             import TCNClassifier
from .transformer_enc import TransformerClassifier
from .stgcn           import STGCNClassifier
from .gat_transformer import GATTransformerClassifier

MODEL_REGISTRY = {
    "bilstm":       BiLSTMClassifier,
    "tcn":          TCNClassifier,
    "transformer":  TransformerClassifier,
    "stgcn":        STGCNClassifier,
    "gat":          GATTransformerClassifier,
}
