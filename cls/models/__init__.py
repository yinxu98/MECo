from .MultiEncoder import MultiEncoder
from .MultiPredictor import MultiPredictor
from .MultiEmbedding import MultiEmbedding
from .Classifier import Classifier

model_list = dict(
    MultiEncoder=MultiEncoder,
    MultiPredictor=MultiPredictor,
    MultiEmbedding=MultiEmbedding,
    Classifier=Classifier,
)


def build_model(cfg):
    return model_list.get(cfg.type)(cfg).cuda()
