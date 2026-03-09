"""histopath-qc: quality control filters for H&E patches."""

from .blur_detector import BlurDetector
from .entropy_filter import EntropyFilter
from .tissue_detector import TissueDetector
from .batch_qc import PatchQC, QCResult, BatchQCReport

__version__ = "0.1.0"
__all__ = ["BlurDetector", "EntropyFilter", "TissueDetector", "PatchQC", "QCResult", "BatchQCReport"]
