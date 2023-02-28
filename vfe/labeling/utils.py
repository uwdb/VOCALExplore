from .oracle_labeler import OracleLabeler
from .randomsample_coordinator import RandomSampleCoordinator
from .kmeans_coordinator import GreedyKMeansCoordinator, CenterKMeansCoordinator
from .activelearning_coordinator import *
from .coresets_coordinator import *
from .softkmeans_coordinator import *
from .lda_coordinator import *
from .incremental_lda_coordinator import *
from .cleaning_coordinator import *
from .snuba_coordinator import *

def get_labeler(labeler_name, **kwargs):
    if labeler_name.lower() == OracleLabeler.name().lower():
        return OracleLabeler(**kwargs)
    else:
        raise RuntimeError(f'Unrecognized labeler_name: {labeler_name}')

def get_coordinator(coordinator_name, **kwargs):
    coordinator_name = coordinator_name.lower()
    clses = [
        RandomSampleCoordinator,
        GreedyKMeansCoordinator,
        CenterKMeansCoordinator,
        LinearUncertaintyALCoordinator,
        QDAUncertaintyALCoordinator,
        CoresetsALCoordinator,
        SizeKMeansSoftLabelCoordinator,
        DistanceKMeansSoftLabelCoordinator,
        DistanceGMMSoftLabelCoordinator,
        UncertaintyKMeansSoftLabelCoordinator,
        UncertaintyGMMSoftLabelCoordinator,
        DisagreementKMeansLDACoordinator,
        DistanceKMeansLDACoordinator,
        DisagreementMultiPointKMeansLDACoordinator,
        DisagreementFarthestKMeansLDACoordinator,
        DisagreementLDAALKMeansLDACoordinator,
        IncorrectLDAALKMeansLDACoordinator,
        DisagreementIncrementalLDACoordinator,
        DisagreementIncrementalOAPLDACoordinator,
        DisagreementIncrementalOAPOLPLDACoordinator,
        DisagreementOAPLDACoordinator,
        BasicCleaningLDACoordinator,
        CheckOracleCleaningLDACoordinator,
        CheckOracle50CleaningLDACoordinator,
        CheckOracle50CleaningLDAWithLabeledCoordinator,
        BasicCleaningLMLDACoordinator,
        CheckOracleCleaningLMLDACoordinator,
        CheckOracle50CleaningLMLDACoordinator,
        DistanceKMeansSoftLabelTrainingCoordinator,
        DistanceKMeansLDATrainerCoordinator,
        BaseSnubaCoordinator,
        LancetSnubaCoordinator,
    ]
    for coordinator_cls in clses:
        if coordinator_name == coordinator_cls.name().lower():
            return coordinator_cls(**kwargs)

    raise RuntimeError(f'Unrecognized coordinator_name: {coordinator_name}')
