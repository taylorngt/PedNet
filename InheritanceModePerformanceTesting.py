from InheritanceModeAnalysis import trial_based_feature_threshold_determination
from pprint import pprint

MoIvsPedSize_ThreshResults = {}
MoIvsPedSize_AccResults = {}
for pedigree_size in range(3,5):
  ThreshResult, AccResult  = trial_based_feature_threshold_determination(
                                                    generation_count= pedigree_size,
                                                    trial_count= 500,
                                                    verbose= True
                            )
  MoIvsPedSize_ThreshResults[pedigree_size] = ThreshResult
  MoIvsPedSize_AccResults[pedigree_size] = AccResult

print('ACCURACY RESULTS')
pprint(MoIvsPedSize_AccResults)
print('THRESHOLD RESULTS')
pprint(MoIvsPedSize_ThreshResults)