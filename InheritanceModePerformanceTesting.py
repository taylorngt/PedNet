from InheritanceModeAnalysis import metric_thresholds_determination
from pprint import pprint
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ---------------------------------------------------------------------
# BENCHMARKING CONFIGURATIONS
# ---------------------------------------------------------------------
NUMBER_TRIALS = 30
NUMBER_PEDIGREES = 1000


# ---------------------------------------------------------------------
# PREDICTION EVALUATION
# ---------------------------------------------------------------------
def evaluate_predictions(true_mode_array, predicted_mode_array):
    certain_prediction_count = len([prediction for prediction in predicted_mode_array if prediction != 'Uncertain'])

    return {
      #General Accuracy Evaluation
      'accuracy' : accuracy_score(true_mode_array, predicted_mode_array),
      'certainty' : certain_prediction_count / len(predicted_mode_array),
      'precision' : precision_score(true_mode_array, predicted_mode_array, labels = ['AD', 'AR'], average= 'micro'),
      'recall': recall_score(true_mode_array, predicted_mode_array, labels = ['AD', 'AR'], average= 'micro'),
      'F1': f1_score(true_mode_array, predicted_mode_array, labels = ['AD', 'AR'], average= 'micro'),

      #Mode-specific Precision Analysis
      'precision_AD' : precision_score(true_mode_array, predicted_mode_array, pos_label= 'AD'),
      'precision_AD' : precision_score(true_mode_array, predicted_mode_array, pos_label= 'AR')
    }

# ---------------------------------------------------------------------
# MAIN BENCHMARKING LOOP
# ---------------------------------------------------------------------
trial_classification_results = []
trial_evaluated_threshold_results = []
for trial in range(NUMBER_TRIALS):
    classification_results, threshold_results = metric_thresholds_determination(
                                                    pedigree_count= NUMBER_PEDIGREES,
                                                    generation_range= (3,3)
                                                    )
    classification_results_df = pd.DataFrame(classification_results)
    trial_prediction_evals = evaluate_predictions(
                                  true_mode_array= classification_results_df['TrueMode'],
                                  predicted_mode_array = classification_results_df['PredictedMode']
    )
  
    evaluated_threshold_results = {**{'TrialID':trial}, **trial_prediction_evals}
    #expanding threshold descriptor terms from ROC into fields for evaluated threshold result table construction
    for metric, descriptors in threshold_results.items():
      for descriptor in descriptors.keys():
        evaluated_threshold_results[f'{metric} {descriptor}'] = threshold_results[metric][descriptor]
    
    trial_evaluated_threshold_results.append(evaluated_threshold_results)
    
    
    for pedigree_entry in classification_results:
      pedigree_entry['TrialID'] = trial
    trial_classification_results.extend(classification_results)
    


# ---------------------------------------------------------------------
# RESULT EXPORT
# ---------------------------------------------------------------------
classification_df = pd.DataFrame(trial_classification_results)
classification_df.to_csv('MOI_Classification_Benchmark_Results.csv', index= False)
print(f'MOI Classification benchmark results save: MOI_Classification_Benchmark_Results.csv')

threshold_df = pd.DataFrame(trial_evaluated_threshold_results)
threshold_df.to_csv('MOI_Threshold_Results.csv', index= False)
print(f'MOI Threshold results save: MOI_Threshold_Results.csv')

