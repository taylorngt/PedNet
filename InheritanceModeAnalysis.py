import random
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import accuracy_score, auc
import pprint
import numpy as np
from collections import OrderedDict
import powerlaw
import itertools as it
from typing import Dict, Set, Tuple
from scipy.optimize import minimize
import copy
from pprint import pprint
from PedigreeDAGAnalysis import generations, aff, construct_pedigree_graph, pedigree_features, graph_metrics, longest_path_length
from PedigreeDataGeneration import pedigree_generator

############### GLOBAL DEFAULT PEDIGREE PARAMETERS ##################
TRIAL_COUNT = 500
MAX_CHILDREN = 5
ALT_FREQ_RANGE = (2, 20)
BACKPROP_LIKELIHOOD_RANGE = (25, 75)
SPOUSE_LIKELIHOOD_RANGE = (25, 75)
AFFECTED_SPOUSE = True

############### GLOBAL DEFAULT MoI PARAMETERS #####################
ACCURACY_THRESHOLD = 0.7

def trial_based_feature_threshold_determination(
                generation_count,
                trial_count= TRIAL_COUNT,
                max_children= MAX_CHILDREN,
                alt_freq_range= ALT_FREQ_RANGE,
                accuracy_threshold = ACCURACY_THRESHOLD,
                BackpropLikelihoodRange = BACKPROP_LIKELIHOOD_RANGE,
                SpouseLikelihoodRange= SPOUSE_LIKELIHOOD_RANGE,
                AffectedSpouse = AFFECTED_SPOUSE,
                verbose = False,
                size_agnostic = False,):
    '''
    Determines optimal inheritence pattern determination thresholds for pedigrees of given generation count
    based on a given number of randomly generated trial pedigrees
    '''

    def trail_results_df_generation():
        nonlocal generation_count, trial_count, alt_freq_range, BackpropLikelihoodRange, SpouseLikelihoodRange, max_children, size_agnostic

        all_trial_pedigree_features = pd.DataFrame()

        for trialID in range(1, trial_count+1):

            FamilyID = 'TestFam' + str(trialID)
            actual_mode = random.choice(['AD', 'AR'])

            #Accounting for cases where we want thresholds that are not specific to a generation count
            if size_agnostic:
                #Run time seems to increase indefinitely if left to be size_agnostic so currently unusable feature
                trial_generation_count = random.randint(3, generation_count)
            else:
                trial_generation_count = generation_count

            QC_pass = False
            while not QC_pass:
                trial_df = pedigree_generator(
                                            FamilyID= FamilyID,
                                            mode= actual_mode,
                                            max_children= max_children,
                                            generation_count= trial_generation_count,
                                            BackpropLikelihoodRange= BackpropLikelihoodRange,
                                            SpouseLikelihoodRange= SpouseLikelihoodRange,
                                            alt_freq_range= alt_freq_range,
                                            AffectedSpouse= AffectedSpouse)

                trial_dg = construct_pedigree_graph(trial_df)

                affecteded_nodes = aff(trial_dg)
                if len(affecteded_nodes) > 1 and len(trial_dg.nodes()) > (generation_count * 2) - 1:
                    QC_pass = True


            trial_feat_met_dict = {**pedigree_features(trial_dg), **graph_metrics(trial_dg)}
            trial_feat_met_dict['actual_mode'] = actual_mode
            trial_feat_met_df = pd.DataFrame(trial_feat_met_dict, index= [0])

            all_trial_pedigree_features = pd.concat(objs= [all_trial_pedigree_features, trial_feat_met_df], ignore_index=True)

        return all_trial_pedigree_features




    def ROC_param_calc(true_labels, predicted_labels):
        real_pos_count = 0
        real_neg_count = 0
        true_pos_count = 0
        false_pos_count = 0

        for i in range(len(true_labels)):
            if true_labels[i] == 'AD':
                real_pos_count += 1
                if predicted_labels[i] == 'AD':
                    true_pos_count += 1
            elif true_labels[i] == 'AR':
                real_neg_count += 1
                if predicted_labels[i] == 'AD':
                    false_pos_count += 1


        TPR = true_pos_count/real_pos_count
        FPR = false_pos_count/real_neg_count

        return TPR, FPR

    def AUC_calc(FPR_scores, TPR_scores):
        FPR_arr = np.array(FPR_scores)
        TPR_arr = np.array(TPR_scores)

        sort_indx = np.argsort(FPR_arr)
        FPR_arr = FPR_arr[sort_indx]
        TPR_arr = TPR_arr[sort_indx]

        auc_score = auc(FPR_arr, TPR_arr)

        return auc_score

    def ROC_plot(features, TPR_score_dict, FPR_score_dict):

        fig = plt.figure()
        ax = plt.subplot(111)

        for feature in features:
            AUC_score = AUC_calc(FPR_scores= FPR_score_dict[feature],
                                 TPR_scores= TPR_score_dict[feature])
            ax.plot(FPR_score_dict[feature], TPR_score_dict[feature],
                    label= f'{feature} = {AUC_score:.2f}')

        ax.plot([0,1], [0,1], linestyle='--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Mode of Inheritance ROC')
        ax.grid(True)

        box= ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  ncol= 2, fancybox=True, shadow=True)
        plt.show()


    def single_feature_threshold_determination(feature_values, actual_mode_labels):
        min_value = min(feature_values)
        max_value = max(feature_values)
        thresh_increment = (max_value - min_value)/100
        min_value = min_value - thresh_increment
        threshold_options = [min_value+(thresh_increment*i) for i in range(103)]

        best_threshold = None
        best_accuracy = 0
        best_direction = None

        #Test accuracy of each threshold (both as upper and lower limit of AD classification) and store accuracy score
        TPR_scores = []
        FPR_scores = []
        for threshold in threshold_options:
            greater_equal_predictions = ['AD' if value > threshold else 'AR' for value in feature_values]
            less_predictions = ['AD' if value <= threshold else 'AR' for value in feature_values]

            greater_equal_accuracy = accuracy_score(actual_mode_labels, greater_equal_predictions)
            less_accuracy = accuracy_score(actual_mode_labels, less_predictions)

            if greater_equal_accuracy > best_accuracy:
                best_accuracy = greater_equal_accuracy
                best_threshold = threshold
                best_direction = 'greater'
            elif less_accuracy > best_accuracy:
                best_accuracy = less_accuracy
                best_threshold = threshold
                best_direction = 'less_equal'

            TPR, FPR = ROC_param_calc(actual_mode_labels, greater_equal_predictions)
            TPR_scores.append(TPR)
            FPR_scores.append(FPR)

        return best_threshold, best_direction, best_accuracy, TPR_scores, FPR_scores

    accuracy_checks = 0
    max_accuracy_checks = 5
    accuracy_QC_pass = False
    while not accuracy_QC_pass and accuracy_checks < max_accuracy_checks:
        accuracy_checks += 1
        trial_features_df = trail_results_df_generation()
        training_features_df = trial_features_df.sample(frac=0.8)
        testing_features_df = trial_features_df.drop(training_features_df.index)



        TPR_scores_dict = {}
        FPR_scores_dict = {}
        thresholds_dict = {}
        for feature in trial_features_df.columns.values:
            if feature == 'FamID' or feature == 'actual_mode':
                continue
            threshold, direction, accuracy, TPR_scores, FPR_scores = single_feature_threshold_determination(training_features_df[feature].values,
                                                                                                            training_features_df['actual_mode'].values)
            thresholds_dict[feature] = {'threshold': threshold, 'direction': direction, 'accuracy': accuracy}
            TPR_scores_dict[feature] = TPR_scores
            FPR_scores_dict[feature] = FPR_scores

        mode_prediction_field = []
        for _,row in testing_features_df.iterrows():
            predicted_mode = MoI_classification(sample = row,
                                                                thresholds_dict = thresholds_dict,
                                                                accuracy_threshold= accuracy_threshold)
            mode_prediction_field.append(predicted_mode)
        testing_features_df['predicted_mode'] = mode_prediction_field

        overall_classification_accuracy = accuracy_score(y_true= testing_features_df['actual_mode'],
                                                         y_pred= testing_features_df['predicted_mode'])

        certain_test_results_df = testing_features_df[testing_features_df['predicted_mode']!='Uncertain']
        num_certain_results = len(certain_test_results_df)
        certain_classification_accuracy = accuracy_score(y_true= certain_test_results_df['actual_mode'],
                                                         y_pred= certain_test_results_df['predicted_mode'])

        if certain_classification_accuracy >= accuracy_threshold and num_certain_results/len(testing_features_df) >= accuracy_threshold:
            accuracy_QC_pass = True
        else:
            accuracy_checks += 1



    certainty_ratio = num_certain_results/len(testing_features_df)
    if verbose:
        ROC_plot(features= thresholds_dict.keys(),
                 TPR_score_dict= TPR_scores_dict,
                 FPR_score_dict= FPR_scores_dict)
        print(f'Number of Certain Results: {certainty_ratio}')
        print(f'Certain Classification Accuracy: {certain_classification_accuracy}')
        print(f'Overall Classification Accuracy: {overall_classification_accuracy}')

    accuracy_metrics = {'certainty_ratio': certainty_ratio,
                        'certain_class_acc': certain_classification_accuracy,
                        'overall_class_acc': overall_classification_accuracy}

    return thresholds_dict, accuracy_metrics







def MoI_classification(
            sample,
            thresholds_dict,
            accuracy_threshold= 0.7,
            ) -> str:
    #checking to see if input is pedigree graph for normal classification or df row for threshold determination calculations
    if isinstance(sample, nx.Graph):
        sample_features = {**pedigree_features(sample), **graph_metrics(sample)}
    elif isinstance(sample, dict):
        sample_features = sample
    else:
        raise TypeError("Input must either be a dictionary with pre-calculated pedigree features or a DAG representation of a pedigree")

    votes= 0
    total= 0
    for feature, descriptors in thresholds_dict.items():
        threshold = descriptors['threshold']
        direction = descriptors['direction']
        accuracy = descriptors['accuracy']
        feature_value = sample_features[feature]

        if accuracy >= accuracy_threshold:
            total += 1
            if direction == 'greater':
                if feature_value > threshold:
                    votes += 1
            elif direction == 'less_equal':
                if feature_value <= threshold:
                    votes += 1

    if total == 0:
        return 'Uncertain'
    elif votes/total > 0.75:
        return 'AD'
    elif votes/total < 0.25:
        return 'AR'
    else:
        return 'Uncertain'

# def classify_pedigree(PedGraph, thresholds_dict, accuracy_threshold = 0.7) -> str:

#     pedigree_feats_mets = 

#     return inheritance_pattern_classification(sample_features= pedigree_feats_mets,
#                                               thresholds_dict= thresholds_dict,
#                                               accuracy_threshold= accuracy_threshold)




# def classify_multiple_pedigrees(Multi_Ped_Dict: dict, thresholds_dict= 0, same_size= True, Verbose= False):
#     if same_size:
#         if not thresholds_dict:
#             threshold_basis_graph = random.choice(list(Multi_Ped_Dict.values()))['PedGraph']
#             thresholds_dict, _ = trial_based_feature_threshold_determination(generation_count= longest_path_length(threshold_basis_graph)+1)
#         for FamilyID in Multi_Ped_Dict.keys():
#             FamilyDG = Multi_Ped_Dict[FamilyID]['PedGraph']
#             Multi_Ped_Dict[FamilyID]['PredMode'] = classify_pedigree(G= FamilyDG, thresholds_dict= thresholds_dict)
#     else:
#         pedigree_sizes = set()
#         for FamilyID in Multi_Ped_Dict.keys():
#             FamilyDG = Multi_Ped_Dict[FamilyID]['PedGraph']
#             pedigree_sizes.add(longest_path_length(FamilyDG)+1)
#         thresholds_2d_dict = {}

#         for pedigree_size in pedigree_sizes:
#             thresholds_2d_dict[pedigree_size], _ = trial_based_feature_threshold_determination(generation_count= pedigree_size, verbose= Verbose)

#         for FamilyID in Multi_Ped_Dict.keys():
#             FamilyDG = Multi_Ped_Dict[FamilyID]['PedGraph']
#             Multi_Ped_Dict[FamilyID]['PredMode'] = classify_pedigree(PedGraph= FamilyDG, thresholds_dict= thresholds_2d_dict[longest_path_length(FamilyDG)+1])

#     return Multi_Ped_Dict

# def pedigree_group_mode_agreement(Multi_Ped_Dict: dict):
#     '''
#     Returns the mutliple pedigree data file with updated predicted modes as well as the
#     most prevelant inheritance mode classification found in the predicted modes
#     '''
#     Multi_Ped_Dict = classify_multiple_pedigrees(Multi_Ped_Dict)
#     mode_lst = [Multi_Ped_Dict[FamilyID]['PredMode'] for FamilyID in Multi_Ped_Dict.keys()]
#     agreed_mode = max(set(mode_lst), key= mode_lst.count)
#     return Multi_Ped_Dict, agreed_mode