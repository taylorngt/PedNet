import pandas as pd
from SegregationScoring import trial_based_segregation_scoring_weight_optimization
from pprint import pprint

# ---------------------------------------------------------------------
# SEGREGATION SCORING BY OPTIMIZATION AND SCORING METHOD
# ---------------------------------------------------------------------
Inheritence_Modes = ['AD', 'AR']
Scoring_Methods = ['Original', 'Extended']
Optimization_Methods = ['None', 'Rank']

for Mode in Inheritence_Modes:

    scoring_performance_results_dict = {
                                        'Optimization Method': Optimization_Methods,
                                        'Original': [],
                                        'Extended': []
                                    }

    for Scoring_Method in Scoring_Methods:
        for Optimization_Method in Optimization_Methods:
            test_pedigree_data, optimized_weights, scoring_perfomance = trial_based_segregation_scoring_weight_optimization(
                                                                                trial_count= 100,
                                                                                Scoring_Method= Scoring_Method,
                                                                                Optimization_Method= Optimization_Method,
                                                                                Mode= Mode,
                                                                                Verbose= False,
                                                                                )

            scoring_performance_results_dict[Scoring_Method].append(scoring_perfomance)


    scoring_performance_results_df = pd.DataFrame.from_dict(scoring_performance_results_dict).set_index('Optimization Method')



    print(f'\n{Mode} SCORING PERFORMANCE SUMMARY')
    print(scoring_performance_results_df)


# ---------------------------------------------------------------------
# SEGREGATION SCORING PERFORMANCE BY PEDIGREE SIZE AND SCORING METHOD
# ---------------------------------------------------------------------
def pedigree_size_performance_test(Generation_Range,
                                   Pedigree_Count,
                                   Variant_Count,
                                   Optimization_Method,
                                   Mode,
                                   max_children= 3,
                                   Known_Linked_Var= 'chr1:100000_A>T',
                                   Verbose= False,
                                   VarScore_Readout= False,
                                   SequenceCoverage= 0.5):
    pedigree_size_scoring_results_dict = {
        'Pedigree Size': [],
        'Original': [],
        'Extended': []
    }
    min_gen = Generation_Range[0]
    max_gen = Generation_Range[1]
    for i in range(min_gen, max_gen+1):
        pedigree_size_scoring_results_dict['Pedigree Size'].append(i)

    for Gen_Count in range(min_gen, max_gen+1):
        print(f'GENERATION COUNT= {Gen_Count} ')
        for Scoring_Method in ['Original', 'Extended']:
            _, _, Scoring_Accuracy = trial_based_segregation_scoring_weight_optimization(trial_count= Pedigree_Count,
                                                                                         Scoring_Method= Scoring_Method,
                                                                                         Optimization_Method= Optimization_Method,
                                                                                         Verbose= VarScore_Readout,
                                                                                         Known_Linked_Var= Known_Linked_Var,
                                                                                         Mode= Mode,
                                                                                         max_children= max_children, 
                                                                                         generation_count= Gen_Count,
                                                                                         sequencing_coverage= SequenceCoverage,
                                                                                         n_bg= Variant_Count-1)
            pedigree_size_scoring_results_dict[Scoring_Method].append(Scoring_Accuracy)
        print()

    pedigree_size_scoring_results_df = pd.DataFrame.from_dict(pedigree_size_scoring_results_dict).set_index('Pedigree Size')

    if Verbose:
        if Optimization_Method == 'None':
            print(f'{Mode} Pedigree Unoptimized Size Scoring Results')
        else:
            print(f'{Mode} Pedigree {Optimization_Method} Optimized Size Scoring Results')
        
        print(pedigree_size_scoring_results_df)

    return pedigree_size_scoring_results_df


#pedigree_size_performance_test(Generation_Range=(3,4), Mode= 'AR', Pedigree_Count=500, Variant_Count= 5, Verbose= True, Optimization_Method= 'None')