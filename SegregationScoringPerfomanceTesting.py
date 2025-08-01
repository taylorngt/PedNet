import pandas as pd
from PedigreeDataGeneration import pedigree_group_generator
from SegregationScoring import segregation_scoring_wrapper

# ---------------------------------------------------------------------
# SEGREGATION SCORING BY OPTIMIZATION AND SCORING METHOD
# ---------------------------------------------------------------------
def segregation_scoring_performance_test(Generation_Count,
                                         Pedigree_Count,
                                         Variant_Count,
                                         Mode= 'AD',
                                         Max_Children= 3,
                                         Known_Linked_Var= 'chr1:100000_A>T',
                                         Verbose= False,
                                         VarScore_Readout= False,
                                         SequenceCoverage= 0.5):

    Multi_Ped_Dict = pedigree_group_generator(pedigree_count= Pedigree_Count,
                                              generation_count= Generation_Count,
                                              max_children= Max_Children,
                                              n_bg= Variant_Count-1,
                                              mode= Mode,
                                              sequence_coverage= SequenceCoverage)
    Scoring_Modes = ['Original', 'Extended']
    Optimization_Methods = ['None', 'Margin', 'Rank']

    scoring_performance_results_dict = {
                                          'Optimization Method': Optimization_Methods,
                                          'Original': [],
                                          'Extended': []
                                        }

    for scoring_mode in Scoring_Modes:
        for optimization_method in Optimization_Methods:
            _, _, scoring_perfomance = segregation_scoring_wrapper(Multi_Ped_Dict= Multi_Ped_Dict,
                                                                    Scoring_Method= scoring_mode,
                                                                    Optimization_Method= optimization_method,
                                                                    Verbose= VarScore_Readout,
                                                                    Known_Linked_Var= Known_Linked_Var,
                                                                    Known_Mode= Mode)
            scoring_performance_results_dict[scoring_mode].append(scoring_perfomance)

    scoring_performance_results_df = pd.DataFrame.from_dict(scoring_performance_results_dict).set_index('Optimization Method')

    if Verbose:
        print(f'{Mode} Scoring Performance Results')
        scoring_performance_results_df

    #return scoring_performance_results_df

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
        Fam_Data = pedigree_group_generator(pedigree_count= Pedigree_Count,
                                            generation_count= Gen_Count,
                                            max_children= max_children,
                                            n_bg= Variant_Count-1,
                                            mode= Mode,
                                            sequence_coverage= SequenceCoverage)
        print(f'GENERATION COUNT= {Gen_Count} ')
        for Scoring_Method in ['Original', 'Extended']:
            _, _, Scoring_Accuracy = segregation_scoring_wrapper(Multi_Ped_Dict= Fam_Data,
                                                                  Scoring_Method= Scoring_Method,
                                                                  Optimization_Method= Optimization_Method,
                                                                  Known_Linked_Var= Known_Linked_Var,
                                                                  Known_Mode= Mode,
                                                                  Verbose= VarScore_Readout)
            pedigree_size_scoring_results_dict[Scoring_Method].append(Scoring_Accuracy)
        print()

    pedigree_size_scoring_results_df = pd.DataFrame.from_dict(pedigree_size_scoring_results_dict).set_index('Pedigree Size')
    if Verbose:
        if Optimization_Method == 'None':
            print(f'{Mode} Pedigree Unoptimized Size Scoring Results')
        else:
            print(f'{Mode} Pedigree {Optimization_Method} Optimized Size Scoring Results')
        pedigree_size_scoring_results_df

    #return pedigree_size_scroing_results_df