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
from PedigreeDAGAnalysis import aff, construct_pedigree_graph
from SegregationScoring import raw_categorical_scoring


#################### Pedigree Generator ####################
'''
Generates a pedigree based on given metrics

'''
def pedigree_generator(max_children, FamilyID, mode, generation_count, alt_freq=0.15, SpouseLikelihood = 0.6, AffectedSpouse= True, BackpropLikelihood= 0.25):
        #-------------------------------------------
        # Helper Functions for Pedigree Propigation
        #-------------------------------------------

        '''
        Basic helper function to add new entry to pedigree dataframe
        '''
        def entry_generator(IndividualID, PaternalID, MaternalID, Sex, Phenotype, Genotype):
            nonlocal family_df
            family_df.loc[IndividualID] = [FamilyID, PaternalID, MaternalID, Sex, Phenotype, Genotype]

        '''
        Helper function to translate between genotype and phenotype
        Dependant on the mode of inheritance
        Input: genotype(int(0,1,2))
        Output: phenotype(int(1,2))
        '''
        def genotype_interpreter(genotype):
            if mode == 'AR':
                phenotype = 2 if genotype == 2 else 1
            if mode == 'AD':
                phenotype = 2 if genotype == 2 or genotype == 1 else 1
            return phenotype

        def calc_inheritance_weights(p,q):

            tt = q**4
            to = 2*p*(q**3)
            tz = (p**2)*(q**2)
            oo = 4*(p**2)*(q**2)
            oz = 2*(p**3)*(q)
            zz = p**4

            homoRef = p**2
            hetero = 2*p*q
            homoAlt = q**2

            inheritance_patterns = {
                'forward_genotypes': {
                #(paternal genotype, maternal genotype) -> [possible child genotypes]
                    (2,2): [2],
                    (2,1): [2,1],
                    (1,2): [2,1],
                    (2,0): [1],
                    (0,2): [1],
                    (1,1): [2,1,0],
                    (0,1): [1,0],
                    (1,0): [1,0],
                    (0,0): [0]
                },
                'forward_weights': {
                    (2,2): [1],
                    (2,1): [1,1],
                    (1,2): [1,1],
                    (2,0): [1],
                    (0,2): [1],
                    (1,1): [1,2,1],
                    (0,1): [1,1],
                    (1,0): [1,1],
                    (0,0): [1]
                },
                #child genotype -> [possible (paternal,maternal) genotypes]
                'reverse_genotypes': {
                    2: [(2,2),(2,1),(1,2),(1,1)],
                    1: [(2,1),(1,2),(2,0),(0,2),(1,1),(1,0),(0,1)],
                    0: [(1,0),(0,1),(0,0)]
                },
                'reverse_weights': {
                    2: [homoAlt**2, homoAlt*hetero, hetero*homoAlt, hetero**2],
                    1: [homoAlt*hetero, hetero*homoAlt, homoAlt*homoRef, homoRef*homoAlt, hetero**2, hetero*homoRef, homoRef*hetero],
                    0: [hetero*homoRef, homoRef*hetero, homoRef**2]
                }
            }

            return inheritance_patterns

        '''
        Wrapper function that generates the primary founder of the pedigree
        By default, this individual is affected
        If AD, 20% chance homozygous, 80% chance heterozygous.
        If AR, 100% chance homozygous.
        Input:
        Output:
        '''
        def primary_founder_generator():
            nonlocal family_df

            if mode == 'AD':
                Genotype = random.choices(population= [1,2],
                                          weights= (0.8, 0.2))[0]
            elif mode == 'AR':
                Genotype= 2

            entry_generator(IndividualID= 1,
                            PaternalID= 0,
                            MaternalID= 0,
                            Sex= random.randint(1,2),
                            Phenotype= 2,
                            Genotype= Genotype)
        '''
        Wrapper function that generates spouses unrelated to primary founder
        Spouse sex dependent on the relative of primary founder.
        Genotype and phenotype dependent on the mode of inheritance and affected spouse paramter.
        Input: relativeID(int)
        Ouput: n/a
        '''
        def spouse_generator(RelativeAnchorID):
            nonlocal family_df, alt_freq, ref_freq

            pp = ref_freq**2
            pq2 = 2*ref_freq*alt_freq
            qq = alt_freq**2

            Sex= 1 if family_df.loc[RelativeAnchorID]['Sex'] == 2 else 2

            if AffectedSpouse:
                Genotype= random.choices(population= [0,1,2],
                                          weights= (pp, pq2, qq),
                                          k=1)[0]

            else:
                Genotype = 0

            entry_generator(IndividualID= len(family_df)+1,
                            PaternalID= 0,
                            MaternalID= 0,
                            Sex= Sex,
                            Phenotype= genotype_interpreter(Genotype),
                            Genotype= Genotype)
        '''
        Wrapper function that generates an entry for the child of two given individuals.
        Child's genotype is chosen from list of allowed gentypes given parents genotypes with equal likelihood.
        Input: PaternalID(int), MaternalID(int)
        Output: n/a
        '''
        def child_generator(PaternalID, MaternalID):
            nonlocal family_df, inheritance_patterns

            parentalGenotype = (int(family_df.loc[PaternalID]['Genotype']), int(family_df.loc[MaternalID]['Genotype']))

            Genotype = random.choices(population= inheritance_patterns['forward_genotypes'][parentalGenotype],
                                      weights= inheritance_patterns['forward_weights'][parentalGenotype],
                                      k=1)[0]

            entry_generator(IndividualID= len(family_df)+1,
                            PaternalID= PaternalID,
                            MaternalID= MaternalID,
                            Sex= random.randint(1,2),
                            Phenotype= genotype_interpreter(Genotype),
                            Genotype= Genotype)
        #---------------------------------------
        # Primary Pedigree Contruction Functions
        #---------------------------------------
        '''
        Function that recursively constructs pedigree in backward direction.
        Infers ancestors of individuals unrelated to primary founder as they are added.
        Input: current_generation(int), RealativeAnchorID(int)
        Output: n/a
        '''
        def recursive_history_backprop(current_generation, RelativeAnchorID):
            nonlocal family_df, generation_count, inheritance_patterns, BackpropLikelihood

            BackpropRNG = random.randint(1,100)/100

            if current_generation > 0 and BackpropRNG <= BackpropLikelihood:

                GenotypeTup = random.choices(population= inheritance_patterns['reverse_genotypes'][family_df.loc[RelativeAnchorID]['Genotype']],
                                                    weights= inheritance_patterns['reverse_weights'][family_df.loc[RelativeAnchorID]['Genotype']],
                                                    k=1)[0]

                ID_list = ['PaternalID', 'MaternalID']

                for i in range(2):
                    entry_generator(IndividualID= len(family_df)+1,
                                    PaternalID= 0,
                                    MaternalID= 0,
                                    Sex= 1 + i,
                                    Phenotype= genotype_interpreter(GenotypeTup[i]),
                                    Genotype= GenotypeTup[i])
                    family_df.at[RelativeAnchorID, ID_list[i]] = len(family_df)
                    recursive_history_backprop(current_generation-1, len(family_df))

        '''
        Function that recursively constructs pedigree in forward direction.
        Input: current_generation(int), RelativeAnchorID(int)
        Output: n/a
        '''
        def recursive_pedigree_construction(current_generation, RelativeAnchorID):
            nonlocal family_df, max_children, generation_count

            if current_generation < generation_count-1:

                spouse_generator(RelativeAnchorID= RelativeAnchorID)

                #Determining Parental Sex for next generation
                if family_df.loc[RelativeAnchorID]['Sex'] == 1:
                    PaternalID = RelativeAnchorID
                    MaternalID = len(family_df)
                else:
                    PaternalID = len(family_df)
                    MaternalID = RelativeAnchorID

                if BackpropLikelihood:
                    recursive_history_backprop(current_generation, len(family_df))

                for child in range(random.randint(1, max_children)):
                    child_generator(PaternalID= PaternalID, MaternalID= MaternalID)
                    reproduction_rng = random.randint(1,100)/100
                    if reproduction_rng <= SpouseLikelihood:
                        recursive_pedigree_construction(current_generation+1, len(family_df))


        #-------------------------------------
        # 1. Construct the empty data frame
        #-------------------------------------
        pedigree_construction_columns = ['FamilyID', 'IndividualID', 'PaternalID', 'MaternalID', 'Sex', 'Phenotype', 'Genotype']
        family_df = pd.DataFrame(columns= pedigree_construction_columns)
        family_df.set_index('IndividualID', inplace=True)

        #-------------------------------------
        # 2. Generating Primary Founder
        #-------------------------------------
        primary_founder_generator()

        #--------------------------------------------
        # 3. Construct Inheritence Pattern Dictionary
        #--------------------------------------------
        ref_freq = 1 - alt_freq
        inheritance_patterns = calc_inheritance_weights(ref_freq, alt_freq)

        #----------------------------------------
        # 4. Generating Pedigree
        #----------------------------------------
        recursive_pedigree_construction(current_generation= 0, RelativeAnchorID= 1)

        #-------------------------------
        # 5. Resetign Standard Indexing
        #-------------------------------
        family_df.reset_index(inplace= True)

        return family_df

#################### Variant Table Generator ####################
'''
Generates a variant table with linked variant based on genotype taken from a given pedigree DAG

'''
def simulate_variant_table(family_df, sequencing_coverage, mode='AD', n_bg= 5, linked_variant = 'chr1:100000_A>T'):
    family_df.set_index('IndividualID', inplace=True)
    samples = list(family_df.index.values)

    #account for imcomplete sequencing coverage across a pedigree
    sequenced_samples = []
    for sample in samples:
        sequencing_rng = random.randint(1,100)/100
        if sequencing_rng <= sequencing_coverage:
            sequenced_samples.append(sample)
    
    VarTable = {}

    #filling out linked variant table entry based on generated genotype data
    VarTable[linked_variant] = {}
    for sample in sequenced_samples:
        VarTable[linked_variant][sample] = family_df[sample]['genotype']

    #filling in unlinked variant table entries using randomly generated gentypes
    # roughly based on an alternate allele frequency of 10%
    for i in range(n_bg):
        VarID = f'chr1:{100200+i}_G>C'
        VarTable[VarID] = {sample : random.choice([0,1,2],[0.8, 0.18,0.02])[0] for sample in sequenced_samples}
    
    return VarTable



def pedigree_group_generator(pedigree_count, mode, max_children, generation_count, sequencing_coverage = 0.75, n_bg= 5, alt_freq = 0):
    Fam_Data_Dict = {}
    for Family_Num in range(1, pedigree_count+1):
        FamilyID = f'FAM{Family_Num}'

        #for cases in which alt_frequency is not given (defaults are mode-dependent)
        #check to see how you can make this align with defaults of pedigree generation
        if not alt_freq:
          alt_freq = random.randint(2,8)/100 if mode == 'AD' else random.randint(5,20)/100

        QC_checks = 0
        ped_QC_pass = False
        while not ped_QC_pass:
            QC_checks += 1
            ped_df = pedigree_generator(max_children= max_children,
                                        FamilyID= FamilyID,
                                        mode= mode,
                                        generation_count= generation_count,
                                        alt_freq= alt_freq,
                                        BackpropLikelihood= random.choice([0.25,0.5,0.75]),
                                        AffectedSpouse= True)
            ped_dg = construct_pedigree_graph(ped_df)
            affected_nodes = aff(ped_dg)
            if len(affected_nodes) > 1 and len(ped_dg.nodes()) >= (generation_count * 2):
                ped_QC_pass = True
            elif QC_checks >= 50:
                print(f'{mode} {FamilyID}: Failed QC checks, included despite QC failure to prioritize futher operations')
                ped_QC_pass = True

        var_QC_pass = False
        while not var_QC_pass:
            var_dict = simulate_variant_table(family_df= ped_df,
                                              sequencing_coverage= sequencing_coverage,
                                              mode= mode,
                                              n_bg= n_bg)
            VarIDs = var_dict.keys()
            #checking to make sure there more than 2 seuqneced individuals in the pedigree
            #selects first varID from the list given they should have the same number of entries (number of sequenced samples)
            if len(var_dict[var_dict[0]].keys()) > 2:
                var_QC_pass = True

        #consider moving the categorical scoring to outside data generation
        #means you dont have to do unneccesary computation when generating multi-pedigree data sets outside of segregation scoring analysis
        cat_score_dict = {}
        for VarID in var_dict.keys():
            cat_score_dict[VarID] = raw_categorical_scoring(G= ped_dg,
                                                            gt= var_dict[VarID])

        Fam_Data_Dict[FamilyID] = {'PedGraph': ped_dg, 'VarTable': var_dict, 'CategoricalScores': cat_score_dict}
    return Fam_Data_Dict