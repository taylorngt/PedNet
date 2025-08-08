#integrate arg parser later
import cv2
import CVPedigreeAnalysis as cvped
import csv
from pprint import pprint
from os import listdir,makedirs
import networkx as nx
from PedigreeDAGAnalysis import pedfile_readin, construct_pedigree_graph, longest_path_length, plot_pedigree_tree
from InheritanceModeAnalysis import classify_pedigree, trial_based_feature_threshold_determination
from PedigreeDataGeneration import VarTableBackpadding
from SegregationScoring import pedigree_segregation_scoring, trial_based_segregation_scoring_weight_optimization
# Compile list of family IDs based on the available images
FamilyIDs = []
avail_ped_images = listdir('data/Pedigree_Images/Raw_Images')
for ped_image in avail_ped_images:
    if ped_image.endswith('.png'):
        FamilyIDs.append(ped_image[:-4])

# Process images into pedfiles using CVPedigreeAnalysis
# print('-------------------\n')
# print('IMAGE PROCESSING:')
# for FamilyID in FamilyIDs:
#     print(f' Processing {FamilyID}')
#     line_img = cvped.pedigree_processing(FamilyID)
#     cv2.imshow(f'{FamilyID} lines', line_img)
# k = cv2.waitKey(0)
# cv2.destroyAllWindows
print('\n-------------------\n')

# Import PedFiles and convert to DAGs
ped_data_dict = {}
ped_sizes = set()
for FamilyID in FamilyIDs:
    ped_data_dict[FamilyID] = {}
    family_df = pedfile_readin(f'data/PedFiles/Automatic_Pedigrees/{FamilyID}auto.ped')
    family_dg = construct_pedigree_graph(family_df)
    ped_data_dict[FamilyID]['PedGraph'] = family_dg

    size = longest_path_length(family_dg)+1
    ped_data_dict[FamilyID]['Size'] = size
    ped_sizes.add(size)

# Determine Optimum Mode of Inheritance Thresholds
moi_thresholds_dict = {}
for ped_size in ped_sizes:
    moi_thresholds_dict[ped_size], _ = trial_based_feature_threshold_determination(generation_count = ped_size,
                                                                                   verbose= False)


# Predict Modes of Inheritance
for FamilyID in FamilyIDs:
    family_dg = ped_data_dict[FamilyID]['PedGraph']
    family_dg_size = longest_path_length(family_dg) + 1
    ped_data_dict[FamilyID]['PredMode'] = classify_pedigree(PedGraph= family_dg,
                                                            thresholds_dict= moi_thresholds_dict[family_dg_size])
    print(f"{FamilyID}: {ped_data_dict[FamilyID]['PredMode']}")


#Extract and format known variant data
with open('data/ProbandInfo/GenotypeData.txt', 'r') as gtData:
    ListxProband = ([line.strip().split(' - ') for line in gtData])
    VariantxProbandDict = {}
    for proband_entry in ListxProband:
        proband = proband_entry[0]
        VariantxProbandDict[proband] = {}
        gene_list = proband_entry[1].split(', ')
        for gene_entry in gene_list:
            gene_name, attributes = gene_entry.split('=')
            VariantxProbandDict[proband][gene_name] = {}
            attribute_list = attributes.split(' ')
            for attribute in attribute_list:
                attribute_name, attribute_value = attribute.split(':')
                VariantxProbandDict[proband][gene_name][attribute_name] = int(attribute_value)
        
print('KNOWN VARIANT DATA:')
pprint(VariantxProbandDict)

print('\n------------------\n')

#Attach Proband ID and Variant Data to Pedigrees
with open('data/ProbandInfo/ProbandList.txt', 'r') as pl:
    ProbandList = ([line.strip() for line in pl])

probandless_families = []
print('PROBANDS FOUND:')
for FamilyID in FamilyIDs:
    family_members = ped_data_dict[FamilyID]['PedGraph'].nodes()
    proband_found = False
    for proband in ProbandList:
        if int(proband) in family_members:
            ped_data_dict[FamilyID]['Proband'] = proband
            proband_found = True
            print(f' {FamilyID}: {proband}')
            
            #generate VarTable if we have the known variant data
            if proband in VariantxProbandDict.keys():
                ped_data_dict[FamilyID]['VarTable'] = VarTableBackpadding(PedGraph= ped_data_dict[FamilyID]['PedGraph'],
                                                                          VariantInfoDict= VariantxProbandDict[proband],
                                                                          TotalNumberVariants= 6)
            
            break
    if not proband_found:
        ped_data_dict[FamilyID]['Proband'] = 'None'
        probandless_families.append(FamilyID)

print('\n------------------\n')

#Determine Optimal Weights for Segregation Scoring
OptimizedWeightsDict = {}
modes = ['AR', 'AD']
for mode in modes:
    OptimizedWeightsDict[mode] = {}
    for ped_size in ped_sizes:
        _, OptimizedWeightsDict[mode][ped_size], _ = trial_based_segregation_scoring_weight_optimization(
                                                        pedigree_count= 500,
                                                        Scoring_Method= 'Original',
                                                        Optimization_Method= 'Rank',
                                                        Mode= mode,
                                                        generation_count= ped_size,
                                                        Verbose= False)

#Segregation Scoring
sequenced_ped_data_dict = {}
for FamilyID in FamilyIDs:
    if 'VarTable' in ped_data_dict[FamilyID].keys():
        sequenced_ped_data_dict[FamilyID] = ped_data_dict[FamilyID]

for FamilyID in sequenced_ped_data_dict.keys():
    PedDict = sequenced_ped_data_dict[FamilyID]
    PredMode = PedDict['PredMode'] 
    if PredMode == 'Uncertain':
        PredMode = 'AR'
    Size = PedDict['Size']
    sequenced_ped_data_dict[FamilyID] = pedigree_segregation_scoring(
                                            Ped_Dict= PedDict,
                                            Scoring_Method= 'Original',
                                            Mode = PredMode,
                                            Weights= OptimizedWeightsDict[PredMode][Size],
                                            )


#Assessment of Segregation Scoring Accuracy
total = 0
correct = 0
incorrect = []
print('CORRECTLY SCORED PEDIGREES')
for FamilyID in sequenced_ped_data_dict.keys():
    if 'Original' in sequenced_ped_data_dict[FamilyID].keys():
        total += 1
        proband = sequenced_ped_data_dict[FamilyID]['Proband']
        LinkedVariant = list(VariantxProbandDict[proband].keys())[0]
        scores = sequenced_ped_data_dict[FamilyID]['Original']
        if max(scores, key= scores.get) == LinkedVariant:
            correct += 1
            print(f'\t{proband}')
        else:
            incorrect.append(proband)
print('\nINCORRECTLY SCORED PEDIGREES')
for i in incorrect:
    print(f'\t{i}')
print(f'\nTOTAL SCORING ACCURACY RATIO: {correct/total}')

print('\n------------------\n')