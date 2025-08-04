#integrate arg parser later
import cv2
import CVPedigreeAnalysis as cvped
from os import listdir,makedirs
import networkx as nx
from PedigreeDAGAnalysis import pedfile_readin, construct_pedigree_graph, longest_path_length, plot_pedigree_tree
from InheritanceModeAnalysis import classify_pedigree, trial_based_feature_threshold_determination

#ped file construction from pedigree 
FamilyIDs = []
avail_ped_images = listdir('data/Pedigree_Images/Raw_Images')
for ped_image in avail_ped_images:
    if ped_image.endswith('.png'):
        FamilyIDs.append(ped_image[:-4])

# for FamilyID in FamilyIDs:
#     line_img = cvped.pedigree_processing(FamilyID)
#     cv2.imshow(f'{FamilyID} lines', line_img)
# k = cv2.waitKey(0)
# cv2.destroyAllWindows

#pedfile import
ped_data_dict = {}
ped_sizes = set()
for FamilyID in FamilyIDs:
    ped_data_dict[FamilyID] = {}
    family_df = pedfile_readin(f'data/PedFiles/Automatic_Pedigrees/{FamilyID}auto.ped')
    family_dg = construct_pedigree_graph(family_df)
    ped_data_dict[FamilyID]['PedGraph'] = family_dg
    ped_sizes.add(longest_path_length(family_dg)+1)

#generate mode of inheritence thresholds
moi_thresholds_dict = {}
for ped_size in ped_sizes:
    moi_thresholds_dict[ped_size], _ = trial_based_feature_threshold_determination(generation_count = ped_size,
                                                                                   verbose= True)


#mode of inheritence analysis 
for FamilyID in FamilyIDs:
    family_dg = ped_data_dict[FamilyID]['PedGraph']
    family_dg_size = longest_path_length(family_dg) + 1
    ped_data_dict[FamilyID]['PredMode'] = classify_pedigree(PedGraph= family_dg,
                                                            thresholds_dict= moi_thresholds_dict[family_dg_size])
    print(f"{FamilyID}: {ped_data_dict[FamilyID]['PredMode']}")
