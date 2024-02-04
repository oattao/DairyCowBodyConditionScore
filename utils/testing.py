import os
import glob
import json
import pandas as pd


def get_ground_truth(ground_truth_data, cow_regno, score_column, feature_column=None):
    df = ground_truth_data[ground_truth_data['cow_regno'] == cow_regno]
    if len(df) == 0:
        print("No data for: ", cow_regno)
        return '', None

    score = df[score_column].values[0]
    feature = df[feature_column].values[0] if feature_column is not None else None
    return score, feature

def make_result_dataframe(samples, features, scores, ground_truth_data, true_score_col, true_feature_col=None):
    results = pd.DataFrame(dict(zip(['cow_regno', 'Measure Score', 'Measure Feature'], [samples, scores, features])))
    results[['KAIA Score', 'KAIA Feature']] = results['cow_regno'].apply(lambda cow_regno: pd.Series(get_ground_truth(ground_truth_data, cow_regno, true_score_col, true_feature_col)))
    return results

def refine_list_test(list_pcd, previous_output_folder):
    list_already = os.listdir(previous_output_folder)
    list_already = [name.split('.')[0] for name in list_already if name.endswith('.ply')]
    refined_list = []
    for path_pcd in list_pcd:
        name = os.path.basename(path_pcd).split('.')[0]
        if name not in list_already:
            refined_list.append(path_pcd)
    return refined_list

def get_marker(config_file_path):
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    marker = config['marker']
    return marker