# To install hloc, see: https://github.com/cvg/Hierarchical-retrivalization
from lighthloc import extract_features, match_features, reconstruction
from lighthloc.associators import pairs_from_retrieval, pairs_from_exhaustive, pairs_from_sequance
from pathlib import Path
import click

mapper_confs = {
    'default' : {'ba_refine_principal_point': 1,},
    'fast' : {'ba_global_max_num_iterations': 20, "ba_global_max_refinements":1, 
              "ba_global_points_freq":200000}
}

@click.command()
@click.option('--data', type=str, help='Path to data directory')
@click.option('--match-type', default='retrival', help='Type of matching to perform (default: retrival)', type=click.Choice(['exhaustive', 'sequential', 'retrival']))
@click.option('--feature-type', default='superpoint_inloc', help='Type of feature extraction (default: superpoint_inloc)', type=click.Choice(['superpoint_inloc', 'superpoint_aachen']))
@click.option('--matcher-type', default='lightglue', help='Type of feature matching (default: lightglue)', type=click.Choice(['lightglue', 'lightglue_trt', 'superglue']))
@click.option('--mapper-type', default='default', help='Type of mapper (default: default)', type=click.Choice(['default', 'fast']))
def main(data, match_type, feature_type, matcher_type, mapper_type):
    images = Path(data) / 'images/'
    outputs = Path(data)

    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'sparse' / '0'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'

    feature_conf = extract_features.confs[feature_type]
    matcher_conf = match_features.confs[matcher_type]
    assert match_type in ['exhaustive', 'sequential', 'retrival']
    if match_type == 'exhaustive':
        references = [p.relative_to(images).as_posix() for p in images.iterdir()]
        print(len(references), "mapping images")
        extract_features.main(feature_conf, images, image_list=references, feature_path=features)
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references, image_options={'camera_model': 'OPENCV'}, mapper_options=mapper_confs[mapper_type])
    elif match_type == 'sequential':
        references = [p.relative_to(images).as_posix() for p in images.iterdir()]
        pairs_from_sequance.main(sfm_pairs, image_list=references, overlap=10, quadratic_overlap=True)
        extract_features.main(feature_conf, images, image_list=references, feature_path=features)
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references, image_options={'camera_model': 'OPENCV'}, mapper_options=mapper_confs[mapper_type])
    else:
        retrieval_conf = extract_features.confs['netvlad']
        retrieval_path = extract_features.main(retrieval_conf, images, outputs)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=20)

        feature_path = extract_features.main(feature_conf, images, outputs)
        match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

        reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path, image_options={'camera_model': 'OPENCV'}, mapper_options=mapper_confs[mapper_type])



if __name__ == '__main__':
    main()
