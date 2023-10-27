# To install hloc, see: https://github.com/cvg/Hierarchical-Localization
from lighthloc import extract_features, match_features, reconstruction
from lighthloc.associators import pairs_from_retrieval, pairs_from_exhaustive
from pathlib import Path
import click

@click.command()
@click.option('--data', type=str)
@click.option('--output-dir', type=str)
@click.option('--match_type', type=str, default='local')
@click.option('--feature-type', type=str, default='superpoint_inloc')
@click.option('--matcher-type', type=str, default='superpoint+lightglue')
def main(data, output_dir, match_type, feature_type, matcher_type):
    images = Path(data) / 'images/'
    outputs = Path(data)

    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'sparse' / '0'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'

    feature_conf = extract_features.confs[feature_type]
    matcher_conf = match_features.confs[matcher_type]

    assert match_type in ['exhaustive', 'local']
    if match_type == 'exhaustive':
        references = [p.relative_to(images).as_posix() for p in images.iterdir()]
        print(len(references), "mapping images")
        extract_features.main(feature_conf, images, image_list=references, feature_path=features)
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references, image_options={'camera_model': 'OPENCV'})
    else:
        retrieval_conf = extract_features.confs['netvlad']
        retrieval_path = extract_features.main(retrieval_conf, images, outputs)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=20)
        feature_path = extract_features.main(feature_conf, images, outputs)
        match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

        reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path, image_options={'camera_model': 'OPENCV'})



if __name__ == '__main__':
    main()
