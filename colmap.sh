INPUT_DIR=$1
./colmap/build/src/exe/colmap feature_extractor  --database_path $INPUT_DIR/db.db --image_path $INPUT_DIR/images --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_PINHOLE
./colmap/build/src/exe/colmap exhaustive_matcher --database_path $INPUT_DIR/db.db  --SiftMatching.guided_matching 1
mkdir -p $INPUT_DIR/sparse
./colmap/build/src/exe/colmap mapper --database_path $INPUT_DIR/db.db --image_path $INPUT_DIR/images --output_path $INPUT_DIR/sparse

mkdir -p $INPUT_DIR/dense
./colmap/build/src/exe/colmap image_undistorter --image_path $INPUT_DIR/images --input_path $INPUT_DIR/sparse/0 --output_path $INPUT_DIR/dense --output_type COLMAP
./colmap/build/src/exe/colmap patch_match_stereo --workspace_path $INPUT_DIR/dense --workspace_format COLMAP
./colmap/build/src/exe/colmap stereo_fusion --workspace_path $INPUT_DIR/dense --workspace_format COLMAP --input_type geometric --output_path $INPUT_DIR/dense/fused.ply --StereoFusion.min_num_pixels 2

