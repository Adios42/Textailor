python scripts/generate_texture.py \
    --input_dir data/claudia/ \
    --output_dir outputs/claudia \
    --obj_name mesh \
    --obj_file mesh.obj \
    --prompt "a woman wearing a white blouse with a ribbon detail, light beige pants, nude-tone heels, and neatly tied blonde hair" \
    --add_view_to_prompt \
    --ddim_steps 30 \
    --new_strength 1.0 \
    --update_strength 0.3 \
    --view_threshold 0.2 \
    --blend 0 \
    --dist 0.7 \
    --num_viewpoints 36 \
    --viewpoint_mode predefined \
    --use_principle \
    --update_steps 30 \
    --update_mode heuristic \
    --seed 42 \
    --post_process \
    --pre_view 12   \
    --device "2080" \
    --resampling 3  \
    --training      \
    --use_patch     \
    --flexible_view   \
    --num_images 3   \
    --use_face
    # assume the mesh is normalized with y-axis as up