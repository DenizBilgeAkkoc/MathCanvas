
python foundations_synthesis.py \
  --generation_size=10000 \
  --validation_size=50 \
  --output_path="outputs/data" \
  --dataset_name="foundational_structure_generation_10k" \
  --image_cache_folder="outputs/images_editing_cache" \
  --workers_num=64 \
  --defs_path="defs.txt" \
  --clause_num=3 \
  --image_cache_num=100