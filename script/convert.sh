for taskdir in /data/dex/RoboTwin/data/*; do
  src="$taskdir/demo_clean_vidar/data"
  dst="$taskdir/demo_clean_vidar/action_gt"
  if [ -d "$src" ]; then
    python export_hdf5_to_pt.py \
      --src_dir "$src" \
      --dst_dir "$dst" \
      --action_key "joint_action/vector"
  fi
done
