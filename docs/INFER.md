If you wish to test on custom 3D scenes or categories, you can utilize our example configs: 
 `tools/cfgs/scannet_models/spconv_clip_openvocab.yaml` and `tools/cfgs/scannet_models/inst/softgroup_clip_openvocab.yaml`

The key parameters to consider are as follows:
- `TEXT_EMBED.CATEGORY_NAMES`

    This parameter allows you to define the category list for segmentation.

- `TASK_HEAD.CORRECT_SEG_PRED_BINARY` and `INST_HEAD.CORRECT_SEG_PRED_BINARY`

    These parameters allow you to decide using binary head to rectify semantic scores or not.


To save the results, you can use the command `--save_results semantic,instance`. Afterward, you can employ the visualization utilities found in tools/visual_utils/visualize_indoor.py to visualize the predicted results.

