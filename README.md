# crack_mmdet
pavement crack detection and segmentaion by using mmdetection.

在使用faster_rcnn_rsnet_50_fpn等检测模型后，出现了loss_rpn_bbox: 0.0000， loss_bbox: 0.0000的情况，查看FAQ文档后发现dataset的确存在问题，如下是FAQ文档中的说明：

“Loss goes Nan”

Check if the dataset annotations are valid: zero-size bounding boxes will cause the regression loss to be Nan due to the commonly used transformation for box regression. Some small size (width or height are smaller than 1) boxes will also cause this problem after data augmentation (e.g., instaboost). So check the data and try to filter out those zero-size boxes and skip some risky augmentations on the small-size boxes when you face the problem.

Reduce the learning rate: the learning rate might be too large due to some reasons, e.g., change of batch size. You can rescale them to the value that could stably train the model.

Extend the warmup iterations: some models are sensitive to the learning rate at the start of the training. You can extend the warmup iterations, e.g., change the warmup_iters from 500 to 1000 or 2000.

Add gradient clipping: some models requires gradient clipping to stabilize the training process. The default of grad_clip is None, you can add gradient clippint to avoid gradients that are too large, i.e., set optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2)) in your config file. If your config does not inherits from any basic config that contains optimizer_config=dict(grad_clip=None), you can simply add optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2)).

其中第一条是检查数据集中是否存在一些小的尺寸boxes，在经过data augmentation后，它会造成上面的boxes loss goes nan的情况，而路面裂缝数据集中标注文件不可避免会存在大量width和height很小的boxes，当然可以不使用一些有影响data augmentation，或者去掉这些width和height很小的boxes来解决，然而这也会影响result，这一点也说明了在路面裂缝检测中如果使用深度学习的方法，第一考虑不应该是基于深度学习的目标检测方法，而是分割的方法。
