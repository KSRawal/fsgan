
from fsgan.inference.swap import FaceSwapping
from fsgan.criterions.vgg_loss import VGGLoss
import os
def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class Swapping:
    def __init__(self):
        weights_dir = '../weights'
        finetune_iterations = 100
        seg_remove_mouth = True
        seg_batch_size = 2
        batch_size = 2

        detection_model = os.path.join(weights_dir, 'v2/WIDERFace_DSFD_RES152.pth')
        pose_model = os.path.join(weights_dir, 'shared/hopenet_robust_alpha1.pth')
        lms_model = os.path.join(weights_dir, 'v2/hr18_wflw_landmarks.pth')
        seg_model = os.path.join(weights_dir, 'v2/celeba_unet_256_1_2_segmentation_v2.pth')
        reenactment_model = os.path.join(weights_dir, 'v2/nfv_msrunet_256_1_2_reenactment_v2.1.pth')
        completion_model = os.path.join(weights_dir, 'v2/ijbc_msrunet_256_1_2_inpainting_v2.pth')
        blending_model = os.path.join(weights_dir, 'v2/ijbc_msrunet_256_1_2_blending_v2.pth')
        criterion_id_path = os.path.join(weights_dir, 'v2/vggface2_vgg19_256_1_2_id.pth')
        criterion_id = VGGLoss(criterion_id_path)
        self.model = FaceSwapping(
                detection_model=detection_model, pose_model=pose_model, lms_model=lms_model,
                seg_model=seg_model, reenactment_model=reenactment_model,
                completion_model=completion_model, blending_model=blending_model,
                criterion_id=criterion_id,
                finetune=True, finetune_save=True, finetune_iterations=finetune_iterations,
                seg_remove_mouth=finetune_iterations, batch_size=batch_size,
                seg_batch_size=seg_batch_size, encoder_codec='mp4v')
                
    def predict(self,source_path,target_path):
        output_path = os.path.splitext(os.path.split(source_path)[1])[0]+'_'+os.path.splitext(os.path.split(target_path)[1])[0]+'.mp4'
        finetune = True
        select_source, select_target = 'longest', 'longest'
        self.model(source_path,target_path,output_path,select_source,select_target,finetune)
        return output_path

# Swapping = Swapping()
# Swapping.predict('data/KSR Animated.mp4','data/AB.mp4')