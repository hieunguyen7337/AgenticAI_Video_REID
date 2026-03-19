import torch
from config import cfg
from model.make_model_clipreid import make_model

def run_simple_inference(model_weight_path):
    """
    Inputs:
        x (torch.Tensor): A batch of video frames.
            Expected shape: (B, T, C, H, W)
            - B: Batch size.
            - T: Temporal sequence length (number of frames per video, usually cfg.INPUT.SEQ_LEN like 8).
            - C: Number of channels (3 for RGB).
            - H: Image height (usually cfg.INPUT.SIZE_TEST[0] like 256).
            - W: Image width (usually cfg.INPUT.SIZE_TEST[1] like 128).
        get_image (bool): False for actual inference.
        cam_label (torch.Tensor or None): Integer labels for cameras (used for SIE camera embedding)
        view_label (torch.Tensor or None): Integer labels for views (used for SIE view embedding)

    Outputs:
        feat (torch.Tensor): The concatenated visual and temporal features used for evaluation.
            If cfg.TEST.NECK_FEAT is 'after': Outputs (B, Dim_Image_BN + Dim_Proj_BN)
            Otherwise: Outputs (B, Dim_Image + Dim_Proj + Dim_Class_Proj)
    """

    # 1. Load default configurations (assume configs/vit_clipreid.yml is used)
    # Automatically merging configs is optional here if default matches what we want
    cfg.merge_from_file('configs/vit_clipreid.yml')
    
    # 2. Initialize the model 
    # (assuming num_classes=100, camera_num=6, view_num=1 for dummy test)
    # Make sure to handle the dataset parameters as needed based on MARS
    model = make_model(cfg, num_class=100, camera_num=6, view_num=1)
    
    # 3. Load model weights
    model.load_param(model_weight_path)
    
    # Ensure model is in eval mode (this disables dropout, uses moving averages for BN, and changes the return structure)
    model.eval()
    model.to("cuda")

    # 4. Create dummy input data matching what the model expects
    B, T, C, H, W = 2, cfg.INPUT.SEQ_LEN, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]
    
    # x shape matches: (Batch, Frames, Channels, Height, Width)
    dummy_input = torch.randn(B, T, C, H, W).cuda() 
    
    # Optional embeddings for Camera and View (Set to None if SIE_CAMERA=False)
    cam_label = torch.tensor([0, 1]).cuda() 
    view_label = torch.tensor([0, 0]).cuda()

    # 5. Run Inference
    with torch.no_grad():
        # During model.eval(), the model returns concatenated features
        features = model(x=dummy_input, get_image=False, cam_label=cam_label, view_label=view_label)

    print(f"Inference complete. Output feature shape: {features.shape}")
    return features

if __name__ == "__main__":
    weight_path = "logs_mars/best_model.pth.tar" 
    run_simple_inference(weight_path)
