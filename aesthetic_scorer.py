import requests
import torch
from PIL import Image
import clip
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV1
import warnings
warnings.filterwarnings('ignore')

class AestheticScorer:
    """
    Robust aesthetic scorer with multiple fallback methods.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None
        self.processor = None
        self.clip_model = None
        self.clip_preprocess = None

        # Try to initialize the best available method
        self._initialize_scorer()
        self._initialize_sim_model()
    
    def _initialize_scorer(self):
        """Initialize the best available scoring method."""

        model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"

        self.model = AestheticsPredictorV1.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        if self.device == "cuda":
            self.model = self.model.to(self.device)

    def _initialize_sim_model(self): 
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        
        # Define aesthetic text prompts
        self.positive_prompts = [
            "a beautiful, high quality professional image of a car",
            "sharp, clear, well-composed image with good lighting",
            "aesthetically pleasing image of a car"
        ]
        
        self.negative_prompts = [
            "blurry, low quality, poorly composed image",
            "dark, unclear, amateur photograph",
            "bad quality, unappealing photo"
        ]
        
        # Precompute text features
        pos_tokens = clip.tokenize(self.positive_prompts).to(self.device)
        neg_tokens = clip.tokenize(self.negative_prompts).to(self.device)
        
        with torch.no_grad():
            self.pos_features = self.clip_model.encode_text(pos_tokens)
            self.neg_features = self.clip_model.encode_text(neg_tokens)
            
            # Normalize features
            self.pos_features = self.pos_features / self.pos_features.norm(dim=-1, keepdim=True)
            self.neg_features = self.neg_features / self.neg_features.norm(dim=-1, keepdim=True)
        

    def score(self, image_input: str | Image.Image) -> float:
        """
        Score an image using the best available method.
        
        Args:
            image_input: Path to the image file, or PIL Image 
            
        Returns:
            Aesthetic score from 0-10, Clip score from 0-10
        """

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Invalid image input type.")

        inputs = self.processor(images=image, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad(): # or `torch.inference_model` in torch 1.9+
            outputs = self.model(**inputs)
        score = outputs.logits.item()

        return score

    def score_clip_similarity(self, image_input: str | Image.Image) -> float:
        """Score using CLIP similarity to aesthetic prompts."""

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Invalid image input type.")

        image_embed = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_embed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            pos_sim = (image_features @ self.pos_features.T).mean().item()
            neg_sim = (image_features @ self.neg_features.T).mean().item()
        
        # Convert to 0-10 scale
        # Similarity ranges from -1 to 1, we want higher pos_sim and lower neg_sim
        score = ((pos_sim - neg_sim) + 1) * 5  # Maps [-1, 1] to [0, 10]
        
        return score


# Test script
if __name__ == "__main__":
    scorer = AestheticScorer()
    
    url = "https://github.com/shunk031/simple-aesthetics-predictor/blob/master/assets/a-photo-of-an-astronaut-riding-a-horse.png?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)

    # image_path = "/mnt/damian/Projects/minRF/data/car_images/ASTONMARTIN_ASTON_MARTIN_Rapide_AMR_2017_0a437262.png"

    score_laion = scorer.score(image)
    print(f"Score of image: {score_laion}")
    
    clip_sim = scorer.score_clip_similarity(image)
    print(f"CLIP similarity score of image: {clip_sim}")