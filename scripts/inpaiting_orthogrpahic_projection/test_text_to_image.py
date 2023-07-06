# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5"#,
                                               #torch_dtype=torch.float16)
)
#pipe = pipe.to("cuda")
pipe = pipe.to("cpu")
prompt = "a photo of an avocado man on top of a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")