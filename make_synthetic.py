import datasets
import diffusers
from datasets import load_dataset,Dataset
import torch
from diffusers import FluxPipeline,DiffusionPipeline
from experiment_helpers.gpu_details import print_details
# Load model directly
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering,AutoModel,AutoModelForCausalLM,BertTokenizer, VisualBertModel

import gc
import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from transformers import pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration,Blip2Processor, Blip2ForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from PIL import Image






parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="dummy")

BINARY="binary"

BLIP2="Salesforce/blip2-opt-2.7b"
BLIP_LARGE="Salesforce/blip-image-captioning-large"
GIT="microsoft/git-base-coco"
LLAVA="llava-hf/llava-v1.6-mistral-7b-hf"

class Captioner:
    def __init__(self,hub_id:str,device:str) -> None:
        self.hub_id=hub_id
        self.device=device
        if hub_id ==BLIP2:
            self.processor = Blip2Processor.from_pretrained(BLIP2,force_download=True)
            self.model = Blip2ForConditionalGeneration.from_pretrained(BLIP2).to(device)
        elif hub_id==BLIP_LARGE:
            self.processor = BlipProcessor.from_pretrained(BLIP_LARGE)
            self.model = BlipForConditionalGeneration.from_pretrained(BLIP_LARGE).to(device)
        elif hub_id==GIT:
            self.processor = AutoProcessor.from_pretrained(GIT)
            self.model = AutoModelForCausalLM.from_pretrained(GIT).to(device)
        elif hub_id==LLAVA:
            self.processor=LlavaNextProcessor.from_pretrained(LLAVA)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(LLAVA).to(device)

    def get_text(self,image:Image.Image)->str:
        if self.hub_id==BLIP2:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(**inputs)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        elif self.hub_id==BLIP_LARGE:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            out = self.model.generate(**inputs)
            text=self.processor.decode(out[0], skip_special_tokens=True)
        elif self.hub_id==GIT:
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

            generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elif self.hub_id==LLAVA:
            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is shown in this image?"},
                    {"type": "image"},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
            # autoregressively complete prompt
            output = self.model.generate(**inputs, max_new_tokens=100)
            text=self.processor.decode(output[0], skip_special_tokens=True)
        return text




def main(args):
    
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))


    image_to_text_list= [ BLIP_LARGE, GIT]
    #image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning",force_download=True,from_pt=True).to(accelerator.device)
    text_to_image_list = ["black-forest-labs/FLUX.1-dev",
                          "stabilityai/stable-diffusion-xl-base-1.0",
                          "stabilityai/stable-diffusion-3-medium-diffusers",
                        "shuttleai/shuttle-3.1-aesthetic"]
    #text_to_image.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    src_dataset=load_dataset("jlbaker361/new_league_data_max_plus",split="train")
    dest_map={
        col:[] for col in src_dataset.column_names+[BINARY]
    }
    captions=[]
    limit=3
    src_dataset=[row for row in src_dataset][:limit]
    for image_to_text_str in image_to_text_list:
        image_to_text=Captioner(image_to_text_str,accelerator.device)
        
        for row in src_dataset:
            for c in row:
                dest_map[c].append(row[c])
            dest_map[BINARY].append("real")
            image=row["splash"]
            text=image_to_text.get_text(image)
            captions.append(text)
        #image_to_text.model.to("cpu")
        del image_to_text
        torch.cuda.empty_cache()
        accelerator.free_memory()
        gc.collect()
    for text_to_image_str in text_to_image_list:
        text_to_image=DiffusionPipeline.from_pretrained(text_to_image_str)
        text_to_image.to(accelerator.device)
        for row,text in zip(src_dataset, captions):
            for c in row:
                if c != "splash":
                    dest_map[c].append(row[c])
            image=text_to_image(text).images[0]
            dest_map["splash"].append(image)
            dest_map[BINARY].append("fake")
        #text_to_image.to("cpu")
        del text_to_image
        torch.cuda.empty_cache()
        accelerator.free_memory()
        gc.collect()
    
    Dataset.from_dict(dest_map).push_to_hub("jlbaker361/real-fake-league")



if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")