import datasets
import diffusers
from datasets import load_dataset,Dataset
import torch
from diffusers import FluxPipeline,DiffusionPipeline
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator

import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from transformers import pipeline



parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="dummy")

BINARY="binary"

def main(args):
    
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))


    image_to_text_list= [pipeline("image-to-text", model="Salesforce/blip-image-captioning-large"),pipeline("image-to-text", model="microsoft/git-base")]
    #image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning",force_download=True,from_pt=True).to(accelerator.device)
    text_to_image_list = [FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev"),
                          DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0"),
                          DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers"),
                          DiffusionPipeline.from_pretrained("shuttleai/shuttle-3.1-aesthetic")]
    #text_to_image.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    src_dataset=load_dataset("jlbaker361/new_league_data_max_plus",split="train")
    dest_map={
        col:[] for col in src_dataset.column_names+[BINARY]
    }
    captions=[]
    limit=2
    src_dataset=[row for row in src_dataset][:limit]
    for image_to_text in image_to_text_list:
        image_to_text.model.to(accelerator.device)
        for row in src_dataset:
            for c in row:
                dest_map[c].append(row[c])
            dest_map[BINARY].append("real")
            text=image_to_text(row["splash"])[0]["generated_text"]
            captions.append(text)
        image_to_text.model.to("cpu")
        torch.cuda.empty_cache()
        accelerator.free_memory()
    for text_to_image in text_to_image_list:
        text_to_image.to(accelerator.device)
        for row,text in zip(src_dataset, captions):
            for c in row:
                if c != "splash":
                    dest_map[c].append(row[c])
            image=text_to_image(text).images[0]
            dest_map["splash"].append(image)
            dest_map[BINARY].append("fake")
        text_to_image.to("cpu")
        torch.cuda.empty_cache()
        accelerator.free_memory()
    
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