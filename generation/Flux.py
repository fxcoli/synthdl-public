import os
import random
import csv
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

######
# Set up.
######
# Parameters.
BASE_PATH = "finetune"
OUTPUT_PATH = "generated_Flux"
CSV_FILENAME = "log_Flux.csv"
DEVICE = "cuda"
MODEL_PATH = "black-forest-labs/FLUX.1-Fill-dev"
#MODEL_PATH = "/scratch/checkpoints/Flux_full.safetensors"

# Global variables.
PATHOLOGIES_LIST = [
    'joint effusion',
    'lateral epicondyle displaced',
    'soft tissue swelling',
    'supracondylar fracture'
]
MASK_AREAS = ['joint', 'upper', 'whole']
VIEWS = ['AP', 'LAT']

# Set up the pipeline.
print(f"====== Loading LoRA from {MODEL_PATH} ======")
# First the base Flux model. Note that in its current state, this is for non-finetuned Flux generation.
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    torch_dtype=torch.bfloat16
).to(DEVICE)
############ ENABLE ME FOR LORAS!!!
# Then the LoRA weights.
#pipe.load_lora_weights(MODEL_PATH, adapter_name="default")

# Logging CSV: Filepath, Normal, Pathology 1, Pathology 2, ..., Prompt.
csv_header = ['Filepath', 'Normal'] + PATHOLOGIES_LIST + ['Prompt']
csv_file = open(CSV_FILENAME, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(csv_header)

######
# Supporting functions.
######
def log_to_csv(filepath, active_pathologies, prompt):
    """
    Logs the generated image and its info to the CSV.
    active_pathologies: List of strings of pathologies present in the image.
    prompt: The string prompt utilized during inference.
    """
    # 1 => active pathology, 0 => this pathology is inactive.
    is_normal = len(active_pathologies) == 0
    
    # Create a row starting with Filepath and Normal status.
    row = [filepath, 1 if is_normal else 0]

    # Append 1 or 0 for each pathology.
    for p in PATHOLOGIES_LIST:
        row.append(1 if p in active_pathologies else 0)

    # Finally, append the prompt used for generation.
    row.append(prompt)
    csv_writer.writerow(row)

def get_view_text(view_code):
    """
    Simple function used to avoid boilerplate code - primarily for constructing the prompt.
    """
    return "An anteroposterior" if view_code == "AP" else "A lateral"

def genImg(prompt, base_image, mask_image, save_folder, filename):
    """
    Generate and save the image using the pipeline.
    """
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask_image,
        height=1024,
        width=1024,
        num_inference_steps=30
    ).images[0]

    result.save(save_path)

######
# Step 1: Generate single pathology images only.
######
print("\n====== Starting single pathology generation ======")

for view in VIEWS: # AP, LAT.
    for mask_area in MASK_AREAS: # joint, upper, whole.
        # e.g: finetune/normal/AP/joint/.
        source_dir = os.path.join(BASE_PATH, "normal", view, mask_area)

        # Ignore if not valid - safety check.
        if not os.path.exists(source_dir):
            print(f"Skipping {source_dir}, not found!")
            continue

        all_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
        total_available = len(all_files)

        if total_available == 0:
            continue

        print(f"Processing {view}/{mask_area} ({total_available} base images available)")

        for pathology in PATHOLOGIES_LIST:
            # Rule: Joint effusion cannot be generated from the AP view.
            if pathology == 'joint effusion' and view == 'AP':
                continue

            # Standard classes = 250 per view/mask combination (6 total = 1500).
            # Joint effusion = 500 per view/mask combination (3 total = 1500).
            images_to_generate = 500 if pathology == 'joint effusion' else 250

            # e.g: generated_Flux/joint effusion/AP/
            dest_folder = os.path.join(OUTPUT_PATH, pathology, view)

            for i in range(images_to_generate):
                # Sample sequentially to ensure that each base image is used at least once, then randomly.
                if i < total_available:
                    base_name = all_files[i]
                else:
                    base_name = random.choice(all_files)

                # Load base image and its respective mask.
                image_path = os.path.join(source_dir, base_name)
                mask_name = base_name.replace(".jpg", "-masklabel.png")
                mask_path = os.path.join(source_dir, mask_name)

                if not os.path.exists(mask_path):
                    print(f"Mask not found for an image! {mask_path}.")
                    continue

                base_image = load_image(image_path)
                mask_image = load_image(mask_path)

                # Construct the prompt.
                view_text = get_view_text(view)
                prompt = f"{view_text} radiograph of an elbow displaying {pathology}"

                # Ensure unique index exists in file name.
                save_name = f"{base_name.replace('.jpg', '')}_{mask_area}_{i}.jpg"

                genImg(prompt, base_image, mask_image, dest_folder, save_name)
                log_to_csv(os.path.join(dest_folder, save_name), [pathology], prompt)

######
# Step 2: Generate normal images.
######
print("\n====== Starting Normal Generation ======")

for view in VIEWS:
    # Aggregating all valid base pathology images to serve as bases for normal generation.
    valid_bases = []
    
    for pathology in PATHOLOGIES_LIST:
        patho_base_path = os.path.join(BASE_PATH, pathology, view)
        if not os.path.exists(patho_base_path):
            continue

        files = [f for f in os.listdir(patho_base_path) if f.endswith('.jpg')]
        for f in files:
            mask_name = f.replace(".jpg", "-masklabel.png")
            mask_path = os.path.join(patho_base_path, mask_name)
            
            # Pre-validate mask existence to avoid interruptions in the quota loop.
            if os.path.exists(mask_path):
                valid_bases.append((patho_base_path, f))

    if not valid_bases:
        print(f"No valid bases found for {view} normal generation. Skipping.")
        continue

    # E.g.: generated_Flux/normal/AP/
    dest_folder = os.path.join(OUTPUT_PATH, "normal", view)
    
    print(f"Processing normal generation for {view} (Drawing 750 samples from {len(valid_bases)} valid bases)")

    # Drawing exactly 750 samples per view (random choice with replacement).
    for i in range(750):
        curr_dir, base_name = random.choice(valid_bases)
        
        image_path = os.path.join(curr_dir, base_name)
        mask_name = base_name.replace(".jpg", "-masklabel.png")
        mask_path = os.path.join(curr_dir, mask_name)

        base_image = load_image(image_path)
        mask_image = load_image(mask_path)

        view_text = get_view_text(view)
        prompt = f"{view_text} radiograph of a healthy elbow"

        # Unique naming enforced utilizing index.
        save_name = f"{base_name.replace('.jpg', '')}_normal_{i}.jpg"

        genImg(prompt, base_image, mask_image, dest_folder, save_name)
        log_to_csv(os.path.join(dest_folder, save_name), [], prompt)

######
# Step 3: Multi-pathology generation. Note, uses whole mask images only.
######
print("\n====== Starting multi-pathology generation ======")
counts_to_generate = [2, 3] # 2, 3, or 4 pathologies simultaneously.
# 1500 total multi-pathologies = 6 sub-combinations (3 counts * 2 views).
# Quota: 250 images per iteration.

for num_pathos in counts_to_generate:
    print(f"Generating images with {num_pathos} pathologies...")
    for view in VIEWS:
        # E.g.: finetune/normal/AP/whole
        source_dir = os.path.join(BASE_PATH, "normal", view, "whole")

        if not os.path.exists(source_dir):
            print(f"Skipping {source_dir}, not found!")
            continue

        all_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
        dest_folder = os.path.join(OUTPUT_PATH, "multi-pathology", view)

        # 250 iterations per count/view combination.
        for i in range(375):
            base_name = random.choice(all_files) 

            image_path = os.path.join(source_dir, base_name)
            mask_name = base_name.replace(".jpg", "-masklabel.png")
            mask_path = os.path.join(source_dir, mask_name)

            if not os.path.exists(mask_path):
                continue

            base_image = load_image(image_path)
            mask_image = load_image(mask_path)

            # Rejection sampling algorithm for applying clinical logic restrictions.
            while True:
                selected_pathos = random.sample(PATHOLOGIES_LIST, k=num_pathos)
                
                # Rule 1: Never generate supracondylar fracture combined with lateral epicondyle displaced.
                if 'supracondylar fracture' in selected_pathos and 'lateral epicondyle displaced' in selected_pathos:
                    continue
                    
                # Rule 2: Never generate an isolated lateral epicondyle fracture with joint effusion (n=2).
                if num_pathos == 2 and set(selected_pathos) == {'lateral epicondyle displaced', 'joint effusion'}:
                    continue
                    
                # Rule 3: Supracondylar fractures mandate either joint effusion, soft tissue swelling, or both.
                if 'supracondylar fracture' in selected_pathos:
                    if 'joint effusion' not in selected_pathos and 'soft tissue swelling' not in selected_pathos:
                        continue
                        
                # Combination passed all clinical constraints.
                break

            patho_str = ", ".join(selected_pathos)
            view_text = get_view_text(view)
            prompt = f"{view_text} radiograph of an elbow displaying {patho_str}"

            save_name = f"{base_name.replace('.jpg', '')}_mixed{num_pathos}_{i}.jpg"

            genImg(prompt, base_image, mask_image, dest_folder, save_name)
            log_to_csv(os.path.join(dest_folder, save_name), selected_pathos, prompt)

# Clean up.
csv_file.close()
print(f"====== Processing Complete. Log saved to {CSV_FILENAME} ======")
