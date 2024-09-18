# third-party imports

import sys
import subprocess
import os
import json
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import random
from dotenv import load_dotenv, find_dotenv
import torch
from peft import PeftModel, LoraConfig
import numpy as np
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

# local import
from utilities import (shell_execute, prep_work, prompt_modification, standard_image_generation, cone_json_maker,
                       generate_yaml_file_mix_of_show, generate_captions, generate_photo_mask,
                       move_previously_generated_images_to)
from evaluate_story import identity_preservation
from character import CHARACTER

_ = load_dotenv(find_dotenv())


def DreamBooth_with_LoRA(characters: list, training_steps: int = 800, lr: float = 1e-4):
    """
    training one LoRA per character
    :param characters: list of characters instances.
    :param training_steps: number of steps to train diffusion models
    :param lr: learning rate
    :return: None
    """
    model_name = 'emilianJR/epiCRealism'

    filename = 'LORA'
    repo_file_name = 'huggingface_peft'
    repo_url = "https://github.com/huggingface/peft"

    for char in characters:
        # initial model setup: include cloning of GitHub repo, creating a venv, installing dependencies
        class_data_dir, output_dir, python_venv_path = prep_work(filename, repo_file_name, repo_url, char,
                                                                 training_steps, lr)

        # Define the path to the Python script
        execute_file = os.path.join(os.getcwd(), filename, repo_file_name,
                                    "examples", "lora_dreambooth", "train_dreambooth.py")

        num_instance_images = len(os.listdir(char.photo_folder))
        num_class_images = num_instance_images * 24

        # Define the command line arguments
        args = [
            f"--pretrained_model_name_or_path={model_name}",
            f"--instance_data_dir={char.photo_folder}",
            f"--class_data_dir={class_data_dir}",
            f"--output_dir={output_dir}",
            "--train_text_encoder",
            "--with_prior_preservation",
            "--prior_loss_weight=1.0",
            "--num_dataloader_workers=1",
            f"--instance_prompt='a photo of {char.unique_token + ' man' if char.gender == 'M' else char.unique_token + ' woman'}'",
            f"--class_prompt='a photo of {'man' if char.gender == 'M' else 'woman'}'",
            "--resolution=512",
            "--train_batch_size=4",
            "--lr_scheduler='constant'",
            "--lr_warmup_steps=0",
            f"--num_class_images={num_class_images}",
            "--checkpointing_steps=10000",
            "--use_lora",
            "--lora_r=16",
            "--lora_alpha=27",
            "--lora_text_encoder_r=16",
            "--lora_text_encoder_alpha=17",
            f"--learning_rate={lr}",
            "--gradient_accumulation_steps=1",
            "--gradient_checkpointing",
            f"--max_train_steps={training_steps}"
        ]

        # path to venv accelerate file
        accelerate_path = os.path.join(os.getcwd(), filename, 'venv', 'bin', 'accelerate')

        # Construct the full command string
        command = f"{python_venv_path} {accelerate_path} launch {execute_file} {' '.join(args)}"

        # Execute the command
        try:
            subprocess.run(command, shell=True, check=True)

        except subprocess.CalledProcessError as e:
            print(f"Error executing the script: {e}")


def DreamBooth_with_LoRA_inference(characters: list,
                                   training_steps: int = 800, sampling_steps: int = 50,
                                   lr: float = 1e-4, CFG_guidance: float = 7.5, combination_type: str = 'linear'):
    """
    build a pipeline from the base model previously used for training LORAs then load and combine multiple LORAs using
    hugging face's peft module for inference
    :param characters: list of characters instances.
    :param training_steps: steps to train diffusion models
    :param sampling_steps: number of inference steps
    :param lr: learning rate
    :return: None
    """

    # The functions below originate from the following:
    # https://github.com/huggingface/peft/blob/main/examples/lora_dreambooth/lora_dreambooth_inference.ipynb
    def get_lora_sd_pipeline(
            ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="cuda", adapter_name="default"
    ):
        unet_sub_dir = os.path.join(ckpt_dir, "unet")
        text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
        if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
            config = LoraConfig.from_pretrained(text_encoder_sub_dir)
            base_model_name_or_path = config.base_model_name_or_path

        if base_model_name_or_path is None:
            raise ValueError("Please specify the base model name or path")

        pipe = StableDiffusionPipeline.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

        if os.path.exists(text_encoder_sub_dir):
            pipe.text_encoder = PeftModel.from_pretrained(
                pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
            )

        if dtype in (torch.float16, torch.bfloat16):
            pipe.unet.half()
            pipe.text_encoder.half()

        pipe.to(device)
        return pipe

    def load_adapter(pipe, ckpt_dir, adapter_name):
        unet_sub_dir = os.path.join(ckpt_dir, "unet")
        text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
        pipe.unet.load_adapter(unet_sub_dir, adapter_name=adapter_name)
        if os.path.exists(text_encoder_sub_dir):
            pipe.text_encoder.load_adapter(text_encoder_sub_dir, adapter_name=adapter_name)

    def set_adapter(pipe, adapter_name):
        pipe.unet.set_adapter(adapter_name)
        if isinstance(pipe.text_encoder, PeftModel):
            pipe.text_encoder.set_adapter(adapter_name)

    def create_weighted_lora_adapter(pipe, adapters, weights, combination_type, adapter_name="default", density=0.5,
                                     majority_sign_method='total'):
        args = {
            'adapters': adapters,
            'weights': weights,
            'adapter_name': adapter_name,
            'combination_type': combination_type
        }

        if combination_type in ['ties', 'ties_svd', 'dare_ties', 'dare_ties_svd']:
            args.update({'density': density, 'majority_sign_method': majority_sign_method})
        elif combination_type in ['dare_linear', 'dare_linear_svd', 'magnitude_prune', 'magnitude_prune_svd']:
            args.update({'density': density})

        pipe.unet.add_weighted_adapter(**args)

        if isinstance(pipe.text_encoder, PeftModel):
            pipe.text_encoder.add_weighted_adapter(**args)

        return pipe

    filename = 'LORA'
    tag = 'two_characters' if len(characters) == 2 else (
        'three_characters' if len(characters) == 3 else 'four_characters')

    # run inference on GPT_4 prompts then switch to Llama_3 prompts
    GPT_4 = True
    for _ in range(2):
        LLM = 'GPT_4' if GPT_4 else 'Llama_3'

        # load prompts from json file
        JSON_name = f'prompt_{LLM}_{tag}.json'
        json_file_name = os.path.join(os.getcwd(), JSON_name)
        with open(json_file_name, "r") as file:
            data = json.load(file)

        # create a folder to store generated images
        output_dir = os.path.join(os.getcwd(), filename, 'results', tag, f'{LLM}', f'{training_steps}', f'{lr}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # move previously generated images in output directory to 'old' folder in the same directory
        move_previously_generated_images_to(output_dir)

        update_prompt_json = []
        for prompt_dict in data:
            modified_prompt_dict = prompt_dict.copy()

            # modify the prompts to incorporate unique character tokens formatted in a method-specific manner.
            prompt = prompt_modification(filename=filename,
                                         characters=modified_prompt_dict['Characters_involved'],
                                         prompt=prompt_dict['prompt_text'])

            # create a generator for reproducibility
            generator = torch.Generator(device="cuda").manual_seed(0)

            negative_prompt = prompt_dict['negative_prompt']
            guidance_scale = CFG_guidance
            image_path = os.path.join(output_dir,
                                      f"{prompt_dict['prompt_number']}_{combination_type}_{sampling_steps}_inference.png")

            if len(modified_prompt_dict['Characters_involved']) > 1:
                # create a pipeline and load the first lORA to it
                trained_model_path = os.path.join(os.getcwd(), filename, str(training_steps), str(lr),
                                                  characters[0].name)

                pipe = get_lora_sd_pipeline(trained_model_path, base_model_name_or_path='emilianJR/epiCRealism',
                                            adapter_name=characters[0].unique_token)

                # load textual inversion for negative prompt
                pipe.load_textual_inversion(
                    "klnaD/negative_embeddings", weight_name="EasyNegativeV2.safetensors", token="EasyNegative"
                )

                # load all other lORAs to pipeline
                for char in characters[1:]:
                    trained_model_path = os.path.join(os.getcwd(), filename, str(training_steps), str(lr), char.name)
                    load_adapter(pipe, trained_model_path, adapter_name=char.unique_token)

                # weighted combinations of multiple LORAs
                pipe = create_weighted_lora_adapter(pipe=pipe,
                                                    adapters=[char.unique_token for char in characters],
                                                    weights=[1 for _ in characters], combination_type=combination_type,
                                                    adapter_name=tag)
                set_adapter(pipe, adapter_name=tag)

                # generate an image that is safe for work
                nsfw = True
                while nsfw:
                    images = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=sampling_steps,
                                  guidance_scale=guidance_scale, generator=generator).images[0]

                    images_np = np.array(images)
                    if not np.array_equal(images_np, np.zeros(images_np.shape)):
                        nsfw = False

            elif len(modified_prompt_dict['Characters_involved']) == 1:
                # create a pipeline and load the first lORA to it
                trained_model_path = os.path.join(os.getcwd(), filename, str(training_steps), str(lr),
                                                  modified_prompt_dict['Characters_involved'][0]['name'])

                pipe = get_lora_sd_pipeline(trained_model_path, base_model_name_or_path='emilianJR/epiCRealism',
                                            adapter_name=modified_prompt_dict['Characters_involved'][0]['unique_token'])

                # load textual inversion for negative prompt
                pipe.load_textual_inversion(
                    "klnaD/negative_embeddings", weight_name="EasyNegativeV2.safetensors", token="EasyNegative"
                )

                # generate an image that is safe for work
                nsfw = True
                while nsfw:
                    images = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=sampling_steps,
                                  guidance_scale=guidance_scale, generator=generator).images[0]
                    images_np = np.array(images)
                    if not np.array_equal(images_np, np.zeros(images_np.shape)):
                        nsfw = False
                        print('IMAGE REGENERATED - because NSFW was detected')

            else:
                height = 512
                width = 512
                images = standard_image_generation(model_name='emilianJR/epiCRealism',
                                                   prompt=prompt, negative_prompt=prompt_dict['negative_prompt'],
                                                   num_inference_steps=sampling_steps, guidance_scale=guidance_scale,
                                                   height=height, width=width)

            images.save(image_path)

            modified_prompt_dict['generated_photo_path']['LORA'] = image_path

            # measure ViTS 16 DINO embeddings, FaceNet, inception_v3 scores
            if len(modified_prompt_dict['Characters_involved']) > 0:
                real_photo_path_list = [char['random_photo'] for char in modified_prompt_dict['Characters_involved']]
                evaluations_method = ['ViTS_16_DINO_embeddings_', 'FaceNet_', 'inception_v3_']
                for eval_method in evaluations_method:
                    scores_int = identity_preservation(image_path, real_photo_path_list, eval_method)
                    for i, x in enumerate(scores_int):
                        modified_prompt_dict['scores'][filename][f'{eval_method}{i + 1}'] = float(x)

            update_prompt_json.append(modified_prompt_dict)

        # Convert the template to JSON format
        template_json = json.dumps(update_prompt_json, indent=4)

        # Save the revised list of prompt dictionaries to a JSON file
        json_file_path = os.path.join(os.getcwd(), f'prompt_{LLM}_{tag}_filled.json')
        with open(json_file_path, "w") as file:
            file.write(template_json)

        # switch to LLaMa 3 json file
        GPT_4 = False


def textual_inversion(characters: list,
                      training_steps: int,
                      lr: float):
    """
    training one textual inversion per character then use hugging face's peft module to combine multiple LoRAs
    :param characters: list of characters instances.
    :param training_steps: number of steps to train diffusion models
    :param lr: learning rate
    :return: None
    """
    model_name = 'emilianJR/epiCRealism'

    # initial model setup: include cloning of GitHub repo, creating a venv, installing dependencies
    filename = 'textual_inversion'
    repo_file_name = 'huggingface_diffusers'
    repo_url = "https://github.com/huggingface/diffusers.git"
    output_dir, python_venv_path = prep_work(filename, repo_file_name, repo_url, characters, training_steps, lr)

    for char in characters:

        # Define the path to the Python script
        execute_file = os.path.join(os.getcwd(), filename, repo_file_name,
                                    "examples", "textual_inversion", "textual_inversion.py")

        # Define the command line arguments
        args = [
            f"--pretrained_model_name_or_path={model_name}",
            f"--train_data_dir {char.photo_folder}",
            f"--learnable_property=object",
            f"--placeholder_token '{'<' + char.unique_token + '-man>' if char.gender == 'M' else '<' + char.unique_token + '-woman>'}'",
            f"--initializer_token  {'woman' if char.gender == 'F' else 'man'}",
            "--resolution=512",
            "--train_batch_size=4",
            '--num_vectors=5',
            '--save_steps=500',
            "--gradient_accumulation_steps=1",
            f"--max_train_steps={training_steps}",
            f"--learning_rate={lr}",
            "--checkpointing_steps=11000",
            "--lr_scheduler constant",
            "--lr_warmup_steps=0",
            "--seed=0",
            "--enable_xformers_memory_efficient_attention",
            f"--output_dir={os.path.join(output_dir, char.name)}"]

        # path to venv accelerate file
        accelerate_path = os.path.join(os.getcwd(), filename, 'venv', 'bin', 'accelerate')

        # Construct the full command string
        command = f"{python_venv_path} {accelerate_path} launch {execute_file} {' '.join(args)}"

        # Execute the command
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing the script: {e}")


def textual_inversion_inference(characters: list,
                                training_steps: int, sampling_steps: int,
                                lr: float, CFG_guidance: float = 7.5, inference_from_checkpoint=None, save_steps=None):
    """
    inference using basemodel and textual inversion of multiple concepts.
    :param characters: list of characters instances.
    :param training_steps: steps to train diffusion models
    :param sampling_steps: number of inference steps
    :param lr: learning rate
    :param CFG_guidance: CFG guidance
    :param inference_from_checkpoint: used to inference from a specific checkpoint
    :param save_steps: used when inference from all checkpoints with interval 'save_steps'
    :return: None
    """
    model_name = 'emilianJR/epiCRealism'

    filename = 'textual_inversion'

    # when inference, not for fine-tuning
    if save_steps is None:
        save_steps = training_steps

    for checkpoint in range(training_steps, 0, -save_steps):
        # specify the embedding name
        if checkpoint == training_steps:
            weight_name = 'learned_embeds.safetensors'
        else:
            weight_name = f'learned_embeds-steps-{checkpoint}.safetensors'

        if not inference_from_checkpoint is None:
            weight_name = f'learned_embeds-steps-{inference_from_checkpoint}.safetensors'

        # build a pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"),
            torch_dtype=torch.float16
        ).to("cuda")

        # change the scheduler to DPM++ 2M Karras
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # load the negative textual inversion embeddings
        pipe.load_textual_inversion(
            "klnaD/negative_embeddings", weight_name="EasyNegativeV2.safetensors", token="EasyNegative"
        )

        # load textual inversion embeddings for two characters
        Model_output_dir = os.path.join(os.getcwd(), filename, str(training_steps), str(lr))
        for char in characters:
            output_dir_char = os.path.join(Model_output_dir, char.name)
            pipe.load_textual_inversion(output_dir_char, weight_name=weight_name)

        # run inference on GPT_4 prompts then switch to Llama_3 prompts
        GPT_4 = True
        for _ in range(2):
            LLM = 'GPT_4' if GPT_4 else 'Llama_3'
            tag = 'two_characters' if len(characters) == 2 else (
                'three_characters' if len(characters) == 3 else 'four_characters')

            # load prompts from json file
            JSON_name = f'prompt_{LLM}_{tag}_filled.json'
            json_file_name = os.path.join(os.getcwd(), JSON_name)
            with open(json_file_name, "r") as file:
                data = json.load(file)

            # create a folder to store generated images
            output_dir = os.path.join(os.getcwd(), filename, 'results', tag, f'{LLM}', f'{training_steps}', f'{lr}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # move previously generated images in output directory to 'old' folder in the same directory
            move_previously_generated_images_to(output_dir)

            update_prompt_json = []
            for prompt_dict in data:
                modified_prompt_dict = prompt_dict.copy()

                # modify the prompts to incorporate unique character tokens formatted in a method-specific manner.
                prompt = prompt_modification(filename="textual_inversion",
                                             characters=modified_prompt_dict['Characters_involved'],
                                             prompt=prompt_dict['prompt_text'])

                # create a generator for reproducibility
                generator = torch.Generator(device="cuda").manual_seed(0)

                num_samples = 1
                guidance_scale = CFG_guidance
                height = 512
                width = 512

                # generate an image that is safe for work
                nsfw = True
                while nsfw:
                    # generate image
                    with autocast("cuda"), torch.inference_mode():
                        images = pipe(
                            prompt,
                            height=height,
                            width=width,
                            negative_prompt=prompt_dict['negative_prompt'],
                            num_images_per_prompt=num_samples,
                            num_inference_steps=sampling_steps,
                            guidance_scale=guidance_scale,
                            generator=generator
                        ).images[0]

                    # if nsfw was not detected then quit the while loop
                    images_np = np.array(images)
                    if not np.array_equal(images_np, np.zeros(images_np.shape)):
                        nsfw = False
                        print('IMAGE REGENERATED - because NSFW was detected')

                # save the generated image
                image_path = os.path.join(output_dir, f'{checkpoint}',
                                          f"{prompt_dict['prompt_number']}_{sampling_steps}_inference_{str(guidance_scale).replace('.','')}guidance_scale.png")
                modified_prompt_dict['generated_photo_path']['textual_inversion'] = image_path
                if not os.path.exists(os.path.join(output_dir, f'{checkpoint}')):
                    os.makedirs(os.path.join(output_dir, f'{checkpoint}'))
                images.save(image_path)

                # measure ViTS 16 DINO embeddings, FaceNet, inception_v3 scores
                if len(modified_prompt_dict['Characters_involved']) > 0:
                    real_photo_path_list = [char['random_photo'] for char in
                                            modified_prompt_dict['Characters_involved']]
                    evaluations_method = ['ViTS_16_DINO_embeddings_', 'FaceNet_', 'inception_v3_']
                    for eval_method in evaluations_method:
                        scores_int = identity_preservation(image_path, real_photo_path_list, eval_method)
                        for i, x in enumerate(scores_int):
                            modified_prompt_dict['scores'][filename][f'{eval_method}{i + 1}'] = float(x)

                update_prompt_json.append(modified_prompt_dict)

            # Convert the template to JSON format
            template_json = json.dumps(update_prompt_json, indent=4)

            # Save the revised list of prompt dictionaries to a JSON file
            json_file_path = os.path.join(os.getcwd(), f'prompt_{LLM}_{tag}_filled.json')
            with open(json_file_path, "w") as file:
                file.write(template_json)

            # switch to LLaMa 3 json file
            GPT_4 = False

        if not inference_from_checkpoint is None:
            break


def dream_booth(characters: list,
                training_steps: int,
                lr: float, start_save=10000):
    """
    train models using dreambooth method
    :param characters: list of characters instances.
    :param training_steps: steps to train diffusion models
    :param lr: learning rate
    :param start_save: steps where saving checkpoints start
    :return: None
    """
    model_name = 'emilianJR/epiCRealism'

    # initial model setup: include cloning of GitHub repo, creating a venv, installing dependencies
    filename = 'dreambooth'
    repo_file_name = 'ShivamShrirao_diffusers'
    repo_url = "https://github.com/rezkanas/diffusers.git"
    class_data_dir, output_dir, python_venv_path = prep_work(filename, repo_file_name, repo_url, characters,
                                                             training_steps, lr)

    # Define the path to the Python script
    execute_file = os.path.join(os.getcwd(), filename, repo_file_name,
                                "examples", 'dreambooth', "train_dreambooth.py")

    # build a sample prompt to be used during training.
    collect = []
    for char in characters:
        if char.gender == 'M':
            collect.append(char.unique_token + ' man')
        else:
            collect.append(char.unique_token + ' woman')
    sample_prompt = (' and '.join(collect) +
                     ' as a blue ajah aes sedai in wheel of time, digital painting, cinematic lighting, '
                     'art by mark brooks and greg rutkowski')

    # define training parameters
    num_instance_images = sum([len(os.listdir(char.photo_folder)) for char in characters])
    num_class_images = num_instance_images * 24

    json_file_path = os.path.join(os.getcwd(), filename, f"concepts_list_{filename}.json")

    # Define the command line arguments
    args = [
        f"--pretrained_model_name_or_path={model_name}",
        "--pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse",
        f"--output_dir={output_dir}",
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        "--seed=0",
        "--resolution=512",
        "--train_batch_size=1",
        "--train_text_encoder",
        "--use_8bit_adam",
        "--gradient_accumulation_steps=1",
        f"--learning_rate={lr}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--num_class_images={num_class_images}",
        "--sample_batch_size=4",
        f"--max_train_steps={training_steps}",
        f"--save_interval=100",
        f"--save_min_steps={start_save}",
        f"--save_sample_prompt='{sample_prompt}'",
        f"--concepts_list {json_file_path}",
    ]

    # Execute the command
    shell_execute(execute_file, args, python_venv_path)


def dream_booth_inference(characters: list,
                          training_steps: int, sampling_steps: int,
                          lr: float, CFG_guidance: float = 7.5):
    """
    using the new model weights for inference
    :param characters: list of characters instances.
    :param training_steps: steps to train diffusion models
    :param sampling_steps: number of inference steps
    :param lr: learning rate
    :return: None
    """
    filename = 'dreambooth'

    # build a pipeline
    model_dir = os.path.join(os.getcwd(), filename, f'{lr}', f'{training_steps}')
    pipe = StableDiffusionPipeline.from_pretrained(model_dir,
                                                   safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                                                       "CompVis/stable-diffusion-safety-checker"),
                                                   torch_dtype=torch.float16).to("cuda")

    # change the scheduler to DPM++ 2M Karras
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # load textual inversion for negative prompt
    pipe.load_textual_inversion(
        "klnaD/negative_embeddings", weight_name="EasyNegativeV2.safetensors", token="EasyNegative"
    )

    # run inference on GPT_4 prompts then switch to Llama_3 prompts
    GPT_4 = True
    for _ in range(2):
        LLM = 'GPT_4' if GPT_4 else 'Llama_3'
        tag = 'two_characters' if len(characters) == 2 else (
            'three_characters' if len(characters) == 3 else 'four_characters')

        # load prompts from json file
        JSON_name = f'prompt_{LLM}_{tag}_filled.json'
        json_file_name = os.path.join(os.getcwd(), JSON_name)
        with open(json_file_name, "r") as file:
            data = json.load(file)

        # create a folder to store generated images
        output_dir = os.path.join(os.getcwd(), filename, 'results', tag, f'{LLM}', f'{lr}', f'{training_steps}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # move previously generated images in output directory to 'old' folder in the same directory
        move_previously_generated_images_to(output_dir)

        update_prompt_json = []
        for prompt_dict in data:
            modified_prompt_dict = prompt_dict.copy()

            # modify the prompts to incorporate unique character tokens formatted in a method-specific manner.
            prompt = prompt_modification(filename="dreambooth",
                                         characters=modified_prompt_dict['Characters_involved'],
                                         prompt=prompt_dict['prompt_text'])

            # create a generator for reproducibility
            g_cuda = torch.Generator(device="cuda").manual_seed(0)

            num_samples = 1
            guidance_scale = CFG_guidance
            height = 512
            width = 512

            # generate an image that is safe for work
            nsfw = True
            while nsfw:
                with autocast("cuda"), torch.inference_mode():
                    images = pipe(
                        prompt,
                        height=height,
                        width=width,
                        negative_prompt=prompt_dict['negative_prompt'],
                        num_images_per_prompt=num_samples,
                        num_inference_steps=sampling_steps,
                        guidance_scale=guidance_scale,
                        generator=g_cuda
                    ).images[0]

                # if nsfw was not detected then quit the while loop
                images_np = np.array(images)
                if not np.array_equal(images_np, np.zeros(images_np.shape)):
                    nsfw = False
                    print('IMAGE REGENERATED - because NSFW was detected')

            # save the generated image
            image_path = os.path.join(output_dir,
                                      f"{prompt_dict['prompt_number']}_{sampling_steps}_inference_{str(guidance_scale).replace('.','')}CFG_guidance.png")
            modified_prompt_dict['generated_photo_path']['dreambooth'] = image_path
            images.save(image_path)

            # measure ViTS 16 DINO embeddings, FaceNet, inception_v3 scores
            if len(modified_prompt_dict['Characters_involved']) > 0:
                real_photo_path_list = [char['random_photo'] for char in modified_prompt_dict['Characters_involved']]
                evaluations_method = ['ViTS_16_DINO_embeddings_', 'FaceNet_', 'inception_v3_']
                for eval_method in evaluations_method:
                    scores_int = identity_preservation(image_path, real_photo_path_list, eval_method)
                    for i, x in enumerate(scores_int):
                        modified_prompt_dict['scores'][filename][f'{eval_method}{i + 1}'] = float(x)

            update_prompt_json.append(modified_prompt_dict)

        # Convert the template to JSON format
        template_json = json.dumps(update_prompt_json, indent=4)

        # Save the revised list of prompt dictionaries to a JSON file
        json_file_name = os.path.join(os.getcwd(), f'prompt_{LLM}_{tag}_filled.json')
        with open(json_file_name, "w") as file:
            file.write(template_json)

        # switch to LLaMa 3 json file
        GPT_4 = False


def custom_diffusion(characters: list,
                     training_steps: int,
                     lr: float):
    """
        train models using custom diffusion method

    :param characters: list of characters instances.
    :param training_steps: steps to train diffusion models
    :param lr: learning rate
    :return: None
    """
    model_name = 'emilianJR/epiCRealism'

    # initial model setup: include cloning of GitHub repo, creating a venv, installing dependencies
    filename = 'custom_diffusion'
    repo_file_name = 'huggingface_diffusers'
    repo_url = "https://github.com/huggingface/diffusers.git"
    class_data_dir, output_dir, python_venv_path = prep_work(filename, repo_file_name, repo_url, characters,
                                                             training_steps, lr)

    # define training parameters
    SUM_num_instance_images = sum([len(os.listdir(char.photo_folder)) for char in characters])
    num_class_images = SUM_num_instance_images * 24
    json_file_path = os.path.join(os.getcwd(), filename, f"concepts_list_{filename}.json")

    # Define the command line arguments
    args = [
        f"--pretrained_model_name_or_path={model_name}",
        f"--output_dir={output_dir}",
        f"--concepts_list {json_file_path}",
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        "--resolution=512",
        "--train_batch_size=1",
        f"--learning_rate={lr}",
        "--lr_warmup_steps=0",
        f"--num_class_images={num_class_images}",
        f"--max_train_steps={training_steps}",
        "--scale_lr",
        "--hflip",
        "--noaug",
        "--use_8bit_adam",
        f"--checkpointing_steps=10000",
        "--no_safe_serialization",
        "--enable_xformers_memory_efficient_attention",
        f"--modifier_token '{'+'.join(['<{}>'.format(char.unique_token) for char in characters])}'",
        "--freeze_model crossattn",
        "--seed=0"
    ]

    # Define the path to the Python script
    execute_file = os.path.join(os.getcwd(), filename, repo_file_name,
                                'examples', 'custom_diffusion', 'train_custom_diffusion.py')

    # Execute the command
    shell_execute(execute_file, args, python_venv_path)


def custom_diffusion_inference(characters: list,
                               training_steps: int = None, sampling_steps: int = 50,
                               lr: float = None, CFG_guidance: float = 7.5):
    """
        using the new model weights for inference

    :param characters: list of characters instances.
    :param training_steps: steps to train diffusion models
    :param sampling_steps: number of inference steps
    :param lr: learning rate
    :return: None
    """
    model_name = 'emilianJR/epiCRealism'

    filename = 'custom_diffusion'

    Model_output_dir = os.path.join(os.getcwd(), filename, str(training_steps), str(lr))
    # build a pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        safety_checker=StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"),
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.unet.load_attn_procs(
        Model_output_dir, weight_name="pytorch_custom_diffusion_weights.bin"
    )

    # change the scheduler to DPM++ 2M Karras
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # load weight for each character
    for char in characters:
        pipe.load_textual_inversion(Model_output_dir, weight_name=f"<{char.unique_token}>.bin")

    # load textual inversion for negative prompt
    pipe.load_textual_inversion(
        "klnaD/negative_embeddings", weight_name="EasyNegativeV2.safetensors", token="EasyNegative"
    )

    # run inference on GPT_4 prompts then switch to Llama_3 prompts
    GPT_4 = True
    for _ in range(2):
        LLM = 'GPT_4' if GPT_4 else 'Llama_3'
        tag = 'two_characters' if len(characters) == 2 else (
            'three_characters' if len(characters) == 3 else 'four_characters')

        # load prompts from json file
        JSON_name = f'prompt_{LLM}_{tag}_filled.json'
        json_file_name = os.path.join(os.getcwd(), JSON_name)

        with open(json_file_name, "r") as file:
            data = json.load(file)

        # create a folder to store generated images
        output_dir = os.path.join(os.getcwd(), filename, 'results', tag, f'{LLM}', f'{training_steps}', f'{lr}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # move previously generated images in output directory to 'old' folder in the same directory
        move_previously_generated_images_to(output_dir)

        update_prompt_json = []
        for prompt_dict in data:
            modified_prompt_dict = prompt_dict.copy()

            # modify the prompts to incorporate unique character tokens formatted in a method-specific manner.
            prompt = prompt_modification(filename="custom_diffusion",
                                         characters=modified_prompt_dict['Characters_involved'],
                                         prompt=modified_prompt_dict['prompt_text'])

            # create a generator for reproducibility
            generator = torch.Generator(device="cuda").manual_seed(0)

            # generate an image that is safe for work
            nsfw = True
            while nsfw:
                images = pipe(
                    prompt,
                    num_inference_steps=sampling_steps,
                    guidance_scale=CFG_guidance,
                    eta=1.0,
                    negative_prompt=modified_prompt_dict['negative_prompt'],
                    generator=generator
                ).images[0]
                # if nsfw was not detected then quit the while loop
                images_np = np.array(images)
                if not np.array_equal(images_np, np.zeros(images_np.shape)):
                    nsfw = False
                    print('IMAGE REGENERATED - because NSFW was detected')

            # save the generated image
            image_path = os.path.join(output_dir,
                                      f"{prompt_dict['prompt_number']}_{sampling_steps}_inference_{str(CFG_guidance).replace('.','')}CFG_guidance.png")
            modified_prompt_dict['generated_photo_path']['custom_diffusion'] = image_path
            images.save(image_path)

            # measure ViTS 16 DINO embeddings, FaceNet, inception_v3 scores
            if len(modified_prompt_dict['Characters_involved']) > 0:
                real_photo_path_list = [char['random_photo'] for char in modified_prompt_dict['Characters_involved']]
                evaluations_method = ['ViTS_16_DINO_embeddings_', 'FaceNet_', 'inception_v3_']
                for eval_method in evaluations_method:
                    scores_int = identity_preservation(image_path, real_photo_path_list, eval_method)
                    for i, x in enumerate(scores_int):
                        modified_prompt_dict['scores'][filename][f'{eval_method}{i + 1}'] = float(x)

            update_prompt_json.append(modified_prompt_dict)

        # Convert the template to JSON format
        template_json = json.dumps(update_prompt_json, indent=4)

        # Save the revised list of prompt dictionaries to a JSON file
        json_file_name = os.path.join(os.getcwd(), f'prompt_{LLM}_{tag}_filled.json')
        with open(json_file_name, "w") as file:
            file.write(template_json)

        # switch to LLaMa 3 json file
        GPT_4 = False


def cones_2_training(characters: list,
                     training_steps: int = None,
                     lr: float = None):
    """
    this function is used to train all characters, one concept at a time.

    :param characters: list of characters instances.
    :param training_steps: steps to train diffusion models
    :param lr: learning rate
    :return: None
    """

    cones_model_name = 'stabilityai/stable-diffusion-2-1-base'

    filename = 'cones_2'
    repo_file_name = 'cones_2_git_repository'
    repo_url = "https://github.com/rezkanas/Cones-V2.git"

    # train each character alone
    for char in characters:

        # initial model setup: include cloning of GitHub repo, creating a venv, installing dependencies
        python_file = prep_work(filename, repo_file_name, repo_url, char, training_steps)

        # path for output folder to store model weights
        if training_steps is None:
            output_dir = os.path.join(os.getcwd(), filename, char.unique_token)
        else:
            output_dir = os.path.join(os.getcwd(), filename, char.unique_token, str(training_steps), str(lr))

        # Define the path to the Python script
        execute_file = os.path.join(os.getcwd(), filename, repo_file_name, 'train_cones2.py')

        # Define the command line arguments
        args = [
            f"--pretrained_model_name_or_path={cones_model_name}",
            f"--instance_data_dir={char.photo_folder}",
            f"--instance_prompt={char.unique_token}",
            "--token_num=1",
            f"--output_dir={output_dir}",
            "--resolution=768",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=1",
            f"--checkpointing_steps={training_steps}",
            f"--learning_rate={lr}",
            "--center_crop",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            f"--max_train_steps={training_steps}",
            "--loss_rate_first=1e-2",
            "--loss_rate_second=1e-3",
            "--use_8bit_adam",
            "--seed=0",
            "--enable_xformers_memory_efficient_attention",
        ]

        # Execute the command
        shell_execute(execute_file, args, python_file)


def cones_2_inference(characters: list, training_steps: int = None,
                      guidance_steps: int = 50, guidance_weight=0.08, weight_negative=-1e8,
                      lr: float = None):
    """
    using the new model weights for inference. limitation to 3 characters

    :param characters: list of characters instances.
    :param training_steps: steps to train diffusion models
    :param guidance_steps: the number of steps of the layout guidance.
    :param guidance_weight: the strength of the layout guidance
    :param weight_negative: the weight for weakening the signal of irrelevant subject.
    :param lr: learning rate
    :return: None
    """

    filename = 'cones_2'
    repo_file_name = 'cones_2_git_repository'
    cones_model_name = 'stabilityai/stable-diffusion-2-1'

    # run inference on GPT_4 prompts then switch to Llama_3 prompts
    GPT_4 = True
    for _ in range(2):
        LLM = 'GPT_4' if GPT_4 else 'Llama_3'
        tag = 'two_characters' if len(characters) == 2 else (
            'three_characters' if len(characters) == 3 else 'four_characters')

        # load prompts from json file
        JSON_name = f'prompt_{LLM}_{tag}_filled.json'
        json_file_name = os.path.join(os.getcwd(), JSON_name)
        with open(json_file_name, "r") as file:
            data = json.load(file)

        # create a folder to store generated images
        output_dir = os.path.join(os.getcwd(), filename, 'results', tag, f'{LLM}', str(training_steps), str(lr))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # move previously generated images in output directory to 'old' folder in the same directory
        move_previously_generated_images_to(output_dir)

        update_prompt_json = []

        for prompt_dict in data:
            modified_prompt_dict = prompt_dict.copy()

            # modify the prompts to incorporate unique character tokens formatted in a method-specific manner.
            prompt = prompt_modification(filename="cones_2",
                                         characters=modified_prompt_dict['Characters_involved'],
                                         prompt=prompt_dict['prompt_text'])
            image_name = (
                f"{prompt_dict['prompt_number']}_{guidance_steps}guidance_{int(guidance_weight * 100)}g_weight_"
                f"{int(weight_negative * 1e8)}weight_negative.png")
            image_path = os.path.join(output_dir, image_name)

            if "prompt_layout_path" in prompt_dict.keys():
                # build json file that contain all details used for inference
                json_file_path = cone_json_maker(filename, output_dir, prompt,
                                                 modified_prompt_dict['Characters_involved'],
                                                 modified_prompt_dict['prompt_layout_path'], training_steps,
                                                 guidance_steps, guidance_weight, weight_negative, lr)
                args = [
                    f"--pretrained_model_name_or_path={cones_model_name}",
                    f"--inference_config={json_file_path}",
                    f"--output_dir={output_dir}",
                    f'--image_name={image_name}',
                    f'--seed=0'
                ]
                python_venv_path = os.path.join(os.getcwd(), filename, 'venv', 'bin', 'python')

                # Define the path to the Python script
                execute_file = os.path.join(os.getcwd(), filename, repo_file_name, 'inference.py')

                shell_execute(execute_file, args, python_venv_path)

            else:
                # in case of no character in the image, use the generic model
                images = standard_image_generation(model_name=cones_model_name, prompt=prompt,
                                                   negative_prompt=prompt_dict['negative_prompt'],
                                                   num_inference_steps=50, guidance_scale=7, height=768, width=768)
                images.save(image_path)

            # save the generated image
            modified_prompt_dict['generated_photo_path']['cones_2'] = image_path

            # measure ViTS 16 DINO embeddings, FaceNet, inception_v3 scores
            if len(modified_prompt_dict['Characters_involved']) > 0:
                real_photo_path_list = [char['random_photo'] for char in modified_prompt_dict['Characters_involved']]
                evaluations_method = ['ViTS_16_DINO_embeddings_', 'FaceNet_', 'inception_v3_']
                for eval_method in evaluations_method:
                    scores_int = identity_preservation(image_path, real_photo_path_list, eval_method)
                    for i, x in enumerate(scores_int):
                        modified_prompt_dict['scores'][filename][f'{eval_method}{i + 1}'] = float(x)

            update_prompt_json.append(modified_prompt_dict)

        # Convert the template to JSON format
        template_json = json.dumps(update_prompt_json, indent=4)

        # Save the revised list of prompt dictionaries to a JSON file
        json_file_name = os.path.join(os.getcwd(), f'prompt_{LLM}_{tag}_filled.json')
        with open(json_file_name, "w") as file:
            file.write(template_json)

        # switch to LLaMa 3 json file
        GPT_4 = False


def mix_of_show_training_multi_character(characters: list, optimize_textenc_iters: int = None,
                                         optimize_unet_iters: int = None):
    """
    train multiple characters using mix of show method

    :param characters: list of characters instances.
    :param optimize_textenc_iters:
    :param optimize_unet_iters:
    :return: None
    """

    filename = 'mix_of_show'
    repo_file_name = 'mix_of_show_git_repository'

    concept_list = []
    for char in characters:
        # define path for the trained EDloRA
        lora_path = os.path.join(os.getcwd(), filename, repo_file_name, 'experiments', f'EDLoRA_{char.name}', 'models',
                                 'edlora_model-latest.pth')
        if not os.path.exists(lora_path):
            # train single concept
            mix_of_show_training_single_concept(char)

            # Collect Concept Models
            concepts_dic = {
                "lora_path": lora_path,
                "unet_alpha": 1,
                "text_encoder_alpha": 1,
                "concept_name": f'<{char.unique_token}1> <{char.unique_token}2>'
            }
            concept_list.append(concepts_dic)
        torch.cuda.empty_cache()

    json_file_name = os.path.join(os.getcwd(), filename, 'multiple.json')

    if len(concept_list) > 0:
        with open(json_file_name, "w") as f:
            json.dump(concept_list, f, indent=4)

    # Define the path to the trained model and the script
    execute_file = os.path.join(os.getcwd(), filename, repo_file_name, 'gradient_fusion.py')
    save_path = os.path.join(os.getcwd(), filename, str(optimize_textenc_iters), str(optimize_unet_iters),
                             'trained_model')

    # define arguments for gradient_fusion.py
    args = [

        f"--concept_cfg={json_file_name}",
        f"--save_path={save_path}",
        f"--pretrained_models={os.path.join(os.getcwd(), filename, repo_file_name, 'experiments', 'pretrained_models', 'chilloutmix')}",
        f"--optimize_textenc_iters={optimize_textenc_iters}",
        f"--optimize_unet_iters={optimize_unet_iters}"
    ]

    python_venv_path = os.path.join(os.getcwd(), filename, 'venv', 'bin', 'python')

    # Execute the command and fuse the multiple LORAs
    shell_execute(execute_file, args, python_venv_path)


def mix_of_show_training_single_concept(character):
    """
    train one single character using mix of show method
    :param character: one single character
    :return: None
    """

    # initial model setup: include cloning of GitHub repo, creating a venv, installing dependencies
    filename = 'mix_of_show'
    repo_file_name = 'mix_of_show_git_repository'
    repo_url = "https://github.com/rezkanas/Mix-of-Show.git"
    output_dir, python_venv_path = prep_work(filename, repo_file_name, repo_url, character)

    # generate captions and binary masks for all input images of character and store them in output_dir
    generate_captions(output_dir, character)
    generate_photo_mask(output_dir)

    # generate yaml file to configure the training
    yaml_path = generate_yaml_file_mix_of_show(output_dir, character)
    execute_file = os.path.join(os.getcwd(), filename, repo_file_name, 'train_edlora.py')

    args = [
        f"-opt {yaml_path}"
    ]
    accelerate_path = os.path.join(os.getcwd(), filename, 'venv', 'bin', 'accelerate')

    command = f"{python_venv_path} -c 'from accelerate.utils import write_basic_config; write_basic_config(mixed_precision=\"fp16\")'"
    subprocess.run(command, shell=True, check=True)

    # Construct the full command string
    command = f"{python_venv_path} {accelerate_path} launch {execute_file} {' '.join(args)}"

    # Execute the command
    try:
        subprocess.run(command, shell=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error executing the script: {e}")


def mix_of_show_inference(characters: list, optimize_textenc_iters: int = None,
                          optimize_unet_iters: int = None):
    filename = 'mix_of_show'
    repo_file_name = 'mix_of_show_git_repository'

    # run inference on GPT_4 prompts then switch to Llama_3 prompts
    GPT_4 = True
    for _ in range(2):

        LLM = 'GPT_4' if GPT_4 else 'Llama_3'
        tag = 'two_characters' if len(characters) == 2 else (
            'three_characters' if len(characters) == 3 else 'four_characters')

        # load prompts from json file
        JSON_name = f'prompt_{LLM}_{tag}_filled.json'
        json_file_name = os.path.join(os.getcwd(), JSON_name)
        with open(json_file_name, "r") as file:
            data = json.load(file)

        # create a folder to store generated images
        result_path = os.path.join(os.getcwd(), filename, "result", f'{LLM}', str(optimize_textenc_iters),
                                   str(optimize_unet_iters))
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)

        # move previously generated images in output directory to 'old' folder in the same directory
        move_previously_generated_images_to(result_path)

        update_prompt_json = []
        for prompt_dict in data:
            modified_prompt_dict = prompt_dict.copy()

            # modify the prompts to incorporate unique character tokens formatted in a method-specific manner.
            prompt = prompt_modification(filename="mix_of_show",
                                         characters=modified_prompt_dict['Characters_involved'],
                                         prompt=modified_prompt_dict['prompt_text'])
            image_path = os.path.join(result_path, f"{prompt_dict['prompt_number']}.png")

            if len(modified_prompt_dict['Characters_involved']) > 1:
                # I use templates provided for keypose conditions which limit the output variability,
                # especially for multi-concept inference
                if len(modified_prompt_dict['Characters_involved']) == 2:
                    option_1 = os.path.join(os.getcwd(), 'mix_of_show', 'mix_of_show_git_repository', 'datasets',
                                            'validation_spatial_condition', 'characters-objects',
                                            'bengio+lecun+chair_pose.png')
                    option_2 = os.path.join(os.getcwd(), 'mix_of_show', 'mix_of_show_git_repository', 'datasets',
                                            'validation_spatial_condition', 'characters-objects',
                                            'harry_heminone_scene_pose.png')
                    keypose_condition = random.choice([option_2, option_1])
                elif len(modified_prompt_dict['Characters_involved']) == 3:
                    option_1 = os.path.join(os.getcwd(), 'mix_of_show', 'mix_of_show_git_repository', 'datasets',
                                            'validation_spatial_condition', 'multi-characters', 'real_pose',
                                            'bengio_lecun_bengio.png')
                    option_2 = os.path.join(os.getcwd(), 'mix_of_show', 'mix_of_show_git_repository', 'datasets',
                                            'validation_spatial_condition', 'multi-characters', 'real_pose',
                                            'harry_hermione_thanos.png')
                    keypose_condition = random.choice([option_2, option_1])

                keypose_adaptor_weight = 1.0
                sketch_condition = ''
                sketch_adaptor_weight = 1.0

                # path to trained 'fused' model
                fused_model = os.path.join(os.getcwd(), filename, str(optimize_textenc_iters), str(optimize_unet_iters),
                                           'trained_model', 'combined_model_base')

                context_neg_prompt = modified_prompt_dict['negative_prompt']

                # prepare prompt that accumulate all necessary info for mix of show inference
                prompt_rewrite = ''
                for i, char in enumerate(modified_prompt_dict['Characters_involved']):
                    context_prompt = modified_prompt_dict['context_prompt'].replace(char['name'], "{}".format(
                        'man' if char['gender'] == 'M' else 'woman'))
                    region1_prompt = modified_prompt_dict['region_{}_prompt'.format(i + 1)].replace(
                        char['unique_token'] + ' person',
                        "<{}1> <{}2>".format(char['unique_token'], char['unique_token']))
                    region1_neg_prompt = modified_prompt_dict['negative_prompt']
                    region1 = str(modified_prompt_dict[f'region_{i + 1}_boundaries'])
                    prompt_rewrite += f"{region1_prompt}-*-{region1_neg_prompt}-*-{region1}|"

                # assemble all prompt parts
                prompt_rewrite = '|'.join(prompt_rewrite.split('|')[:-1])

                # path to python script
                execute_file = os.path.join(os.getcwd(), filename, 'mix_of_show_git_repository',
                                            'regionally_controlable_sampling.py')

                # arguments for regionally_controlable_sampling.py
                args = [
                    f"--pretrained_model={fused_model}",
                    f"--sketch_adaptor_weight={sketch_adaptor_weight}",
                    f"--sketch_condition={sketch_condition}",
                    f"--keypose_adaptor_weight={keypose_adaptor_weight}",
                    f"--keypose_condition={keypose_condition}",
                    f"--save_dir={image_path}",
                    f"--prompt=\"{context_prompt}\"",
                    f"--negative_prompt=\"{context_neg_prompt}\"",
                    f"--prompt_rewrite=\"{prompt_rewrite}\"",
                    "--seed=0"
                ]

                # path to venv python
                python_venv_path = os.path.join(os.getcwd(), 'mix_of_show', 'venv', 'bin', 'python')

                # Execute the command
                shell_execute(execute_file, args, python_venv_path)

            elif len(modified_prompt_dict['Characters_involved']) == 1:
                # define a path for the base model used
                pretrained_model_path = os.path.join(os.getcwd(), filename, repo_file_name, 'experiments',
                                                     'pretrained_models', 'chilloutmix')
                # define a path for the trained lora
                lora_model_path = os.path.join(os.getcwd(), filename, repo_file_name, 'experiments',
                                               f"EDLoRA_{modified_prompt_dict['Characters_involved'][0]['name']}",
                                               'models',
                                               'edlora_model-latest.pth')
                enable_edlora = True

                # add directory to python path in order to be able to import functions from the GitHub repo
                import sys
                sys.path.insert(0, os.getcwd())
                sys.path.insert(0, os.path.join(os.getcwd(), filename, 'mix_of_show_git_repository'))
                from mix_of_show.mix_of_show_git_repository.mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline
                from mix_of_show.mix_of_show_git_repository.mixofshow.utils.convert_edlora_to_diffusers import \
                    convert_edlora

                # create a pipeline
                pipe_class = EDLoRAPipeline if enable_edlora else StableDiffusionPipeline
                pipe = pipe_class.from_pretrained(pretrained_model_path,
                                                  scheduler=DPMSolverMultistepScheduler.from_pretrained(
                                                      pretrained_model_path, subfolder='scheduler'),
                                                  torch_dtype=torch.float16).to('cuda')
                pipe, new_concept_cfg = convert_edlora(pipe, torch.load(lora_model_path), enable_edlora=enable_edlora,
                                                       alpha=0.7)

                pipe.set_new_concept_cfg(new_concept_cfg)
                # generate an image that is safe for work
                nsfw = True
                while nsfw:

                    images = \
                        pipe(prompt, negative_prompt=modified_prompt_dict['negative_prompt'], height=768, width=512,
                             num_inference_steps=50,
                             guidance_scale=7).images[0]
                    images.save(image_path)
                    images_np = np.array(images)
                    if not np.array_equal(images_np, np.zeros(images_np.shape)):
                        nsfw = False

            else:
                pretrained_model_path = os.path.join(os.getcwd(), filename, repo_file_name, 'experiments',
                                                     'pretrained_models', 'chilloutmix')
                # generate an image that is safe for work
                nsfw = True
                while nsfw:
                    images = standard_image_generation(model_name=pretrained_model_path,
                                                       prompt=prompt, negative_prompt=prompt_dict['negative_prompt'],
                                                       num_inference_steps=50, guidance_scale=7,
                                                       height=768, width=512)
                    images.save(image_path)
                    images_np = np.array(images)
                    if not np.array_equal(images_np, np.zeros(images_np.shape)):
                        nsfw = False
                        print('IMAGE REGENERATED - because NSFW was detected')

            modified_prompt_dict['generated_photo_path']['mix_of_show'] = image_path

            # measure ViTS 16 DINO embeddings, FaceNet, inception_v3 scores
            if len(modified_prompt_dict['Characters_involved']) > 0:
                real_photo_path_list = [char['random_photo'] for char in modified_prompt_dict['Characters_involved']]
                evaluations_method = ['ViTS_16_DINO_embeddings_', 'FaceNet_', 'inception_v3_']
                for eval_method in evaluations_method:
                    scores_int = identity_preservation(image_path, real_photo_path_list, eval_method)
                    for i, x in enumerate(scores_int):
                        modified_prompt_dict['scores'][filename][f'{eval_method}{i + 1}'] = float(x)

            update_prompt_json.append(modified_prompt_dict)

        # Convert the template to JSON format
        template_json = json.dumps(update_prompt_json, indent=4)

        # Save the revised list of prompt dictionaries to a JSON file
        json_file_name = os.path.join(os.getcwd(), f'prompt_{LLM}_{tag}_filled.json')
        with open(json_file_name, "w") as file:
            file.write(template_json)

        # switch to LLaMa 3 json file
        GPT_4 = False


def Lambda_ECLIPSE_Prior(characters: list, sampling_steps: int = 50, guidance: float = 4.0):
    """
    Lambda ECLIPSE_Prior method, Fine-tuning Free (Fast Personalization) techniques. No training, inference directly.
    :param characters: list of characters instances.
    :param sampling_steps: number of inference steps
    :param guidance: guidance scale
    :return: None
    """

    # initial model setup: include cloning of GitHub repo, creating a venv, installing dependencies
    filename = 'Lambda_ECLIPSE'
    repo_file_name = 'lambda_eclipse_inference_git_repo'
    repo_url = "https://github.com/rezkanas/lambda-eclipse-inference.git"
    _, _ = prep_work(filename, repo_file_name, repo_url, characters)

    from transformers import (
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )

    # Include the current directory and GitHub repository in the system path to facilitate importing methods
    # from the repository.
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), filename, repo_file_name))

    from Lambda_ECLIPSE.lambda_eclipse_inference_git_repo.src.pipelines.pipeline_kandinsky_subject_prior import \
        KandinskyPriorPipeline
    from Lambda_ECLIPSE.lambda_eclipse_inference_git_repo.src.priors.lambda_prior_transformer import PriorTransformer
    from diffusers import DiffusionPipeline

    # assemble components of Lambda ECLIPSE Prior pipeline
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        projection_dim=1280,
        torch_dtype=torch.float16,
    )
    tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    prior = PriorTransformer.from_pretrained("ECLIPSE-Community/Lambda-ECLIPSE-Prior-v1.0", torch_dtype=torch.float16)
    pipe_prior = KandinskyPriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior",
        prior=prior,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe = DiffusionPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder",
        torch_dtype=torch.float16
    ).to("cuda")

    # run inference on GPT_4 prompts then switch to Llama_3 prompts
    GPT_4 = True
    for _ in range(2):
        LLM = 'GPT_4' if GPT_4 else 'Llama_3'
        tag = 'two_characters' if len(characters) == 2 else (
            'three_characters' if len(characters) == 3 else 'four_characters')

        # load prompts from json file
        JSON_name = f'prompt_{LLM}_{tag}_filled.json'
        json_file_name = os.path.join(os.getcwd(), JSON_name)
        with open(json_file_name, "r") as file:
            data = json.load(file)

        # create a folder to store generated images
        output_dir = os.path.join(os.getcwd(), filename, 'results', tag, f'{LLM}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # move previously generated images in output directory to 'old' folder in the same directory
        move_previously_generated_images_to(output_dir)

        update_prompt_json = []
        for prompt_dict in data:
            modified_prompt_dict = prompt_dict.copy()

            # modify the prompts to incorporate unique character tokens formatted in a method-specific manner.
            prompt = prompt_modification(filename="Lambda_ECLIPSE",
                                         characters=modified_prompt_dict['Characters_involved'],
                                         prompt=modified_prompt_dict['prompt_text'])

            # specify seed
            g_cuda = torch.Generator(device="cuda").manual_seed(0)

            # pass on the images of the characters along with their unique tokens
            raw_data = {
                "prompt": prompt,
                "subject_images": [char['random_photo'] for char in modified_prompt_dict['Characters_involved']],
                "subject_keywords": [
                    f"{char['unique_token']} man" if char['gender'] == 'M' else f"{char['unique_token']} woman"
                    for char in modified_prompt_dict['Characters_involved']
                ]
            }

            image_emb, negative_image_emb = pipe_prior(
                raw_data=raw_data,
            ).to_tuple()

            # generate an image that is safe for work
            nsfw = True
            while nsfw:

                # generate image
                images = pipe(
                    image_embeds=image_emb,
                    negative_image_embeds=negative_image_emb,
                    num_inference_steps=sampling_steps,
                    guidance_scale=guidance,
                    generator=g_cuda

                ).images[0]

                images_np = np.array(images)
                if not np.array_equal(images_np, np.zeros(images_np.shape)):
                    nsfw = False
                    print('IMAGE REGENERATED - because NSFW was detected')

            # save the generated image
            image_path = os.path.join(output_dir,
                                      f"{prompt_dict['prompt_number']}_{sampling_steps}step_{guidance}guidance.png")
            modified_prompt_dict['generated_photo_path']['Lambda_ECLIPSE'] = image_path
            images.save(image_path)

            # measure ViTS 16 DINO embeddings, FaceNet, inception_v3 scores
            if len(modified_prompt_dict['Characters_involved']) > 0:
                real_photo_path_list = [char['random_photo'] for char in modified_prompt_dict['Characters_involved']]
                evaluations_method = ['ViTS_16_DINO_embeddings_', 'FaceNet_', 'inception_v3_']
                for eval_method in evaluations_method:
                    scores_int = identity_preservation(image_path, real_photo_path_list, eval_method)
                    for i, x in enumerate(scores_int):
                        modified_prompt_dict['scores'][filename][f'{eval_method}{i + 1}'] = float(x)

            update_prompt_json.append(modified_prompt_dict)

        # Convert the template to JSON format
        template_json = json.dumps(update_prompt_json, indent=4)

        # Save the revised list of prompt dictionaries to a JSON file
        json_file_name = os.path.join(os.getcwd(), f'prompt_{LLM}_{tag}_filled.json')
        with open(json_file_name, "w") as file:
            file.write(template_json)

        # switch to LLaMa 3 json file
        GPT_4 = False


if __name__ == "__main__":
    photos_folder_1 = os.path.join(os.getcwd(), 'photos', 'Rizeh')
    photos_folder_2 = os.path.join(os.getcwd(), 'photos', 'Basel')
    photos_folder_3 = os.path.join(os.getcwd(), 'photos', 'Mamasalme')

    character_1 = CHARACTER(
        photos_folder_1,
        gender='F',
        name="Rizeh",
        traits={
            'positive traits': ['Dutiful', 'Honest'],
            'neutral traits': ['Irreverent', 'Undemanding'],
            'negative traits': ['Tense', 'Ignorant']},
        unique_token='znrz'
    )
    character_2 = CHARACTER(
        photos_folder_2,
        gender='M',
        name="Basel",
        unique_token='nsnn'
    )

    character_3 = CHARACTER(
        photos_folder_3,
        gender='F',
        name="Mamasalme",
        traits={
            'positive traits': ['Fun-loving', 'Adventurous'],
            'neutral traits': ['Pure', 'Stylish'],
            'negative traits': ['Cynical', 'Conventional']},
        unique_token='mlmlsm'
    )

    '''Models'''
    # try also one of the following checkpoints:
    # 'SG161222/RealVisXL_V4.0', 'windwhinny/chilloutmix', 'SG161222/Realistic_Vision_V6.0_B1_noVAE', "CompVis/stable-diffusion-v1-4"
    # 'emilianJR/epiCRealism'

    '''textual inversion tuning camp'''
    torch.cuda.empty_cache()
    for lr in [5e-4, 5e-5]:
        for training_step in [10000]:
            textual_inversion([character_1, character_2, character_3], training_step, lr)
            for sampling_steps in [25, 50, 75, 100]:
                for CFG_guidance in [2, 4, 7, 10]:
                    textual_inversion_inference(characters=[character_1, character_2, character_3], training_steps=training_step,
                                                sampling_steps=sampling_steps, lr=lr, CFG_guidance=CFG_guidance,
                                                save_steps=500)

    '''LORA tuning camp '''
    combination_types = ['svd', 'linear', 'cat', 'ties', 'ties_svd', 'dare_ties', 'dare_linear', 'dare_ties_svd',
                         'dare_linear_svd', 'magnitude_prune', 'magnitude_prune_svd']
    for lr in [1e-4, 5e-5, 5e-4]:
        for training_step in [500, 800, 1000, 1500, 2500]:
            DreamBooth_with_LoRA([character_1, character_2], training_steps=training_step, lr=lr)
            for combination_type in combination_types:
                for s_step in [50, 75, 100, 125]:
                    for CFG_guidance in [7.5]:
                        DreamBooth_with_LoRA_inference([character_1, character_2], training_steps=training_step,
                                                       sampling_steps=s_step, lr=lr, combination_type=combination_type,
                                                       CFG_guidance=CFG_guidance)

    '''Dream booth tuning camp '''
    for training_steps in [1500, 2100, 2700, 3000]:
        for lr in [1e-6, 2e-6]:
            dream_booth([character_1, character_2], training_steps=training_steps,  lr=lr)
            for sampling_steps in [25, 50, 75, 100, 125]:
                for CFG_guidance in [2, 4, 7]:
                    dream_booth_inference([character_1, character_2], training_steps=training_steps,
                                          sampling_steps=sampling_steps, lr=lr, CFG_guidance=CFG_guidance)

    '''custom diffusion tuning camp '''

    for training_steps in [1500, 1800, 2000, 2500]:
        for lr in [1e-4, 5e-4, 1e-6]:
            custom_diffusion([character_1, character_2], training_steps=training_steps, lr=lr)
            for sampling_steps in [25, 50, 75, 100, 125]:
                for CFG_guidance in [7]:
                    custom_diffusion_inference([character_1, character_2], training_steps=training_steps,
                                               sampling_steps=sampling_steps, lr=lr, CFG_guidance=CFG_guidance)

    '''Cones 2 tuning camp '''
    for step in [4500]:  # , 3500,4500, 5000
        for lr in [1e-6]:
            for guidance_steps in [25, 50, 75]:
                for weight_negative in [-1e8, -5e7, -5e8]:
                    for guidance_weight in [0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.28, 0.35, 0.4, 0.5, 0.6]:
                        cones_2_inference([character_1, character_2], training_steps=step,
                                          guidance_steps=guidance_steps,
                                          guidance_weight=guidance_weight, weight_negative=weight_negative, lr=lr)

    '''MIX of SHOW tuning camp '''
    torch.cuda.empty_cache()
    for step1 in [400, 500, 600]:
        for step2 in [30, 50, 70, 100]:
            mix_of_show_training_multi_character([character_1, character_2], step1, step2)
            mix_of_show_inference([character_1, character_2], step1, step2)

    '''Lambda ECLIPSE Prior'''
    for step in [25, 50, 100, 125, 200, 250, 300]:
        for guidance in [4, 7, 1, 10, 3]:
            Lambda_ECLIPSE_Prior([character_1, character_2], step, guidance)
