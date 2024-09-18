# third-party imports
from hugchat import hugchat
from hugchat.login import Login
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import base64
import yaml
from git import Repo
import subprocess
from PIL import Image
from torch import autocast
import torch
import cv2
from ultralytics import YOLO
import random
import shutil
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import time
import glob

# read local .env file
_ = load_dotenv(find_dotenv())


def query_gpt(messages,
              model="gpt-3.5-turbo",
              temperature=0.2, top_p=0.1, max_tokens=4096,
              continue_conversation: bool = False, previous_chat=None,
              image_url_list: list = None):
    """
    query to OpenAI API

    :param messages: a tuple containing system and user messages
    :param model: LLM model
    :param temperature: LLM temperature influences the language model's output, determining whether the output is
     more random and creative or more predictable
    :param top_p: is another hyperparameter that controls the randomness of language model output
    :param max_tokens
    :param continue_conversation: continue previous conversation.
    :param previous_chat: conversation history
    :param image_url_list: list of image URL
    :return: query response
    """
    # initiate client for OpenAI API
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # unpack messages to user and system messages
    system_message, user_message = messages
    messages = [
        {'role': 'system',
         'content': system_message},
        {'role': 'user',
         'content': f"{user_message}"},
    ]

    # when continuing previous conversation
    if continue_conversation:
        old_messages = [
            {'role': 'system',
             'content': previous_chat[0]},
            {'role': 'user',
             'content': previous_chat[1]},
            {'role': 'assistant',
             'content': previous_chat[2]},
        ]
        messages = old_messages + messages

    # if the query include images
    if not image_url_list is None:
        messages = [
            {'role': 'system',
             'content': system_message},
            {'role': 'user',
             'content': [
                 {"type": "text", "text": user_message}
             ]}
        ]

        # Extend the 'content' list with image_url_list
        for message in messages:
            if message['role'] == 'user':
                message['content'].extend(image_url_list)

    # send the query
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,

    )

    return chat_completion.choices[0].message.content


def query_LLM(system_prompt: str, prompt: str, default_llm=1):
    """
    query to Huggingchat API. It provides an interface for multiple LLM.

    :param system_prompt: system prompt
    :param prompt: user prompt
    :param default_llm: choosing LLM out of the available ones in huggingchat
    :return: query response
    """

    # Log in to huggingface and grant authorization to huggingchat
    huggingchat_user_email = os.environ['huggingchat_user_email']
    huggingchat_user_pwd = os.environ['huggingchat_user_pwd']

    sign = Login(email=huggingchat_user_email, passwd=huggingchat_user_pwd)
    cookies = sign.login()

    # Save cookies to the local directory
    cookie_path_dir = "./cookies_snapshot"
    sign.saveCookiesToDir(cookie_path_dir)

    # Create a ChatBot
    chatbot = hugchat.ChatBot(system_prompt=system_prompt, cookies=cookies.get_dict(),
                              default_llm=default_llm)
    # return the query result

    return str(chatbot.active_model).split('/')[1].replace('-', '_').replace('.', ''), chatbot.chat(prompt)['text']


def encode_image(image_path):
    # Function to encode the image to base 64
    # CITATION: function used directly from https://platform.openai.com/docs/guides/vision
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_prompt_layout(list_of_coordinates, prompt_number, tag, GPT_4):
    """
    generate a photo with layout that will be used for inference by Cone 2 method
    :param list_of_coordinates: bounding box coordinates as defined by LLM responses.
    :param prompt_number: prompt number
    :param tag: story tag
    :param GPT_4: LLM used
    :return:
    """
    # Set the figure size to 768x768 pixels
    fig, ax = plt.subplots(figsize=(768 / 100, 768 / 100))

    # Create a white background
    ax.set_facecolor('white')

    # Draw rectangles
    if list_of_coordinates:
        for i, color_ in zip(list_of_coordinates, ['r', 'b', 'g'][:len(list_of_coordinates)]):
            if isinstance(i, (list, tuple, dict)):
                x, y, w, h = i if isinstance(i, (list, tuple)) else i.values()

            rectangle_ = patches.Rectangle((x, y), width=w, height=h, linewidth=1, color=color_)
            ax.add_patch(rectangle_)

    # Set aspect ratio to equal
    ax.set_aspect('equal', adjustable='box')

    # Set limits
    ax.set_xlim(0, 768)
    ax.set_ylim(0, 768)
    ax.axis('off')

    # define path to save image
    LLM = 'GPT_4' if GPT_4 else "LLama_3"
    image_path = os.path.join(os.getcwd(), f'prompt_layout', f'{LLM}_{tag}', f'prompt_{prompt_number}.png')
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))

    # remove all previous runs
    if prompt_number == 1:
        for filename in os.listdir(os.path.dirname(image_path)):
            filepath = os.path.join(os.path.dirname(image_path), filename)
            os.remove(filepath)

    # Save the plot as an image
    plt.savefig(image_path, dpi=100)
    return image_path


def prompt_modification(filename, characters, prompt):
    """
    modify raw prompt generated by LLM to be recognized by the particularity of the method's text encoder.
    :param filename: method name
    :param characters: list of characters
    :param prompt: raw prompt text
    :return: modified prompt text
    """

    # generic modification to replace character names with token, just in case LLM made a mistake leaving character names
    for char in characters:
        prompt = prompt.replace(char['name'], char['unique_token'] + ' man' if char['gender'] == 'M' else char[
                                                                                                              'unique_token'] + ' woman')
        prompt = prompt.replace(char['unique_token'] + ' person', f"{char['unique_token']} man" if char[
                                                                                                       'gender'] == 'M' else f"{char['unique_token']} woman")

    # apply modification to reformat prompt based on the method
    if filename in ['dreambooth', 'cones_2', 'LORA', 'Lambda_ECLIPSE']:
        return prompt

    if filename == 'textual_inversion':
        for char in characters:
            prompt = prompt.replace(char['unique_token'], '<' + char['unique_token'] + '> man '
            if char['gender'] == 'M' else '<' + char['unique_token'] + '> woman').replace('person', '')
        return prompt

    elif filename == 'custom_diffusion':
        for char in characters:
            prompt = prompt.replace(char['unique_token'], '<' + char['unique_token'] + '> man '
            if char['gender'] == 'M' else '<' + char['unique_token'] + '> woman').replace('person', '')
        return prompt

    elif filename == 'mix_of_show':
        for char in characters:
            prompt = prompt.replace(char['unique_token'],
                                    '<' + char['unique_token'] + '1> <' + char['unique_token'] + '2>').replace(
                'person ', '')
        return prompt


def convert_yml_to_txt(env_yaml_path, requirements_txt_path):
    """
    convert yml file in GitHub repo to a text file
    :param env_yaml_path: path to yaml file
    :param requirements_txt_path: path to text file
    :return: None
    """
    # open yaml file
    with open(env_yaml_path, 'r') as yaml_file:
        env_data = yaml.safe_load(yaml_file)

    # write yaml file content to a text file
    with open(requirements_txt_path, 'w') as requirements_file:
        for package in env_data['dependencies']:
            if isinstance(package, str):
                try:
                    pkg_name, pkg_version, _ = package.split('=')
                except:
                    pkg_name, pkg_version = package.split('=')
                requirements_file.write(f"{pkg_name}=={pkg_version}\n")
            elif isinstance(package, dict):
                # Iterate over the package dictionary
                for name, version in package.items():
                    for element in version:
                        requirements_file.write(f"{element}\n")


def prep_work(filename, repo_file_name, repo_url, characters, training_step=None, lr=None):
    """
    Prepares the environment for a new method by cloning the GitHub repo, creating a virtual environment,
    installing dependencies, and setting up folder structure.

    :param filename: Method name
    :param repo_file_name: Folder name for the GitHub repo.
    :param repo_url: GitHub repo URL
    :param characters: List of characters
    :param training_step: Training steps
    :param lr: Learning rate
    :return: Relevant paths for further processing
    """
    path = os.path.join(os.getcwd(), filename, repo_file_name)
    venv_path = os.path.join(os.getcwd(), filename, 'venv', 'bin', 'python')
    environment_newly_created = False

    # Clone repo and create virtual environment if not exists
    if not os.path.exists(path):
        Repo.clone_from(repo_url, path)
        subprocess.run(["python3", "-m", "venv", os.path.join(os.getcwd(), filename, 'venv')], check=True)
        environment_newly_created = True

    if environment_newly_created:
        _install_dependencies(filename, venv_path, path)

    if filename in ['dreambooth', 'custom_diffusion', 'textual_inversion', 'LORA', 'Lambda_ECLIPSE']:
        return _setup_folder_structure(filename, characters, training_step, lr, venv_path)

    if filename == 'cones_2':
        return venv_path

    if filename == 'mix_of_show':
        return _setup_mix_of_show(filename, characters, training_step, venv_path, repo_file_name)


def _install_dependencies(filename, venv_path, path):
    """Install method-specific dependencies."""
    dependencies = {
        'dreambooth': [
            'git+https://github.com/rezkanas/diffusers.git#ShivamShriraodiffusers', '--upgrade --pre triton',
            'torchvision==0.15.2', 'transformers', 'ftfy', 'bitsandbytes', 'xformers==0.0.20',
            'safetensors', 'accelerate', 'gradio', 'natsort', 'torch==2.0.1', 'diffusers==0.17.0'
        ],
        'custom_diffusion': [
            '--upgrade --pre triton', 'transformers', 'accelerate', 'bitsandbytes', 'wandb', 'torch==2.0.1',
            'torchvision==0.15.2',
            'torchaudio==2.0.2', 'xformers==0.0.20', 'git+https://github.com/huggingface/diffusers'
        ],
        'textual_inversion': [
            '--upgrade --pre triton', 'transformers', 'bitsandbytes', 'wandb', 'accelerate', 'torch==2.0.1',
            'torchvision==0.15.2', 'torchaudio==2.0.2', 'xformers==0.0.20',
            'git+https://github.com/huggingface/diffusers'
        ],
        'cones_2': [
            'xformers==0.0.20', 'numpy==1.24.4', 'bitsandbytes', 'triton'
        ],
        'mix_of_show': [
            'diffusers==0.20.2', 'accelerate', 'omegaconf', 'opencv-python', 'einops', 'IPython', 'xformers==0.0.16',
            'torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html'
        ],
        'LORA': [
            'accelerate', 'torch==2.0.1', 'torchvision==0.15.2', 'torchaudio==2.0.2', 'xformers==0.0.20',
            'diffusers==0.21', 'peft'
        ]
    }
    if filename in ['LORA', 'mix_of_show', 'cones_2', 'Lambda_ECLIPSE']:
        _install_from_requirements(venv_path, path)

    if filename in dependencies:
        for req in dependencies[filename]:
            if req == '--upgrade --pre triton':
                command = [venv_path, '-m', 'pip', 'install', '--upgrade', '--pre', 'triton']
            elif req.startswith('torch==1.13.1+cu117'):
                command = [venv_path, '-m', 'pip', 'install',
                           'torch==1.13.1+cu117',
                           'torchvision==0.14.1+cu117',
                           '-f', 'https://download.pytorch.org/whl/torch_stable.html']
            else:
                command = [venv_path, '-m', 'pip', 'install', req]
            subprocess.run(command, check=True)


def _install_from_requirements(venv_path, path):
    """Install dependencies from requirements file."""
    if not os.path.exists(os.path.join(path, 'requirements.txt')):
        try:
            convert_yml_to_txt(os.path.join(path, 'environment.yml'), os.path.join(path, 'requirements.txt'))
        except:
            convert_yml_to_txt(os.path.join(path, 'environment.yaml'), os.path.join(path, 'requirements.txt'))

    with open(os.path.join(path, 'requirements.txt'), 'r') as f:
        requirements = f.readlines()

    for req in requirements:
        req = req.strip()
        if req and not req.startswith('#'):
            try:
                subprocess.run([venv_path, '-m', 'pip', 'install', req], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error installing {req}: {e}")


def _setup_folder_structure(filename, characters, training_step, lr, venv_path):
    """Setup folder structure and return relevant paths."""
    class_data_dir = os.path.join(os.getcwd(), filename, "person")
    os.makedirs(class_data_dir, exist_ok=True)

    if training_step is None:
        output_dir = os.path.join(os.getcwd(), filename)
    elif filename == 'dreambooth':
        output_dir = os.path.join(os.getcwd(), filename, str(lr))
    elif filename == 'LORA':
        output_dir = os.path.join(os.getcwd(), filename, str(training_step), str(lr), characters.name)
        return class_data_dir, output_dir, venv_path
    else:
        output_dir = os.path.join(os.getcwd(), filename, str(training_step), str(lr))

    if filename in ['textual_inversion', 'Lambda_ECLIPSE']:
        return output_dir, venv_path

    concepts_list = [{
        "instance_prompt": f"photo of %s person" % (
            char.unique_token if filename == 'dreambooth' else '<' + char.unique_token + '>'),
        "class_prompt": "person",
        "instance_data_dir": char.photo_folder,
        "class_data_dir": class_data_dir
    } for char in characters]

    json_file_name = os.path.join(os.getcwd(), filename, f"concepts_list_{filename}.json")
    with open(json_file_name, "w") as f:
        json.dump(concepts_list, f, indent=4)

    return class_data_dir, output_dir, venv_path


def _setup_mix_of_show(filename, characters, training_step, venv_path, repo_file_name):
    """Setup folder structure for mix_of_show method."""
    home_dir = os.path.expanduser('~')
    bin_dir = os.path.join(home_dir, 'bin')
    git_lfs_path = os.path.join(bin_dir, 'git-lfs')

    # Ensure the bin directory exists
    os.makedirs(bin_dir, exist_ok=True)

    # Download git-lfs if it doesn't already exist
    if not os.path.exists(git_lfs_path):
        git_lfs_url = 'https://github.com/git-lfs/git-lfs/releases/download/v2.13.3/git-lfs-linux-amd64-v2.13.3.tar.gz'
        tarball_path = os.path.join(home_dir, 'git-lfs-linux-amd64-v2.13.3.tar.gz')
        subprocess.run(['wget', git_lfs_url, '-O', tarball_path], check=True)

        # Extract the tarball
        subprocess.run(['tar', '-xvzf', tarball_path, '-C', home_dir], check=True)

        # Move the binary to ~/bin
        os.rename(os.path.join(home_dir, 'git-lfs'), git_lfs_path)

        # Initialize git-lfs
        subprocess.run([git_lfs_path, 'install'], check=True)

        # Clean up the tarball
        os.remove(tarball_path)

    # Create the target directory
    target_directory = os.path.join(os.getcwd(), filename, repo_file_name, "experiments",
                                    "pretrained_models", "chilloutmix")
    os.makedirs(target_directory, exist_ok=True)

    # Clone the repository to the target directory if it doesn't already exist
    if not os.listdir(target_directory):  # Check if the directory is empty
        repo_url = "https://huggingface.co/windwhinny/chilloutmix.git"
        subprocess.run(['git', 'clone', repo_url, target_directory], check=True)

        # Pull the LFS objects in the target directory
        subprocess.run(['git', '-C', target_directory, 'lfs', 'pull'], check=True)

    # Now start with preparing the folder structure
    output_dir = os.path.join(os.getcwd(), filename,
                              characters.name if training_step is None else f"{characters.name}/{training_step}")
    os.makedirs(os.path.join(output_dir, 'captions'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image'), exist_ok=True)

    _resize_images(characters.photo_folder, os.path.join(output_dir, 'image'), characters.unique_token)

    single_concept_json = [{
        "instance_prompt": f"<{characters.unique_token}>",
        "instance_data_dir": os.path.join(output_dir, 'image'),
        "caption_dir": os.path.join(output_dir, 'captions'),
        "mask_dir": os.path.join(output_dir, 'mask')
    }]

    with open(os.path.join(output_dir, f"{characters.name}.json"), "w") as f:
        json.dump(single_concept_json, f, indent=4)

    return output_dir, venv_path


def _resize_images(source_folder, target_folder, unique_token):
    """Resize images while maintaining the aspect ratio."""
    i = 1
    for image_name in os.listdir(source_folder):
        img = Image.open(os.path.join(source_folder, image_name))
        width, height = img.size
        aspect_ratio = width / height
        target_short_edge = 550

        if min(width, height) > 512:
            new_width, new_height = (
                target_short_edge, int((target_short_edge / width) * height)) if width < height else (
                int((target_short_edge / height) * width), target_short_edge)
            img.resize((new_width, new_height)).save(os.path.join(target_folder, f'{unique_token}{i}.png'))
            i += 1


def shell_execute(execute_file, args, python_venv_path='python3'):
    # Construct the full command string
    command = f"{python_venv_path} {execute_file} {' '.join(args)}"

    # Execute the command
    try:
        subprocess.run(command, shell=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error executing the script: {e}")


def standard_image_generation(model_name, prompt, negative_prompt, num_inference_steps=50, guidance_scale=7, height=512,
                              width=512):
    """
    used when prompt does not include any character so a generic model could be used
    :param model_name: base model used for training
    :param prompt: prompt text
    :param negative_prompt: negative prompt
    :param num_inference_steps: sampling steps
    :param guidance_scale: guidance_scale
    :param height: output image height
    :param width: output image width
    :return: generated image
    """
    # build a pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        safety_checker=StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"),
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    num_samples = 1

    # generate an image that is safe for work
    nsfw = True
    while nsfw:
        with autocast("cuda"), torch.inference_mode():
            images = pipe(
                prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_samples,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(0)
            ).images[0]
        images_np = np.array(images)
        if not np.array_equal(images_np, np.zeros(images_np.shape)):
            nsfw = False
            print('regenerate because of NSFW detected')

    return images


def cone_json_maker(filename, output_dir, prompt, involved_characters, prompt_layout_path, step, step2, guidance_weight,
                    weight_negative, lr):
    file_dict = {}
    # find the trained model path for each character.
    for char in involved_characters:
        file_dict[char['unique_token']] = os.path.join(os.getcwd(), filename, char['unique_token'], str(step), str(lr),
                                                       'residual.pt')

    # "color_context": the color information of different regions in the layout and their corresponding subjects,
    # along with the weight for strengthening the signal of target subject. (default: 2.5).
    color_context = {
        color: [char['unique_token'], 2.5] for char, color in
        zip(involved_characters, ['255,0,0', '0,0,255', '0,255,0'][:len(involved_characters)])
    }
    subject_list = [[char['unique_token'], prompt.split().index(char['unique_token']) + 1] for char in
                    involved_characters if char['unique_token'] in prompt]

    guidance_config = [{
        "prompt": prompt,
        "residual_dict": file_dict,
        "color_context": color_context,
        "guidance_steps": step2,
        "guidance_weight": guidance_weight,
        "weight_negative": weight_negative,
        "layout": prompt_layout_path,
        "subject_list": subject_list
    }
    ]

    json_file_path = os.path.join(os.getcwd(), filename, f"guidance_config_{filename}.json")

    with open(json_file_path, "w") as f:
        json.dump(guidance_config, f, indent=4)

    return json_file_path


def generate_yaml_file_mix_of_show(output_dir, character):
    # this yaml file is built according to instruction provided by Mix of show GitHub readme.
    filename = 'mix_of_show'
    repo_file_name = 'mix_of_show_git_repository'
    directory = os.path.join(os.getcwd(), filename, repo_file_name,
                             'options', 'train', 'EDLoRA', 'real', '8101_EDLoRA_potter_Cmix_B4_Repeat500.yml')

    with open(directory, 'r') as file:
        data = yaml.safe_load(file)

    # Modify the 'concept_list' and '<TOK>' parts
    data['name'] = f'EDLoRA_{character.name}'
    data['datasets']['train']['concept_list'] = os.path.join(output_dir, f'{character.name}.json')
    data['datasets']['train']['replace_mapping']['<TOK>'] = f'<{character.unique_token}1>+<{character.unique_token}2>'
    if character.gender == 'M':
        data['datasets']['val_vis']['prompts'] = os.path.join(os.getcwd(), filename, repo_file_name,
                                                              'datasets/validation_prompts/single-concept/characters/test_man.txt')
    else:
        data['datasets']['val_vis']['prompts'] = os.path.join(os.getcwd(), filename, repo_file_name,
                                                              'datasets/validation_prompts/single-concept/characters/test_girl.txt')
    data['datasets']['val_vis']['replace_mapping']['<TOK>'] = f'<{character.unique_token}1>+<{character.unique_token}2>'

    data['models']['new_concept_token'] = f'<{character.unique_token}1>+<{character.unique_token}2>'
    data['models']['initializer_token'] = '<rand-0.013>+girl' if character.gender == 'F' else '<rand-0.013>+man'
    data['models']['pretrained_path'] = os.path.join(os.getcwd(), filename, repo_file_name,
                                                     'experiments/pretrained_models/chilloutmix')

    yaml_path = os.path.join(output_dir, f'EDLoRA_{character.name}.yaml')

    # Write the modified data to a new YAML file
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file)

    return yaml_path


def generate_photo_mask(output_dir):
    """
    generate binary masks for photos
    :param output_dir: directory to store the masks
    :return: None
    """
    directory = os.path.join(output_dir, 'image')

    # check if all photos has corresponding masks
    if len(os.listdir(directory)) != len(os.listdir(os.path.join(output_dir, 'mask'))):

        # get the file name of the image
        for image_file_name in os.listdir(directory):

            # Load the image
            img = cv2.imread(os.path.join(directory, image_file_name))

            # get the yolo model
            model = YOLO('yolov8m-seg.pt')

            # configure the model
            results = model.predict(source=img.copy(), save=True, save_txt=False, stream=True)

            # use the predictions
            for result in results:
                # get array results
                masks = result.masks.data
                boxes = result.boxes.data

                # extract classes
                clss = boxes[:, 5]
                # get indices of results where class is 0 (people in COCO)
                people_indices = torch.where(clss == 0)

                # use these indices to extract the relevant masks
                people_masks = masks[people_indices]

                # scale for visualizing results
                people_mask = torch.any(people_masks, dim=0).int() * 255
                path = os.path.join(output_dir, 'mask', '{}.png'.format(image_file_name.split('.')[0]))

                cv2.imwrite(path, people_mask.cpu().numpy())

                im = Image.open(path)
                im = im.resize(tuple(reversed(img.shape[:2])))
                im.save(path)


def generate_captions(output_dir, character):
    directory = os.path.join(output_dir, 'image')
    if len(os.listdir(directory)) != len(os.listdir(os.path.join(output_dir, 'captions'))):

        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        for image_file_name in os.listdir(directory):
            image_file_path = os.path.join(directory, image_file_name)
            image = Image.open(image_file_path)

            inputs = processor(image, return_tensors="pt").to(device, torch.float16)

            generated_ids = model.generate(**inputs)  # , max_new_tokens=20
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            # add character unique tokens to captions
            words = generated_text.split()  # Split the string into words
            target_word = 'man' if character.gender == 'M' else 'woman'
            incorrect_recognition = True
            for i, word in enumerate(words):
                if word == target_word:
                    words.insert(i, f'<{character.unique_token}1>+<{character.unique_token}2>')
                    incorrect_recognition = False
                    break

            # Join the words back into a string
            generated_text = ' '.join(words)

            if incorrect_recognition:
                # in case of incorrect captions were generated, replace it with a standard token
                generated_text = f'<{character.unique_token}1>+<{character.unique_token}2> ' + target_word

            with open(os.path.join(output_dir, 'captions', f"{image_file_name.split('.')[0]}.txt"), "w") as file:
                # Write each piece of information about the photos to the file
                for info in generated_text:
                    file.write(str(info))


def move_previously_generated_images_to(output_dir):
    """
    move previously generated image to 'old' folder in output_dir directory
    :param output_dir: directory of the generated images
    :return: None
    """
    # Create 'old' folder if it doesn't exist
    old_folder = os.path.join(output_dir, "old")
    if not os.path.exists(old_folder):
        os.makedirs(old_folder)

    # Iterate over files in the directory
    for filename in os.listdir(output_dir):
        if filename.endswith(".png"):
            # Rename the file with random digit
            file_path = os.path.join(output_dir, filename)
            file_dir, file_name = os.path.split(file_path)
            file_name, file_ext = os.path.splitext(file_name)
            random_digit = str(random.randint(1, 19))
            new_filename = f"{file_name}_{random_digit}{file_ext}"
            # Move the file to 'old' folder
            old_file_path = os.path.join(old_folder, new_filename)
            shutil.move(os.path.join(output_dir, filename), old_file_path)


def check_file_upload(directory, file_name, file_name_2=None):
    """
    Check if a file exists in a specific directory
    If the file exists, return True
    Otherwise, ask the user to upload the file and wait for confirmation
    """

    def file_exists(file_path):
        return os.path.exists(file_path)

    file_path = os.path.join(directory, file_name)
    file_path_2 = os.path.join(directory, file_name_2) if file_name_2 else None

    while not (file_exists(file_path) or (file_name_2 and file_exists(file_path_2))):
        missing_file = file_name_2 if file_name_2 and not file_exists(file_path_2) else file_name
        print(f"{missing_file} is not located in the {directory} directory. Please upload it before continuing.")
        input("Press Enter when the file has been uploaded...")
        print("Verifying file upload...")
        time.sleep(1)  # Wait for 1 second to allow the file to be uploaded

    uploaded_file = file_name if file_exists(file_path) else file_name_2
    print(f"File {uploaded_file} has been successfully uploaded to {directory}.")
    return True


def test_if_file_in_place(key, args, characters, method_name):
    """
    test if files for all or one method is in place before inference
    :param key: number of characters
    :param args: fine-tuned parameters
    :param characters: list of characters
    :param method_name: method name either 'all' or individual method
    :return: None
    """
    # check if textual inversion files are in place before inference
    if method_name == 'textual_inversion' or method_name == 'all':
        file_name = 'learned_embeds.safetensors'
        count = 1
        condition = True

        # check if the files are there for each character
        for char in characters:
            directory = os.path.join(os.getcwd(), 'textual_inversion',
                                     str(args[key]['textual_inversion']['training_steps']),
                                     str(args[key]['textual_inversion']['lr']), char.name)
            if method_name == 'textual_inversion':
                condition = os.path.exists(os.path.join(directory, file_name)) and condition
                if count == len(characters):
                    return condition
                count += 1
            else:
                check_file_upload(directory, file_name)

    # check if LORA files are in place before inference
    if method_name == 'LORA' or method_name == 'all':
        file_name = 'adapter_model.safetensors'
        count = 1
        condition = True

        # check if the files are there for each character
        for char in characters:
            directory = os.path.join(os.getcwd(), 'LORA',
                                     str(args[key]['LORA']['training_steps']),
                                     str(args[key]['LORA']['lr']), char.name, 'unet')
            if method_name == 'LORA':
                condition = os.path.exists(os.path.join(directory, file_name)) and condition
                if count == len(characters):
                    return condition
                count += 1
            else:
                check_file_upload(directory, file_name)

    # check if Dreambooth files are in place before inference
    if method_name == 'dreambooth' or method_name == 'all':
        file_name = 'diffusion_pytorch_model.bin'
        file_name_2 = 'diffusion_pytorch_model.safetensors'
        directory = os.path.join(os.getcwd(), 'dreambooth', str(args[key]['dreambooth']['lr']),
                                 str(args[key]['dreambooth']['training_steps']), 'unet')
        if method_name == 'dreambooth':
            return os.path.exists(os.path.join(directory, file_name)) or os.path.exists(
                os.path.join(directory, file_name_2))
        else:
            check_file_upload(directory, file_name, file_name_2)

    # check if custom diffusion files are in place before inference
    if method_name == 'custom_diffusion' or method_name == 'all':
        count = 1
        condition = True

        # check if the files are there for each character
        for char in characters:
            file_name = '<' + char.unique_token + '>.bin'
            directory = os.path.join(os.getcwd(), 'custom_diffusion',
                                     str(args[key]['custom_diffusion']['training_steps']),
                                     str(args[key]['custom_diffusion']['lr']))
            if method_name == 'custom_diffusion':
                condition = os.path.exists(os.path.join(directory, file_name)) and condition
                if count == len(characters):
                    return condition
                count += 1
            else:
                check_file_upload(directory, file_name)

    # check if cones 2 files are in place before inference
    if method_name == 'cones_2' or method_name == 'all':
        file_name = 'diffusion_pytorch_model.bin'
        count = 1
        condition = True

        # check if the files are there for each character
        for char in characters:
            directory = os.path.join(os.getcwd(), 'cones_2', char.unique_token,
                                     str(args[key]['cones_2']['training_steps']),
                                     str(args[key]['cones_2']['lr']), 'unet')
            if method_name == 'cones_2':
                condition = os.path.exists(os.path.join(directory, file_name)) and condition
                if count == len(characters):
                    return condition
                count += 1
            else:
                check_file_upload(directory, file_name)

    # check if mix of show files are in place before inference
    if method_name == 'mix_of_show' or method_name == 'all':
        file_name = 'diffusion_pytorch_model.safetensors'
        directory = os.path.join(os.getcwd(), 'mix_of_show',
                                 str(args[key]['mix_of_show']['optimize_textenc_iters']),
                                 str(args[key]['mix_of_show']['optimize_unet_iters']),
                                 'trained_model', 'combined_model_base', 'unet')
        if method_name == 'mix_of_show':
            return os.path.exists(os.path.join(directory, file_name))
        else:
            check_file_upload(directory, file_name)
    file_name = 'bpe_simple_vocab_16e6.txt.gz'
    directory = os.path.join(os.getcwd(), '*', 'lib', '*', 'site-packages', 'hpsv2', 'src',
                             'open_clip')
    matching_paths = glob.glob(directory)
    if matching_paths:
        for path in matching_paths:
            check_file_upload(path, file_name)
            break


if __name__ == "__main__":
    print('add function you would like to check')
