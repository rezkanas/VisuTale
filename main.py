# third-party imports
import os
import torch
from dotenv import load_dotenv, find_dotenv

# local import
from character import CHARACTER
from visual_story import (textual_inversion, textual_inversion_inference, dream_booth,
                          dream_booth_inference, custom_diffusion, custom_diffusion_inference, cones_2_training,
                          mix_of_show_training_multi_character, mix_of_show_inference, DreamBooth_with_LoRA,
                          DreamBooth_with_LoRA_inference, Lambda_ECLIPSE_Prior, cones_2_inference)
import evaluate_story
from utilities import prep_work, test_if_file_in_place
from textual_story import STORYTELLING
import analysis

_ = load_dotenv(find_dotenv())  # read local .env file

# Input: characters, story logline, nicknames
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
    unique_token='znrz',
    random_photo=os.path.join(photos_folder_1, 'znrz (1).JPG')
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

#  Mark the variable below True for training the models using multiple method
Train = False

# A Dictionary of the Fine-Tuned Parameters for Each Method
# The parameters are indexed based on the number of characters: two, three, and four.
args = {
    '2':
        {'textual_inversion': {
            'training_steps': 3000,
            'lr': 5e-4,
            'sampling_steps': 100,
            'CFG_guidance': 7.5
        },
            'dreambooth': {
                'training_steps': 2700,
                'lr': 1e-6,
                'CFG_guidance': 7.5,
                'sampling_steps': 75},
            'Lambda_ECLIPSE': {
                'sampling_steps': 50,
                'guidance': 4},
            'LORA': {
                'training_steps': 1000,
                'sampling_steps': 75,
                'combination_type': 'dare_ties',
                'CFG_guidance': 7.5,
                'lr': 5e-5},
            'custom_diffusion': {
                'training_steps': 2300,
                'lr': 3e-5,
                'CFG_guidance': 7.5,
                'sampling_steps': 75},
            'cones_2': {
                'training_steps': 4000,
                'lr': 5e-6,
                'guidance_steps': 75,
                'guidance_weight': 0.12,
                'weight_negative': -5e7,
            },
            'mix_of_show': {
                'optimize_textenc_iters': 500,
                'optimize_unet_iters': 50}},
    '3':
        {'textual_inversion': {
            'training_steps': 3000,
            'lr': 5e-4,
            'sampling_steps': 100,
            'CFG_guidance': 7.5
        },
            'dreambooth': {
                'training_steps': 2700,
                'lr': 1e-6,
                'CFG_guidance': 7.5,
                'sampling_steps': 75},
            'Lambda_ECLIPSE': {
                'sampling_steps': 50,
                'guidance': 4},
            'LORA': {
                'training_steps': 1000,
                'sampling_steps': 75,
                'combination_type': 'dare_ties',
                'lr': 5e-5},
            'custom_diffusion': {
                'training_steps': 2300,
                'lr': 3e-5,
                'CFG_guidance': 7.5,
                'sampling_steps': 75},
            'cones_2': {
                'training_steps': 4000,
                'lr': 5e-6,
                'guidance_steps': 75,
                'guidance_weight': 0.16,
                'weight_negative': -5e7,
            },
            'mix_of_show': {
                'optimize_textenc_iters': 500,
                'optimize_unet_iters': 50}},
    '4':
        {'textual_inversion': {
            'training_steps': 3000,
            'lr': 5e-4,
            'sampling_steps': 100,
            'CFG_guidance': 7.5
        },
            'dreambooth': {
                'training_steps': 2700,
                'lr': 1e-6,
                'CFG_guidance': 7.5,
                'sampling_steps': 75},
            'Lambda_ECLIPSE': {
                'sampling_steps': 50,
                'guidance': 4},
            'LORA': {
                'training_steps': 1000,
                'sampling_steps': 75,
                'combination_type': 'dare_ties',
                'lr': 5e-5},
            'custom_diffusion': {
                'training_steps': 2300,
                'lr': 3e-5,
                'CFG_guidance': 7.5,
                'sampling_steps': 75},
            'cones_2': {
                'training_steps': 4000,
                'lr': 5e-6,
                'guidance_steps': 75,
                'guidance_weight': 0.16,
                'weight_negative': -5e7,
            },
            'mix_of_show': {
                'optimize_textenc_iters': 500,
                'optimize_unet_iters': 50}},
}

# here is where you define which characters you will use to create the
# story/train and inference the models
for multi_character in [[character_1, character_2]]:
    key = str(len(multi_character))

    # make a story/ prompts using GPT_4
    print('##   Generating GPT_4 story  ##')
    story_instance = STORYTELLING(characters=multi_character, GPT_4=True)
    story = story_instance.generate_story()
    story_instance.chunk_story(story)
    print('done generating GPT_4 story and prompts and now Llama 3 ...')

    # make a story/ prompts using LLaMa3
    success = False
    print('##   Generating Llama 3 story    ##')
    while not success:
        try:
            story_instance = STORYTELLING(characters=multi_character, GPT_4=False)
            _, story = story_instance.generate_story()
            story_instance.chunk_story(story)
            success = True
        except Exception as e:
            print(f"Caught an exception: {e}")
            print("Retrying to query HuggingChat API...")
    print('done with Llama 3 generation')

    if Train:

        print('##############################')
        print('########## TRAINING STARTS #########')
        print('##############################')

        # train textual inversion model
        while not test_if_file_in_place(key=key, args=args, characters=multi_character,
                                        method_name='textual_inversion'):
            print('TEXTUAL INVERSION TRAINING STARTS... ')
            textual_inversion(characters=multi_character,
                              training_steps=args[key]['textual_inversion']['training_steps'],
                              lr=args[key]['textual_inversion']['lr'])
            torch.cuda.empty_cache()

        # train LORA model
        while not test_if_file_in_place(key=key, args=args, characters=multi_character, method_name='LORA'):
            print('LORA TRAINING STARTS... ')
            DreamBooth_with_LoRA(characters=multi_character,
                                 training_steps=args[key]['LORA']['training_steps'],
                                 lr=args[key]['LORA']['lr'])
            torch.cuda.empty_cache()

        # train dreambooth
        while not test_if_file_in_place(key=key, args=args, characters=multi_character, method_name='dreambooth'):
            print('DREAMBOOTH TRAINING STARTS... ')
            dream_booth(characters=multi_character,
                        training_steps=args[key]['dreambooth']['training_steps'],
                        lr=args[key]['dreambooth']['lr'])
            torch.cuda.empty_cache()

        # train custom diffusion
        while not test_if_file_in_place(key=key, args=args, characters=multi_character, method_name='custom_diffusion'):
            print('CUSTOM DIFFUSION TRAINING STARTS... ')
            custom_diffusion(characters=multi_character,
                             training_steps=args[key]['custom_diffusion']['training_steps'],
                             lr=args[key]['custom_diffusion']['lr'])
            torch.cuda.empty_cache()

        # train cones 2
        while not test_if_file_in_place(key=key, args=args, characters=multi_character, method_name='cones_2'):
            print('CONE 2 TRAINING STARTS... ')
            cones_2_training(characters=multi_character,
                             training_steps=args[key]['cones_2']['training_steps'],
                             lr=args[key]['cones_2']['lr'])
            torch.cuda.empty_cache()

        # train mix of show
        while not test_if_file_in_place(key=key, args=args, characters=multi_character, method_name='mix_of_show'):
            print('MIX OF SHOW TRAINING STARTS... ')
            mix_of_show_training_multi_character(characters=multi_character,
                                                 optimize_textenc_iters=args[key]['mix_of_show'][
                                                     'optimize_textenc_iters'],
                                                 optimize_unet_iters=args[key]['mix_of_show']['optimize_unet_iters'])
            torch.cuda.empty_cache()

        print("TRAINING has been finalized, moving to inference...")

    else:
        # prep venv, folders and install dependencies
        filename = 'LORA'
        repo_file_name = 'huggingface_peft'
        repo_url = "https://github.com/huggingface/peft"
        # prep work for each character
        for char in multi_character:
            _, _, _ = prep_work(filename, repo_file_name, repo_url, char,
                                args[key]['LORA']['training_steps'],
                                args[key]['LORA']['lr'])

        filename = 'textual_inversion'
        repo_file_name = 'huggingface_diffusers'
        repo_url = "https://github.com/huggingface/diffusers.git"
        _, _ = prep_work(filename, repo_file_name, repo_url, multi_character,
                         args[key]['textual_inversion']['training_steps'], args[key]['textual_inversion']['lr'])

        filename = 'dreambooth'
        repo_file_name = 'ShivamShrirao_diffusers'
        repo_url = "https://github.com/rezkanas/diffusers.git"
        _, _, _ = prep_work(filename, repo_file_name, repo_url, multi_character,
                            args[key]['dreambooth']['training_steps'], args[key]['dreambooth']['lr'])

        filename = 'custom_diffusion'
        repo_file_name = 'huggingface_diffusers'
        repo_url = "https://github.com/huggingface/diffusers.git"
        _, _, _ = prep_work(filename, repo_file_name, repo_url, multi_character,
                            args[key]['custom_diffusion']['training_steps'], args[key]['custom_diffusion']['lr'])

        filename = 'cones_2'
        repo_file_name = 'cones_2_git_repository'
        repo_url = "https://github.com/rezkanas/Cones-V2.git"
        # prep work for each character
        for char in multi_character:
            _ = prep_work(filename, repo_file_name, repo_url, char, args[key]['cones_2']['training_steps'])

        filename = 'mix_of_show'
        repo_file_name = 'mix_of_show_git_repository'
        repo_url = "https://github.com/rezkanas/Mix-of-Show.git"
        for char in multi_character:
            _, _ = prep_work(filename, repo_file_name, repo_url, char)

        filename = 'Lambda_ECLIPSE'
        repo_file_name = 'lambda_eclipse_inference_git_repo'
        repo_url = "https://github.com/rezkanas/lambda-eclipse-inference.git"
        _, _ = prep_work(filename, repo_file_name, repo_url, multi_character)

        # Prompt the user to perform an action outside of Python
        while True:
            user_input = input(
                "Please complete the action number 10 as instructed in attached README file. Type 'yes' when done: ").strip().lower()
            if user_input == 'yes':
                test_if_file_in_place(key=key, args=args,
                                      characters=multi_character, method_name='all')
                break

        # Continue with the rest of the Python code
        print("Setup is done, Moving now to inference...")

    print('############################################################')
    print('#################### INFERENCE #############################')
    print('############################################################')

    # inference using trained LORAs
    print('LORA INFERENCE... ')
    DreamBooth_with_LoRA_inference(characters=multi_character,
                                   training_steps=args[key]['LORA']['training_steps'],
                                   sampling_steps=args[key]['LORA']['sampling_steps'],
                                   lr=args[key]['LORA']['lr'],
                                   CFG_guidance=args[key]['LORA']['CFG_guidance'],
                                   combination_type=args[key]['LORA']['combination_type'])
    torch.cuda.empty_cache()

    # inference using trained textual inversion
    print('TEXTUAL INVERSION INFERENCE... ')
    textual_inversion_inference(characters=multi_character,
                                training_steps=args[key]['textual_inversion']['training_steps'],
                                sampling_steps=args[key]['textual_inversion']['sampling_steps'],
                                lr=args[key]['textual_inversion']['lr'],
                                CFG_guidance=args[key]['textual_inversion']['CFG_guidance'])
    torch.cuda.empty_cache()

    # inference using trained dreambooth model
    print('DREAMBOOTH INFERENCE... ')
    dream_booth_inference(characters=multi_character,
                          training_steps=args[key]['dreambooth']['training_steps'],
                          sampling_steps=args[key]['dreambooth']['sampling_steps'],
                          CFG_guidance=args[key]['dreambooth']['CFG_guidance'],
                          lr=args[key]['dreambooth']['lr'])
    torch.cuda.empty_cache()

    # inference using trained custom diffusion model
    print('CUSTOM DIFFUSION INFERENCE... ')
    custom_diffusion_inference(characters=multi_character,
                               training_steps=args[key]['custom_diffusion']['training_steps'],
                               sampling_steps=args[key]['custom_diffusion']['sampling_steps'],
                               CFG_guidance=args[key]['custom_diffusion']['CFG_guidance'],
                               lr=args[key]['custom_diffusion']['lr'])
    torch.cuda.empty_cache()

    # inference using trained cones 2
    print('CONE 2 INFERENCE... ')
    cones_2_inference(characters=multi_character,
                      training_steps=args[key]['cones_2']['training_steps'],
                      guidance_steps=args[key]['cones_2']['guidance_steps'],
                      guidance_weight=args[key]['cones_2']['guidance_weight'],
                      weight_negative=args[key]['cones_2']['weight_negative'],
                      lr=args[key]['cones_2']['lr'])
    torch.cuda.empty_cache()

    # inference using trained mix of show
    print('MIX OF SHOW INFERENCE... ')
    mix_of_show_inference(characters=multi_character,
                          optimize_textenc_iters=args[key]['mix_of_show']['optimize_textenc_iters'],
                          optimize_unet_iters=args[key]['mix_of_show']['optimize_unet_iters'])
    torch.cuda.empty_cache()

    # inference Lambda ECLIPSE Prior
    Lambda_ECLIPSE_Prior(characters=multi_character,
                         sampling_steps=args[key]['Lambda_ECLIPSE']['sampling_steps'],
                         guidance=args[key]['Lambda_ECLIPSE']['guidance'])
    torch.cuda.empty_cache()

    # evaluate all generated images, select top 5 ensemble stories, final evaluation
    evaluate_story.main(characters=multi_character)

    # the final analysis to get the graphs and plots
    analysis.main()
