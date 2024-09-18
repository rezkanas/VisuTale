# third-party imports
import os
from dotenv import load_dotenv, find_dotenv
import json
import re
import random

# local import
from character import CHARACTER
from utilities import query_gpt, generate_prompt_layout, query_LLM

# read local .env file
_ = load_dotenv(find_dotenv())


class STORYTELLING:
    """a class having multiple methods to make story about defined characters."""

    def __init__(self, characters: list,
                 logline: str = None, GPT_4: bool = None,
                 active_llm: int = 1):
        """
        a class that generate the story then chunk it into prompts

        :param characters: list of characters used to build the story.
        :param logline: story logline if provided by the user, otherwise, LLM would create one.
        :param GPT_4: boolean when True then GPT-4 will be used otherwise LLaMa 3 will be adopted.
        :param active_llm: used only when selecting the best LLM to use for the experiment.
        """

        self.GPT_4 = GPT_4

        # create story logline in case user has not specified one
        # Create it using GPT-4 in case GPT_4 is True
        if (logline is None or not isinstance(logline, str)) and GPT_4:
            self.logline = self.generate_logline(characters, GPT_4)
            self.LLM = 'GPT_4'

        # else create it using LLaMa 3
        elif logline is None or not isinstance(logline, str):
            self.LLM, self.logline = self.generate_logline(characters, GPT_4, active_llm)
            if self.LLM == 'Meta_Llama_3_70B_Instruct':
                self.LLM = 'Llama_3'
            self.active_llm = active_llm
        else:
            self.logline = logline

        self.characters = characters
        self.tag = 'two_characters' if len(characters) == 2 else (
            'three_characters' if len(characters) == 3 else 'four_characters')

        path = os.path.join(os.getcwd(), "story", self.tag, self.LLM)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # store locally the story logline
        with open(os.path.join(path, f"story_logline_{self.LLM}.txt"), "w") as f:
            f.write(self.logline.strip())

        # below dictionary has been obtained through multiple iterations to choose the suitable LLM
        number_of_scenes_dic = {
            'GPT_4': {'two_characters': 11, 'three_characters': 10, 'four_characters': 9},
            'c4ai_command_r_plus': {'two_characters': 9, 'three_characters': 7, 'four_characters': 6},
            'Llama_3': {'two_characters': 7, 'three_characters': 5, 'four_characters': 4},
            'zephyr_orpo_141b_A35b_v01': {'two_characters': 8, 'three_characters': 7, 'four_characters': 6},
            'Mixtral_8x7B_Instruct_v01': {'two_characters': 8, 'three_characters': 7, 'four_characters': 6},
            'Nous_Hermes_2_Mixtral_8x7B_DPO': {'two_characters': 4, 'three_characters': 3, 'four_characters': 3},
            'gemma_11_7b_it': {'two_characters': 3, 'three_characters': 2, 'four_characters': 2},
            'Mistral_7B_Instruct_v02': {'two_characters': 7, 'three_characters': 5, 'four_characters': 5},
            'Phi_3_mini_4k_instruct': {'two_characters': 2, 'three_characters': 7, 'four_characters': 6},
        }

        # Based on trial and error, determine the number of scenes each LLM can fully generate, primarily to comply with response length constraints.
        self.number_of_scenes = number_of_scenes_dic[self.LLM][self.tag]

    @staticmethod
    def generate_logline(characters, GPT_4, active_llm=1):
        """generate story logline in case user has not provide one"""
        delimiter = "####"

        # write a system message to instruct LLM 
        system_message = (
            "you are a celebrated storyteller renowned for your talent in crafting captivating logline, ensure "
            "seamless integration of all characters outlined in the user message. Remain faithful to the traits "
            f"assigned to each character, as detailed within the user message following the {delimiter}. "
            "Construct a story logline comprising a maximum of six sentences, vividly bringing the characters and "
            "their distinct traits to life. you should refer implicitly to the genre of stories, e.g. adventure, "
            "mystery, romance, fantasy, science fiction, historical fiction, thriller, horror, drama or else.."
            "reply to user with logline only, Do not add any details before or after."
        )

        # preparing user message
        character_names = " and ".join(
            [character.name for character in characters]
        )

        # include character names in user prompt
        user_message = f"""you are a storyteller. Write a story logline featuring {len(characters)} characters namely; {character_names}.{delimiter}"""

        list_of_traits = [", ".join(
            [value for values_list in character.traits.values() for value in values_list])
            for character in characters
        ]
        list_of_gender = ['man' if character.gender == "M" else 'woman' for character in characters]

        # include character traits and their genders in user prompt
        for character, gender, character_trait in zip(characters, list_of_gender, list_of_traits):
            user_message += '{} is a {} known for being {}. '.format(character.name, gender, character_trait)

        # passing the prompt to LLM
        if GPT_4:
            messages = (system_message, user_message)
            return query_gpt(messages,
                             model="gpt-3.5-turbo",
                             temperature=0.7, top_p=0.8, max_tokens=2000)
        else:
            return query_LLM(system_prompt=system_message, prompt=user_message, default_llm=active_llm)

    def generate_story(self):
        """generate a story based on the story logline"""
        delimiter = "####"

        # Write a system message to instruct LLM
        system_message = (
            "You are a celebrated storyteller known for your ability to craft lengthy narratives. Adhere closely to "
            "the provided story logline. Ensure seamless integration of all characters outlined in the logline into "
            "your tale. Remain faithful to the traits assigned to each character, as described within the logline. "
            "Your primary objective is to produce an extensive story that captivates the audience's imagination. Let "
            "your creativity flourish as you embark on this storytelling journey. Craft a story that vividly brings "
            "the characters and their distinct traits to life. Stick to the story genre as proposed in the story logline, "
            "do not rush to the main plot. Build a story arc. Avoid negative clichés, repetitive language, and flat characters. "
            "Reply to the user with the story only. Do not add any details before or after."
            )

        # preparing user message
        user_message = f"""
        write me a story whose logline is described after the delimiter: {delimiter}{self.logline}"""

        # passing the prompt to LLM
        if self.GPT_4:
            messages = (system_message, user_message)
            return query_gpt(messages, model="gpt-4o",
                             temperature=0.7, top_p=0.8, max_tokens=4096)
        else:
            return query_LLM(system_prompt=system_message, prompt=user_message, default_llm=self.active_llm)

    def chunk_story(self, story):
        """ the main function in this class, it is responsible to chunk the story down to useful prompts"""

        # replace character names with their unique token
        replace_name_with_unique_token = ' and replace '.join(
            [
                f'{char.name} with {char.unique_token} man' if char.gender == 'M' else f'{char.name} with {char.unique_token} woman'
                for char in self.characters])

        path = os.path.join(os.getcwd(), "story", self.tag, self.LLM)
        with open(os.path.join(path, f"story_{self.LLM}.txt"), "w") as f:
            f.write(story.strip())

        # based on 'OPPENLAENDER, J, A Taxonomy of Prompt Modifiers for Text-To-Image Generation, 14 June 2023, University of Jyväskylä'
        style_modifier = """You may use a style modifier however be careful to stick to one style modifier and use it for all prompts and never use dissimilar art styles. The role of Style modifiers is to enhance prompts to generate images in specific styles or artistic mediums consistently. These modifiers encompass characteristics like "oil painting" or "mixed media," and can even mimic renowned artists like Francisco Goya. Examples include "oil on canvas," "#pixelart," "hyperrealistic," "abstract painting," "surreal," "Cubism," "cabinet card," "in the style of a cartoon," "by Claude Lorrain," and "in the style of Hudson River School." They cover art periods, schools, styles, materials, techniques, and artists, with modifiers like "by Greg Rutkowski" and "by James Gurney" being popular for text-to-image art, ensuring specific styles and quality."""
        quality_boosters = """Adding quality boosters to a prompt can enhance the aesthetic appeal and level of detail in generated images. These modifiers include terms like "trending on artstation," "award-winning," "masterpiece," "highly detailed," "awesome," "#wow," "epic," and "rendered in Unreal Engine." You can also include additional descriptors or "extra fluff" in the prompt to increase verbosity and improve image quality. However, be aware that increasing verbosity may result in less control over the subject matter. For example, enhancing the prompt "painting of an exploding heart" with modifiers like "highly detailed, eclectic, fiery, vfx, rendered in octane, postprocessing, 8k" can potentially elevate the quality and detail of the generated image."""
        repeating_terms = """Repeated terms in prompts improve results. For instance, "space whale. a whale in space" yields better outcomes than single terms like "space" or "whale." Varying phrasing and synonyms helps. Prompts like "a very very very very very beautiful landscape," which produces better images ."""

        # base on https://github.com/sharonzhou/long_stable_diffusion/blob/master/effective_prompts_fs.txt
        local_story_path = os.path.join(os.getcwd(), "story")
        with open(os.path.join(local_story_path, "effective_prompts.txt"), "r") as f:
            prompts_examples = f.readlines()

        # Select 4 random lines from the file
        random_prompts = random.sample(prompts_examples, 4)

        # Join the selected lines with '. Also, ' as the separator
        prompts_examples = '. Also, '.join([line.strip() for line in random_prompts])

        # collected from https://medium.com/stablediffusion/100-negative-prompts-everyone-are-using-c71d0ba33980
        with open(os.path.join(local_story_path, "negative_prompt_examples.txt"), "r") as f:
            negative_prompt_examples = f.read()

        delimiter = "####"

        # write a system message to instruct LLM to follow a chain of thoughts

        system_message = (
            "Follow these steps to execute the task:  "
            # first step divide the story to scenes

            f"Step 1: As a skilled screenwriter, divide the user\'s provided story into at least {self.number_of_scenes} scenes. Character names"
            f"must be included if relevant, avoiding pronouns. Ensure all story details are preserved"

            # second step transform the scenes to visually rich prompts that include certain elements and sounds similar to provided examples 
            f"Step 2: you are a master artist, well-versed in artistic terminology with a vast vocabulary for being able to "
            f"describe visually things that you see. Utilize the scenes created in Step 1 to generate inter- and intra- coherent"
            f"prompts that will be used to generate images using stable diffusion model. Keep prompts concise while retaining scene "
            f"details, including character names if applicable. Provide the output in noun sentences, separated by commas."
            f"When applicable, maintain consistent backgrounds and styles description."
            f'Each prompt should follow a formula, including elements such as the subject, emotions, verb, adjectives,'
            f'environment, lighting, photography type, and quality. Ensure that subject and environment align and emphasize '
            f'each other. Between the two {delimiter} delimiters, there are samples for how prompts should sound like. I would'
            f' like you to analyse the components of these prompts, the sentence structure, how they are laid out and the common'
            f' pattern between all of them. {delimiter}{prompts_examples}{delimiter}. Be mindful that stable diffusion pays more '
            f'attention to what\'s at the beginning of the prompt and that attention declines the closet to the end of the positive'
            f' prompt that you get. {style_modifier}. On the other hand, {quality_boosters}.And only when necessary, {repeating_terms}.'

            # third step transform the prompts to a certain python format while ensuring story coherence and consistency between prompts 
            f"Step 3: Replace character names in each prompt with unique pseudonyms as follows: {replace_name_with_unique_token}."
            f"Convert the prompts into a list of dictionaries in Python format. Each dictionary should represent one prompt, with"
            f" keys preceding the colon and values following."
        )

        system_message_1 = \
            f""" You must include all the key, value pairs below.
               ```python
               {{
                    "prompt_number": <number>,
                    "prompt_text": "Generate detailed Stable Diffusion prompts based on the guidelines from Step 3. When creating scenes involving characters, use pseudonyms for character names and integrate these character details fluidly within the scene description. Ensure global consistency across all prompts by maintaining the same style throughout. Describe characters' clothing and the background, ensuring a consistent story across prompts. Include details on the style, media type, color palette, tint/ambience, saturation/contrast, and overall feel but stick to the same across prompts whenever possible. Use prompt modifiers and noun sentences. Ensure global consistency across all prompts by maintaining the same style throughout.",
                    "Characters_involved": ['<a list of pseudonym of characters involved in this prompt>'],
                    "negative_prompt": "<Specify what you don\'t want to see in the generated image using keywords, use affirmative noun sentences with no negation, do not refer to character names or pseudonyms>",                    
                }}
               ```
                Examples of negative prompt include: {negative_prompt_examples}. When applicable, make sure to append the 2 above python code in one dictionary per prompt and collect all 
                prompts dictionaries in a single list.  A story will have a certain identity and its prompts should preserve it by ensuring global consistency 
                across prompts in terms of style including style modifiers, media type, color palette, tint/ambience, saturation/contrast, and overall feel. 
                This consistency should extend to characters, encompassing character attire and physical appearance. Ensure 
                generating a consistent series of prompts based on the before and after relationship between them. 
                Use the following format: <response to user request only return a Python list of prompt dictionaries. Make sure the style 
                is consistent across all prompts. Do not add any details before or after.>
                 """

        # preparing user message that include the story
        user_message = f"""
        break the story after the delimiter to minimum {self.number_of_scenes} visually rich prompts. Ensure consistency between prompts when 
        describing characters, style and background {delimiter}{story}"""

        # GPT-4 can handle conversation history therefore, system message was breaking into two conversation rounds in order to give the model more flexibility to use the response length
        system_message_2 = \
            f""" you are a master artist, well-versed in artistic terminology with a vast vocabulary for being able to 
                describe visually things that you see. Utilize previous response to create further details about the prompts.
                In your response, you must include all the key, value pairs below.
               ```python
               {{
                    "context_prompt": "<Describe the context of this prompt including background and style, do not mention character names or pseudonym>",
                }}
               ```
                For prompts involving characters, make sure to append the following information to the previous dictionary for each character mentioned in the prompt.                            
                ```python
                {{
                    "region_<number>_prompt": "<Describe in detail the character's physical conditions, clothes, and stance, start with the pseudonym of the character, only refer to one character>",
                    'region_<number>_boundaries': <on a 768x768 image, specify in a python list the coordinates (x, y) of the lower-left corner, along with the width and height dimensions
                    on a 768x768 image, for the region where each character mentioned in the prompt will be positioned.>
                }}
                ```
                when applicable, make sure to append the 2 above python code in one dictionary per prompt and collect all prompts dictionaries in a single list, exclude previous response from your response. 
                 """
        # the user message passed in the second round of conversation
        user_message_2 = 'give more details about prompts you provided earlier'

        # the system message used by all other LLM (beside GPT-4) combining system_message_1 and system_message_2
        system_message_3 = \
            f""" You must include all the key, value pairs below.
               ```python
               {{
                    "prompt_number": <number>,
                    "prompt_text": "Generate detailed Stable Diffusion prompts based on the guidelines from Step 3. When creating scenes involving characters, use pseudonyms for character names and integrate these character details fluidly within the scene description. Ensure global consistency across all prompts by maintaining the same style throughout. Describe characters' clothing and the background, ensuring a consistent story across prompts. Include details on the style, media type, color palette, tint/ambience, saturation/contrast, and overall feel but stick to the same across prompts whenever possible. Use prompt modifiers and noun sentences. Ensure global consistency across all prompts by maintaining the same style throughout.",
                    "Characters_involved": ['<a list of pseudonym of characters involved in this prompt, list only the characters that has a pseudonym>'],
                    "negative_prompt": "<Specify what you don\'t want to see in the generated image using keywords, use affirmative noun sentences with no negation, do not refer to character names or pseudonyms>",
                    "context_prompt": "<Describe the context of this prompt including background and style, do not mention character names or pseudonym>",
                }}
               ```
                Examples of negative prompt include: {negative_prompt_examples}. For prompts involving characters, 
                append the following information to the previous dictionary for EACH character mentioned in the prompt.
                ```python
                {{
                    "region_<number>_prompt": "<Describe in detail the character's physical conditions, clothes, and stance, start with the pseudonym of the character, only refer to single character>",
                    'region_<number>_boundaries': <on a 768x768 image, specify in a python list the coordinates (x, y) of the lower-left corner, along with the width and height dimensions
                    on a 768x768 image, for the region where each character mentioned in the prompt will be positioned. 
                    Allocate substantial region for characters>
                }}
                ```
                Remember to include the 2 pairs for each character involved in the prompt.
                when applicable, make sure to append the 2 above python code in one dictionary per prompt and collect all
                prompts dictionaries in a single list. A story will have a certain identity and its prompts should 
                preserve it by ensuring global consistency across prompts in terms of style including style modifiers, 
                media type, color palette, tint/ambience, saturation/contrast, and overall feel.
                This consistency should extend to characters, encompassing character attire and physical appearance. Ensure
                generating a consistent series of prompts based on the before and after relationship between them.
                Use the following format: <response to user request only return a Python list of prompt dictionaries 
                obtained in step 3. Make sure the style is consistent across all prompts. Do not add any details before 
                or after. Do not show result of step 1 and 2.  Make sure that ALL region_<number>_prompt and 
                region_<number>_boundaries are included in the respective dictionary of python list.>
                 """

        # passing the prompt to LLM
        if self.GPT_4:
            success = False
            while not success:
                try:
                    # first round of conversation
                    messages = (system_message + system_message_1, user_message)
                    result = query_gpt(messages, model="gpt-4o", max_tokens=4096)  # gpt-4-turbo

                    # storing the first response locally
                    with open(os.path.join(path, f"1stPart_Finalresult_{self.LLM}.txt"), "w") as f:
                        f.write(result.strip())

                    # second round of conversation
                    messages = (system_message_2, user_message_2)
                    result_2 = query_gpt(messages, model="gpt-4o", max_tokens=4096,
                                         continue_conversation=True,
                                         previous_chat=(system_message + system_message_1, user_message, result))
                    # storing the second response locally
                    with open(os.path.join(path, f"2ndPart_Finalresult_{self.LLM}.txt"), "w") as f:
                        f.write(result_2.strip())

                    # parse the results
                    result_1 = self.parse_result(result)
                    result_2 = self.parse_result(result_2)
                    result = []

                    for dic1, dic2 in zip(result_1, result_2):
                        merged_dict = dic1.copy()
                        merged_dict.update(dic2)
                        result.append(merged_dict)

                    # store the prompts dictionary locally in a json file
                    STORYTELLING.generate_prompt_json(result, self.characters, self.tag, self.GPT_4, self.LLM)
                    success = True
                except Exception as e:
                    print(f"Caught an exception: {e}")
                    print("GPT-4 incomplete response")
                    print("Retrying query GPT-4...")

        else:
            # passing the prompt to LLM other than GPT-4
            success = False
            while not success:
                try:
                    _, result = query_LLM(system_prompt=system_message + system_message_3, prompt=user_message,
                                          default_llm=self.active_llm)

                    # storing the response locally
                    with open(os.path.join(path, f"final_result_{self.LLM}.txt"), "w") as f:
                        f.write(result.strip())

                    result = self.parse_result(result)

                    # store the prompts dictionary locally in a json file
                    STORYTELLING.generate_prompt_json(result, self.characters, self.tag, self.GPT_4, self.LLM)
                    success = True
                except Exception as e:
                    print(f"Caught an exception: {e}")
                    print("huggingchat sent an incomplete response")
                    print("Re-querying huggingchat API...")

    def parse_result(self, result):
        """
        although the response formate is defined, depending on the LLM, the response formate has been different hence
         information parsing is different

        """

        if '```python' in result or '```json' in result:
            result = re.findall(r'```(?:json|python)\n(.*?)\n```', result, re.DOTALL)
            result_1 = [eval(part) for part in result][0]

        elif '**Prompt 1:**' in result:
            result = re.findall(r'{(.*?)}', result)
            result = ['{' + part + '}' for part in result]
            result_1 = eval(result)
        elif '```' in result:
            try:
                result = re.findall(r'```(.*?)```', result, re.DOTALL)
                result_1 = eval(result[0])
            except:
                result = re.findall(r'```(.*?)```', result, re.DOTALL)
                result_1 = eval(result)
        elif '<' in result and '>' in result:
            result = result.split('>')[1]
            result_1 = eval(result)
        else:
            result_1 = eval(result)

        return result_1

    @staticmethod
    def generate_prompt_json(prompts, characters, tag, GPT_4, LLM) -> None:
        """
        Generate a JSON file containing prompts with unique IDs and placeholder for multiple fields that will be
        used across the experiment.

        :param prompts: list of dictionaries that were parsed from LLM response
        :param characters: list of characters used to build the story.
        :param tag: story tag
        :param GPT_4: boolean when True then GPT-4 will be used otherwise Llama 3 will be adopted.
        :param LLM: the name of the LLM used

        Returns:    None
        """

        # Create an empty list to store the prompts
        template = []

        for index, prompt in enumerate(prompts, start=1):
            prompt = {key.replace('\\', ''): value for key, value in prompt.items()}

            if not isinstance(prompt, dict):
                continue

            prompt_text = prompt['prompt_text'].replace('_', ' ').replace("[", "").replace("]", "")
            negative_prompt = 'EasyNegative, ' + prompt['negative_prompt'].replace("<", "").replace(">", "") + (
                ', deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, '
                'mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, '
                'extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, '
                'amputation, out of frame, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, '
                'duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly '
                'drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned '
                'face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, '
                'fused fingers, too many fingers, long neck, username, watermark, signature')

            # Create a dictionary template for the prompt
            prompt_dict = {
                "prompt_number": index,
                "prompt_text": prompt_text,
                'negative_prompt': negative_prompt,
                'Characters_involved': prompt['Characters_involved'],
                'context_prompt': prompt['context_prompt'],
                'generated_photo_path': {
                    "textual_inversion": None,
                    "dreambooth": None,
                    "custom_diffusion": None,
                    "cones_2": None,
                    "mix_of_show": None,
                    "LORA": None,
                    "Lambda_ECLIPSE": None,
                },
                'scores': {key: {'PickScore': 0, 'HPS': 0, 'TIFA_metric_score': 0} for key in
                           ["textual_inversion", "dreambooth", "custom_diffusion", "cones_2", "mix_of_show", "LORA",
                            "Lambda_ECLIPSE"]}
            }

            # only when the prompt contain a character
            if prompt['Characters_involved']:
                # transfer the character class to a dictionary and attach all character info to the prompt in JSON file
                list_of_characters_dict = [
                    char.to_dict() for char_ in prompt['Characters_involved']
                    for char in characters
                    if char_.split('_')[0].lower() == char.unique_token.lower() or char_.split(' ')[
                        0].lower() == char.unique_token.lower()
                ]

                if list_of_characters_dict:
                    prompt_dict["Characters_involved"] = list_of_characters_dict

                    # Filter the characters to ensure only those cited in the prompt text are included
                    prompt_dict['Characters_involved'] = [
                        char_ for char_ in list_of_characters_dict
                        if
                        char_['unique_token'].lower() in prompt['prompt_text'].lower() or char_['name'].lower() in
                        prompt[
                            'prompt_text'].lower()
                    ]

                    if prompt_dict['Characters_involved']:

                        # add one DINO embedding score per character
                        DINO_dic = {f'ViTS_16_DINO_embeddings_{x + 1}': 0 for x in
                                    range(len(prompt_dict['Characters_involved']))}

                        # add one adaface score per character
                        adaface_dic = {f'adaface_{x + 1}': 0 for x in range(len(prompt_dict['Characters_involved']))}

                        # add one FaceNet score per character
                        FaceNet_dic = {f'FaceNet_{x + 1}': 0 for x in range(len(prompt_dict['Characters_involved']))}

                        # add one inception v3 score per character
                        inception_v3_dic = {f'inception_v3_{x + 1}': 0 for x in range(len(prompt_dict['Characters_involved']))}

                        for key in prompt_dict['scores'].keys():
                            prompt_dict['scores'][key].update(DINO_dic)
                            prompt_dict['scores'][key].update(adaface_dic)
                            prompt_dict['scores'][key].update(FaceNet_dic)
                            prompt_dict['scores'][key].update(inception_v3_dic)

                        # collect the characters boundaries box coordinates
                        list_of_characters_boundaries = []
                        for i in range(len(prompt_dict['Characters_involved'])):
                            region_key = f'region_{i + 1}_boundaries'
                            prompt_dict[f'region_{i + 1}_prompt'] = prompt[f'region_{i + 1}_prompt']

                            # depends on the various formate on the LLM output, parsing of information is different
                            try:
                                boundaries = eval(prompt[region_key]) if GPT_4 else prompt[region_key]
                                if isinstance(boundaries, list) and len(boundaries) == 3:
                                    x, y, z = boundaries
                                    prompt_dict[region_key] = [x[0], x[1], y, z]
                                else:
                                    prompt_dict[region_key] = boundaries
                            except:
                                prompt_dict[region_key] = prompt[region_key] if isinstance(prompt[region_key],
                                                                                           list) else list(
                                    eval(prompt[region_key]).values())

                            list_of_characters_boundaries.append(prompt_dict[region_key])

                        # generate characters layout in 768*768 image to be used for Cone 2 method
                        prompt_dict['prompt_layout_path'] = generate_prompt_layout(list_of_characters_boundaries, index,
                                                                                   tag, GPT_4)
            # Append the prompt dictionary to the template list
            template.append(prompt_dict)

        # Convert the template to JSON format
        with open(os.path.join(os.getcwd(), f"prompt_{LLM}_{tag}.json"), "w") as f:
            json.dump(template, f, indent=4)


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

    # write a story using GPT_4 for two characters
    story_instance = STORYTELLING(characters=[character_1, character_2], GPT_4=True)
    story = story_instance.generate_story()
    story_instance.chunk_story(story)

    # write a story using Llama 3 for two characters
    story_instance = STORYTELLING(characters=[character_1, character_2], GPT_4=None, active_llm=1)
    _, story = story_instance.generate_story()
    story_instance.chunk_story(story)
