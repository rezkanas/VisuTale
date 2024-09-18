# VisuTale
Immersing oneself in a personally tailored visual story, complete with individualized details like names, character descriptions, and personal images, promises a truly captivating experience—an ideal gift for any avid comic enthusiast in our lives! A personalized visual story, specifically designed for a targeted individual or audience, differing from traditional visual story with fixed characters and storylines. Instead, it is crafted to incorporate personal details and preferences of the intended recipient. This unique customization encompasses elements like the person's nickname,  appearance, and personality traits, aiming to create a truly one-of-a-kind, engaging visual story that deeply resonates with the individual it's crafted for. 
Firstly, this project aims to autonomously generate personalized visual stories based on provided character images and traits. The second research objective is to evaluate whether combining images from multiple methods can yield more consistent narratives. The project successfully generates personalized stories using various techniques and demonstrates that compiling new stories from different 
methods, based on the weighted average score of their images—including Pick score and HPS—results in better story style and character consistency compared to those generated by individual methods Additionally, GPT-4 generate more consistent visual story compared to Llama 3.


# Setting Up This Project Environment

1. Create a virtual environment in your desired directory:
   ```bash
   python3 -m venv myenv
   ```

2. Activate the virtual environment:
   ```bash
   source myenv/bin/activate
   ```

3. Clone the required GitHub repositories to the same directory as 'myenv':
   ```bash
   git clone https://github.com/rezkanas/AdaFace.git
   git clone https://github.com/rezkanas/tifa.git
   git clone --branch v2.4.1 --single-branch https://github.com/timesler/facenet-pytorch.git facenet_pytorch
   ```
4. Install dependencies in two steps:

step 1 
   ```bash
   pip install tifascore GitPython facenet-pytorch===2.4.1 Faker bs4 accelerate python-dotenv ultralytics diffusers==0.21.1 transformers==4.39.3 hpsv2 openai wget hugchat torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 xformers==0.0.20 peft==0.11.0 PyYAML datasets==2.18.0 torchmetrics==0.11.4 hydra-core==1.0.7 omegaconf==2.0.6 antlr4-python3-runtime==4.8 
   ```
step 2
   ```bash
   pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 
   pip install opencv-python==4.7.0.72
   ```
5. Create a .env file including the following: 
OPENAI_API_KEY= <key>
huggingchat_user_email = <email>
huggingchat_user_pwd = <password>

6. Place the following files in the same directory as 'myenv':
   - `character.py`
   - `evaluate_story.py`
   - `main.py`
   - `utilities.py`
   - `textual_story.py`
   - `visual_story.py`
   - `analysis.py`

7. Create 'photos' folder then inside create a folder for each character images.

8. For training:
    - Open `main.py` and ensure `Train = True`.
    - Run `main.py`.
if you encounter a problem during training, be mindful of note 2 below. 

9. To access the generated images for each method, navigate to its 'results' folder.

10. After the file has finished running, review the top 3 stories for each LLM as evaluated by the final GPT-4. Navigate to the '3_best_two_characters_stories' directory. The results of the evaluation are recorded in `GPT_4_score_two_characters.json`.

11. To access the relevant analysis figures, enter the 'analysis' folder.

### Complete Folder Structure at the End of Run

project_folder
├── textual_inversion
│   ├── huggingface_diffusers
│   ├── venv
│   ├── 3000
│   └── results
├── LORA
│   ├── huggingface_peft
│   ├── venv
│   ├── 1000
│   └── results
├── dreambooth
│   ├── ShivamShrirao_diffusers
│   ├── venv
│   ├── 1e-06
│   └── results
├── custom_diffusion
│   ├── huggingface_diffusers
│   ├── venv
│   ├── 2300
│   └── results
├── cones_2
│   ├── cones_2_git_repository
│   ├── venv
│   ├── nsnn
│   ├── znrz
│   └── results
├── mix_of_show
│   ├── mix_of_show_git_repository
│   ├── venv
│   ├── 500
│   └── results
├── analysis
│   ├── 
│   └──
├── prompt_layout
│   ├── 
│   └──
├── facenet_pytorch
│   ├── 
│   └──
├── AdaFace
│   ├── 
│   └──
├── tifa
│   ├── 
│   └──
├── story
│   ├── two_characters
│   │   ├── LlaMa_3 
│   │   └── GPT_4
│   ├── effective_prompts.txt
│   └── negative_prompt_examples.txt
├── myenv
├── 3_Best_stories_two_characters
├── photos
│   ├── Basel
│   └── Rizeh
├── pretrained
│   └── adaface_ir50_ms1mv2.ckpt
├── character.py
├── evaluate_story.py
├── main.py
├── utilities.py
├── textual_story.py
├── visual_story.py
├── analysis.py 
└── .env


NOTES: 

1 - It's not uncommon for https://huggingface.co/chat/ to experience periodic downtime. However, these interruptions are usually brief. If you encounter the following error message repeatedly for more than 4 times in a row:

```
Caught an exception: Something went wrong!
huggingchat sent an incomplete response
Re-querying huggingchat API...
```
then you need to stop the code and check the link to ensure that the website is operational before running the code again. 
 
2 - If you need to interrupt the code for any reason while it's setting up the environment for one of the personalization methods, ensure that you remove its GitHub repository folder inside the method folder before re-running the code.

