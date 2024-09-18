from faker import Faker
import requests
import random
from bs4 import BeautifulSoup
import os
import shutil


class CHARACTER:
    """a class defining each character in the story."""
    def __init__(self, photo_folder: str, gender: str, name: str = None, traits=None, unique_token: str = None, random_photo: str = None):

        # allocate name to character
        if name is None or not isinstance(name, str):
            self.name = self.generate_random_name(gender)
        else:
            self.name = name

        # allocate traits to character in case none were provided
        if traits is None or not isinstance(traits, dict):
            self.traits = self.generate_random_traits()
        else:
            self.traits = traits

        # allocate gender and photo path to character
        self.gender = gender
        self.photo_folder = photo_folder

        # select random photo path of character
        if random_photo is None or not os.path.exists(random_photo):
            self.random_photo = random.choice([os.path.join(photo_folder, file) for file in os.listdir(photo_folder)])
        else:
            self.random_photo = random_photo

        # select unique token to character
        if unique_token is None:
            self.unique_token = self.generate_unique_token(self.name)
        else:
            self.unique_token = unique_token

        # create a mini photo folder for methods that use only 4-5 images
        self.mini_photo_folder = self.select_and_copy_images(photo_folder)

    @staticmethod
    def generate_unique_token(name):
        # in case no unique token is provided, this function will generate one based on character name
        vowels = "aeiouAEIOU"
        filtered_string = ''.join([char.lower() for char in name if char not in vowels and char != ' '])
        token_length = min(len(filtered_string), 5)

        if token_length < 4:
            return ''.join(random.choices(filtered_string, k=4))

        return ''.join(random.sample(filtered_string, token_length))

    @staticmethod
    def select_and_copy_images(original_folder_path):
        # Get the title of the original images folder
        folder_title = os.path.basename(original_folder_path)
        # Create the new folder name for the copied images
        new_folder_name = f"{folder_title}_mini"
        # Construct the path for the new folder
        new_folder_path = os.path.join(os.path.dirname(original_folder_path), new_folder_name)
        # Create the new folder
        os.makedirs(new_folder_path, exist_ok=True)
        if len(os.listdir(new_folder_path)) == 5:
            return new_folder_path

        # List all files in the original folder
        image_files = [f for f in os.listdir(original_folder_path) if
                       os.path.isfile(os.path.join(original_folder_path, f))]
        # Select 5 random images
        selected_images = random.sample(image_files, min(5, len(image_files)))

        # Copy selected images to the new folder
        for image_file in selected_images:
            original_image_path = os.path.join(original_folder_path, image_file)
            new_image_path = os.path.join(new_folder_path, image_file)
            shutil.copy2(original_image_path, new_image_path)
        return new_folder_path

    @staticmethod
    def generate_random_name(gender):
        """generate random full name in case user do not provide a name for the character."""
        fake = Faker()
        return fake.simple_profile(sex=gender)['name']

    @staticmethod
    def generate_random_traits():
        """generate random positive, negative and neutral traits in case user do not provide a name for the
        characters."""

        # URL of the website to scrape
        url = "https://ideonomy.mit.edu/essays/traits.html"

        # Send an HTTP GET request to the website
        response = requests.get(url)

        # Parse the HTML code using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the relevant information from the HTML code
        # Find all <ol> elements
        ols = soup.find_all('ol')

        # build a dictionary to fill up all traits
        traits_type = ['positive traits', 'neutral traits', 'negative traits']
        all_traits = {key: [] for key in traits_type}

        # Iterate over each <ol> element
        for ol, trait_type in zip(ols, traits_type):

            # Find all <li> elements inside the current <ol> element
            li_elements = ol.find_all('li')

            # Append each <li> element to the corresponding list in the dictionary
            for li in li_elements:
                all_traits[trait_type].append(li.get_text()[1::])

        return {key: random.sample(all_traits[key], 2) for key in all_traits.keys()}

    def to_dict(self):
        return {
            'name': self.name,
            'traits': self.traits,
            'photo_folder': self.photo_folder,
            'random_photo': self.random_photo,
            'unique_token': self.unique_token,
            'gender': self.gender,
            'mini_photo_folder': self.mini_photo_folder,
        }


if __name__ == "__main__":
    photos_folder_1 = os.path.join(os.getcwd(), 'photos', 'Rizeh')

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
    print('Character name is', character_1.name,
          '\nCharacter traits are:', character_1.traits,
          '\nCharacter unique token:', character_1.unique_token)
