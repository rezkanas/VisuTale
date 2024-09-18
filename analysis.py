from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os
import json
import matplotlib.pyplot as plt


def Image_Scores_Plot(df):
    # Define the metrics and their labels
    metrics = ['PickScore', 'HPS', 'TIFA_metric_score', 'ViTS_16_DINO_embeddings_average', 'adaface_average',
               'FaceNet_average', 'inception_v3_average']
    metric_labels = ['Pick Score', 'HPS', 'TIFA Metric Score', 'ViTS 16 DINO Embeddings Average', 'Adaface Average',
                     'FaceNet Average', 'Inception v3 average']
    markers = ['o', 's', '^', 'D', 'P', '*', 'X']  # Marker styles for each metric
    colors = plt.cm.tab20(np.linspace(0, 1, len(metrics)))  # Generate evenly spaced colors
    Models = df['Model'].unique()

    fig, axs = plt.subplots(3, 2, figsize=(14, 18), facecolor='white')

    for number_of_character, ax_row in zip(['prompt_include_two_character', 'prompt_include_one_character',
                                            'prompt_include_no_character'], axs):
        for model, ax in zip(Models, ax_row):

            data_frame = df[(df[number_of_character] == True) & (df['Model'] == model)][metrics + ['Method']]
            methods = df['Method'].unique()

            for method in methods:
                subset = data_frame[data_frame['Method'] == method].drop(columns=['Method'])
                for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
                    x = subset[metric]
                    y = [method] * len(x)  # Create a list of the method name with the same length as y
                    ax.scatter(x, y, label=label if method == methods[
                        0] and model == 'GPT_4' and number_of_character == 'prompt_include_two_character' else "",
                               alpha=0.7, marker=markers[j], color=colors[j])

            ax.set_title(
                f'Score for images generated\nusing {model} {number_of_character.replace("_", " ").replace("prompt", "prompts").replace("include", "including")}',
                fontsize=16)
            ax.set_xlabel('Metric Values', fontsize=14)
            ax.set_ylabel('Method', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)

    # Create a single legend for all graphs outside the subplots
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), shadow=True, ncol=len(metrics), fontsize=12)

    # Create a folder to store analysis result
    if not os.path.exists(os.path.join(os.getcwd(), 'analysis')):
        os.makedirs(os.path.join(os.getcwd(), 'analysis'))

    plt.tight_layout()

    output_path = os.path.join(os.getcwd(), 'analysis', 'LLM Image Scores.png')
    plt.savefig(output_path, bbox_inches='tight')


def metric_score_per_story(df):
    # Define the metrics and their labels
    metrics = ['ViTS_16_DINO_embeddings_average', 'HPS', 'TIFA_metric_score', 'PickScore', 'adaface_average',
               'FaceNet_average', 'inception_v3_average']
    metric_labels = ['ViTS16 DINO Embeddings Average', 'HPS', 'TIFA Metric Score', 'Pick Score', 'Adaface Average',
                     'FaceNet Average', 'Inception v3 average']
    data_frame = df[metrics + ['Method', 'Model']]

    # Melt the DataFrame to long format for easier plotting with boxplot
    melted_df = data_frame.melt(id_vars=['Method', 'Model'], value_vars=metrics, var_name='Metric', value_name='Score')

    # Replace metric names with their corresponding labels
    melted_df['Metric'] = melted_df['Metric'].replace(dict(zip(metrics, metric_labels)))

    # Plot the boxplot for all scores of each metric
    plt.figure(figsize=(14, 8), facecolor='white')
    ax = sns.boxplot(x='Metric', y='Score', data=melted_df, hue='Method')

    plt.title('Scores for Each Metric by Method', fontsize=18)
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)

    # Positioning the legend
    plt.legend(loc='upper right', title='Method')
    plt.tight_layout()
    output_path = os.path.join(os.getcwd(), 'analysis', f'Scores for Each Metric by Story.jpg')
    plt.savefig(output_path)

    plt.show()


def correlation_coefficients_and_p_value(df):
    # Calculate correlation coefficients and p-values

    # Define metrics and columns
    story_metrics = ['overall_style_consistency', 'average_entity_consistency_score']
    image_score_metrics = ['PickScore', 'HPS', 'TIFA_metric_score', 'ViTS_16_DINO_embeddings_average',
                           'adaface_average',
                           'FaceNet_average', 'inception_v3_average']
    models = df['Model'].unique()

    # Initialize overall correlation and p-value DataFrames
    overall_corr_df = pd.DataFrame(index=story_metrics, columns=image_score_metrics)
    overall_p_value_df = pd.DataFrame(index=story_metrics, columns=image_score_metrics)

    path = os.path.join(os.getcwd(), 'analysis')
    file_path = os.path.join(path, "correlation_analysis.txt")
    # Open a file for writing
    with open(file_path, "w") as file:
        dataframe = df.groupby('Method')[story_metrics + image_score_metrics].mean()

        # Initialize correlation and p-value DataFrames for each model
        corr_df = pd.DataFrame(index=story_metrics, columns=image_score_metrics)
        p_value_df = pd.DataFrame(index=story_metrics, columns=image_score_metrics)

        file.write(f'Pearson correlation between:\n')
        for story_metric in story_metrics:
            for image_score_metric in image_score_metrics:
                score_metric = dataframe[image_score_metric].fillna(0).to_numpy()
                story_metric_value = dataframe[story_metric].fillna(0).to_numpy()

                # Ensure non-constant arrays before computing correlation
                if np.unique(score_metric).size > 1 and np.unique(story_metric_value).size > 1:
                    corr, p_value = pearsonr(score_metric, story_metric_value)
                else:
                    corr, p_value = np.nan, np.nan

                if p_value < 0.05:
                    file.write(
                        f' - {story_metric} and {image_score_metric} is statistically significant (alpha is 0.05).\n')
                elif 0.05 <= p_value < 0.1:
                    file.write(
                        f' - {story_metric} and {image_score_metric} is statistically significant (alpha is 0.1).\n')
                else:
                    file.write(f' - {story_metric} and {image_score_metric} is not statistically significant.\n')
                corr_df.at[story_metric, image_score_metric] = corr
                p_value_df.at[story_metric, image_score_metric] = p_value

        # Concatenate correlation and p-value DataFrames for each model with overall DataFrames
        overall_corr_df = pd.concat([overall_corr_df, corr_df], axis=0)
        overall_p_value_df = pd.concat([overall_p_value_df, p_value_df], axis=0)

        file.write(f'Pearson correlation coefficient\n{corr_df}\n\n')
        file.write(f'P value table\n{p_value_df}\n\n\n')

    # Save overall corr_df and overall_p_value_df as CSV files
    overall_corr_df.to_csv(os.path.join(path, "overall_corr.csv"))
    overall_p_value_df.to_csv(os.path.join(path, "overall_p_value.csv"))


def scatter_plot_of_story_evaluation_metrics(df):
    # Define the metrics and their labels
    story_metrics = ['overall_style_consistency', 'average_entity_consistency_score', 'average_image_score']
    models = df['Model'].unique()
    methods = df['Method'].unique()

    # Define a list of more distinctive marker shapes for each method
    method_markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>', '8', 'p', 'h']

    # Ensure there are enough markers for the number of methods
    assert len(method_markers) >= len(methods), "Not enough markers for the number of methods"

    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))  # Generate evenly spaced colors for models

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set background to white
    fig.patch.set_facecolor('white')

    # Plot data points
    for model_idx, (model, color) in enumerate(zip(models, colors)):
        dataframe = df[df['Model'] == model].groupby('Method')[story_metrics].mean()

        for method_idx, method in enumerate(methods):
            if method in dataframe.index:
                subset = dataframe.loc[method]
                x = subset['overall_style_consistency']
                y = subset['average_entity_consistency_score']
                z = subset['average_image_score']
                marker = method_markers[method_idx]  # Get marker by index
                ax.scatter(x, y, z, label=model if method == methods[0] else "", alpha=0.7, color=color, marker=marker,
                           s=100, linewidths=2)

    # Set axis labels
    ax.set_xlabel('Overall Style Consistency', fontsize=12, labelpad=10, color='black')
    ax.set_ylabel('Average Entity Consistency Score', fontsize=12, labelpad=10, color='black')
    ax.set_zlabel('Average Image Score', fontsize=12, labelpad=10, color='black')
    ax.set_title('Scatter Plot of Story Evaluation Metrics', fontsize=16, color='black')

    # Set the axis colors to black
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis='z', colors='black')

    # Set the grid lines to black
    ax.xaxis._axinfo["grid"].update(color='black', linestyle='-')
    ax.yaxis._axinfo["grid"].update(color='black', linestyle='-')
    ax.zaxis._axinfo["grid"].update(color='black', linestyle='-')

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Create a custom legend
    legend_elements = []
    for method_idx, method in enumerate(methods):
        marker = method_markers[method_idx]
        legend_elements.append(
            plt.Line2D([0], [0], marker=marker, color='w', label=method, markerfacecolor='k', markersize=10))

    # Add models to the legend elements
    for model_idx, model in enumerate(models):
        color = colors[model_idx]
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label=model, markerfacecolor=color, markersize=10))

    # Positioning the legend
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 1), shadow=True, framealpha=0.5)

    output_path = os.path.join(os.getcwd(), 'analysis', f'Scatter Plot of Story Evaluation Metrics.jpg')
    plt.savefig(output_path, bbox_inches='tight')

    # Rotate the view
    ax.view_init(elev=10, azim=-40)  # Adjust elevation and azimuth

    plt.tight_layout()
    plt.show()


def build_story_grid(image_paths, output_path, labels, header_texts, rows=13, columns=11):
    # Load all images
    images = [Image.open(path) for path in image_paths]

    # Assuming all images are of the same size
    img_width, img_height = images[0].size

    # Define the font and size for labels and headers
    try:
        bold_font = ImageFont.truetype("arialbd.ttf", 24)  # Bold font with increased size
        header_font = ImageFont.truetype("arial.ttf", 20)  # Regular font for header texts
    except IOError:
        bold_font = ImageFont.load_default()
        header_font = ImageFont.load_default()

    # Calculate the space for labels
    label_width = 100  # Decreased label width
    header_height = 70

    # Create a new blank image with appropriate size
    grid_width = columns * img_width + label_width
    grid_height = rows * img_height + header_height
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')

    # Create a drawing object
    draw = ImageDraw.Draw(grid_img)

    # Wrap and draw the header texts
    x_offset = label_width
    y_offset = 0
    wrap_width = int(img_width / 6)  # Increased wrap width
    for i, header in enumerate(header_texts):
        wrapped_text = textwrap.fill(header, width=wrap_width)  # Wrap text to fit within the image width
        draw.text((x_offset, y_offset), wrapped_text, fill="black", font=header_font)
        x_offset += img_width

    # Draw the images and labels
    y_offset = header_height
    next_image_index = 0
    for row in range(rows):
        # Draw the label for the row
        label = labels[row]
        bbox = draw.textbbox((0, 0), label, font=bold_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        label_x = (label_width - text_width) / 2
        draw.text((label_x, y_offset + (img_height - text_height) // 2), label, fill="black", font=bold_font)

        # Draw the images in the row
        x_offset = label_width
        for col in range(columns):
            if next_image_index < len(images):
                img = images[next_image_index]
                grid_img.paste(img, (x_offset, y_offset))
                next_image_index += 1
            x_offset += img_width
        y_offset += img_height

    # Save the output image
    grid_img.save(output_path)


def main():
    GPT_4 = True
    for _ in range(2):
        # load stories
        LLM = 'GPT_4' if GPT_4 else 'Llama_3'
        tag = 'two_characters'
        JSON_name = f'GPT_4_score_{tag}.json'
        json_file_name = os.path.join(os.getcwd(), JSON_name)

        with open(json_file_name, "r") as file:
            data = json.load(file)

        # collect all images generated
        stories = list(data[LLM].keys())
        for i in range(2):
            image_paths = []
            if i == 0:
                stories_ = stories[:7]
            else:
                stories_ = stories[7:]
            for method in stories_:
                for path in data[LLM][method]['image_path']:
                    # collect image paths
                    image_paths.append(path)

            columns = len(data[LLM][method]['image_path'])

            # create a folder to store analysis result
            if not os.path.exists(os.path.join(os.getcwd(), 'analysis')):
                os.makedirs(os.path.join(os.getcwd(), 'analysis'))

            # save image to...
            output_path = os.path.join(os.getcwd(), 'analysis', f'all_images_{LLM}_{tag}_{i + 1}.jpg')

            # collect prompts from json file
            prompt_text = []
            JSON_name = f'prompt_{LLM}_{tag}_filled.json'
            json_file_name = os.path.join(os.getcwd(), JSON_name)
            with open(json_file_name, "r") as file:
                data_2 = json.load(file)
            for prompt_dict in data_2:
                for char in prompt_dict['Characters_involved']:
                    # modify prompt to remove unique tokens
                    prompt_dict['prompt_text'] = prompt_dict['prompt_text'].replace(
                        f"{char['unique_token']} man" if char['gender'] == 'M' else f"{char['unique_token']} woman",
                        char['name'])
                    prompt_dict['prompt_text'] = prompt_dict['prompt_text'].replace(
                        f"{char['unique_token']} person" if char['gender'] == 'M' else f"{char['unique_token']} person",
                        char['name'])
                    prompt_dict['prompt_text'] = prompt_dict['prompt_text'].replace(
                        f"{char['unique_token']}" if char['gender'] == 'M' else f"{char['unique_token']}", char['name'])
                prompt_text.append(prompt_dict['prompt_text'])

            labels = [story.replace('_', ' ') for story in stories_]

            # make a story grid
            build_story_grid(image_paths, output_path,
                             labels=labels, header_texts=prompt_text,
                             rows=len(stories_), columns=columns)
        GPT_4 = False

    tag = 'two_characters'
    GPT4_score_path = os.path.join(os.getcwd(), f"GPT_4_score_{tag}.json")
    with open(GPT4_score_path, "r") as file:
        story_figures = json.load(file)

    # Flatten the dictionary structure
    flattened_data = []

    for model, methods in story_figures.items():
        for method, details in methods.items():
            scores = details['scores']
            style_consistency_score = details.get('style_consistency_score', {})
            entity_consistency_score = details.get('entity_consistency_score', {})

            for idx, score in enumerate(scores):
                row = {
                    'Model': model,
                    'Method': method,
                    'Index': idx + 1,
                    **score,
                    **style_consistency_score,
                    **entity_consistency_score,
                    'prompt_include_one_character': details['prompt_include_one_character'][idx],
                    'prompt_include_two_character': details['prompt_include_two_character'][idx],
                    'prompt_include_no_character': details['prompt_include_no_character'][idx],
                    'compilation_rank': details.get('rank', 0)
                }
                flattened_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(flattened_data)

    # Combine ViTS,FaceNet, inception_v3 and adaface metrics
    df['ViTS_16_DINO_embeddings_average'] = df[['ViTS_16_DINO_embeddings_1', 'ViTS_16_DINO_embeddings_2']].fillna(
        0).mean(axis=1)
    df['adaface_average'] = df[['adaface_1', 'adaface_2']].fillna(0).mean(axis=1)
    df['FaceNet_average'] = df[['FaceNet_1', 'FaceNet_2']].fillna(0).mean(axis=1)
    df['inception_v3_average'] = df[['inception_v3_1', 'inception_v3_2']].fillna(0).mean(axis=1)

    # plot all the analysis and store them locally
    Image_Scores_Plot(df)
    metric_score_per_story(df)
    scatter_plot_of_story_evaluation_metrics(df)
    correlation_coefficients_and_p_value(df)
