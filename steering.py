import os
from functools import partial
from typing import Optional

import fire
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens import SAE, ActivationsStore, HookedSAETransformer

torch.set_grad_enabled(False)


def get_dashboard_html(
    sae_release: str = "llama-3-8b-it-res-jh",
    sae_id: str = "blocks.25.hook_resid_post",
    feature_idx: int = 0,
):
    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    return html_template.format(sae_release, sae_id, feature_idx)


def find_max_activation(
    model: HookedSAETransformer,
    sae: SAE,
    activation_store: ActivationsStore,
    feature_idx: int,
    num_batches: int = 100,
):
    """
    Find the maximum activation for a given feature index. This is useful for
    calibrating the right amount of the feature to add.
    """
    max_activation = 0.0

    pbar = tqdm(range(num_batches))
    for _ in pbar:
        tokens = activation_store.get_batch_tokens()

        _, cache = model.run_with_cache(
            tokens,
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[sae.cfg.hook_name],
        )
        sae_in = cache[sae.cfg.hook_name]
        feature_acts = sae.encode(sae_in).squeeze()

        feature_acts = feature_acts.flatten(0, 1)
        batch_max_activation = feature_acts[:, feature_idx].max().item()
        max_activation = max(max_activation, batch_max_activation)

        pbar.set_description(f"Max activation: {max_activation:.4f}")

    return max_activation


def steering(
    activations: torch.Tensor,
    hook: str,
    steering_strength: float = 1.0,
    steering_vector: Optional[torch.Tensor] = None,
    max_act: float = 1.0,
) -> torch.Tensor:
    if steering_vector is None:
        return activations
    return activations + max_act * steering_strength * steering_vector


def generate_with_steering(
    model: HookedSAETransformer,
    sae: SAE,
    prompt: str,
    steering_feature: int,
    max_act: float,
    steering_strength: float = 1.0,
    max_new_tokens: int = 95,
):
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)

    steering_vector = sae.W_dec[steering_feature].to(model.cfg.device)

    steering_hook = partial(
        steering,
        max_act=max_act,
        steering_strength=steering_strength,
        steering_vector=steering_vector,
    )

    # standard transformerlens syntax for a hook context for generation
    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos=False,
            prepend_bos=sae.cfg.prepend_bos,
        )

    return model.tokenizer.decode(output[0])


def main(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    release_name: str = "llama-3-8b-it-res-jh",
    sae_id: str = "blocks.25.hook_resid_post",
    steering_strengths: str = "-4.0,-2.0,-0.5,0.5,2.0,4.0",
    prompt: str = "Let x = 1. What is x << 3 in Python 3?",
    max_new_tokens: int = 256,
    output_path: str = "./steering_output/mmlu_cs.html",
):
    device: str = "cuda"
    model = HookedSAETransformer.from_pretrained(model_name)

    # the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
    # Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
    # We also return the feature sparsities which are stored in HF for convenience.
    sae, _cfg_dict, _sparsity = SAE.from_pretrained(
        release=release_name,
        sae_id=sae_id,  # <- SAE id (not always a hook point!)
        device=device,
    )

    _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])

    # let's print the top 5 features and how much they fired
    vals, inds = torch.topk(cache["blocks.25.hook_resid_mid"][0, -1, :], 5)
    for val, ind in zip(vals, inds):
        print(f"Feature {ind} fired {val:.2f}")

    # save the top 1 feature index for later as an int
    steering_feature = int(inds[0].item())
    print(f"Steering feature: {steering_feature}")

    # a convenient way to instantiate an activation store is to use the from_sae method
    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        # fairly conservative parameters here so can use same for larger
        # models without running out of memory.
        store_batch_size_prompts=8,
        train_batch_size_tokens=4096,
        n_batches_in_buffer=32,
        device=device,
    )

    # Find the maximum activation for this feature
    max_act = find_max_activation(model, sae, activation_store, steering_feature)

    print(f"Maximum activation for feature {steering_feature}: {max_act:.4f}")

    # note we could also get the max activation from Neuronpedia (https://www.neuronpedia.org/api-doc#tag/lookup/GET/api/feature/{modelId}/{layer}/{index})

    # Generate text without steering for comparison
    normal_text = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        stop_at_eos=False,
        prepend_bos=sae.cfg.prepend_bos,
    )

    # Experiment with different steering strengths
    # Convert string of strengths to list of floats
    try:
        strengths = [float(s) for s in steering_strengths.split(",")]
    except ValueError:
        raise ValueError("steering_strengths must be a comma-separated list of numbers")

    if not strengths:
        raise ValueError("steering_strengths cannot be empty")

    # Create a DataFrame to store and compare responses
    responses = {
        "Type": ["Original Prompt", "Normal Response"]
        + [f"Steering {strength}" for strength in strengths],
        "Text": [normal_text],
    }

    # Collect steered responses
    for strength in strengths:
        steered_text = generate_with_steering(
            model,
            sae,
            prompt,
            steering_feature,
            max_act,
            steering_strength=strength,
            max_new_tokens=max_new_tokens,
        )
        responses["Text"].append(steered_text)

    # Create DataFrame
    df = pd.DataFrame(responses)

    # Create an HTML file with formatted output
    html_output = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                max-width: 1200px;
                margin: auto;
            }}
            .response-container {{
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .response-type {{
                font-weight: bold;
                color: #2c5282;
                margin-bottom: 10px;
            }}
            .response-text {{
                white-space: pre-wrap;
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
            }}
            .feature-info {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f0f4f8;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <h2>Feature Information</h2>
        <div class="feature-info">
            <p>Feature {steering_feature} maximum activation: {max_act:.4f}</p>
            <h3>Top 5 Features:</h3>
            {''.join(f'<p>Feature {ind.item()}: {val.item():.2f}</p>' for val, ind in zip(vals, inds))}
        </div>
        
        <h2>Responses</h2>
        {''.join(f'''
        <div class="response-container">
            <div class="response-type">{row['Type']}</div>
            <div class="response-text">{row['Text']}</div>
        </div>
        ''' for _, row in df.iterrows())}
    </body>
    </html>
    """

    # Save HTML file
    if not output_path.endswith(".html"):
        output_path += ".html"
    with open(output_path, "w") as f:
        f.write(html_output)

    print(f"\nResponses have been saved to {output_path}")

    # Visualize feature activations
    plt.figure(figsize=(10, 6))
    plt.bar(range(5), vals.cpu().numpy())
    plt.title("Top 5 Feature Activations")
    plt.xlabel("Feature Index")
    plt.ylabel("Activation Value")
    tick_labels = [str(idx.item()) for idx in inds.cpu()]
    plt.xticks(range(5), tick_labels)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
