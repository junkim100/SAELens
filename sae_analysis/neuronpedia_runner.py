import os
from typing import Any, Optional, cast

# set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time

import torch
from sae_vis.data_fetching_fns import get_feature_data
from sae_vis.data_storing_fns import FeatureVisParams, to_str_tokens
from tqdm import tqdm

import numpy as np
from sae_training.utils import LMSparseAutoencoderSessionloader
import json

from matplotlib import colors

BG_COLOR_MAP = colors.LinearSegmentedColormap.from_list(
    "bg_color_map", ["white", "darkorange"]
)


class NpEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)


class NeuronpediaRunner:

    def __init__(
        self,
        sae_path: str,
        feature_sparsity_path: Optional[str] = None,
        neuronpedia_parent_folder: str = "./neuronpedia_outputs",
        init_session: bool = True,
        # token pars
        n_batches_to_sample_from: int = 2**12,
        n_prompts_to_select: int = 4096 * 6,
        # sampling pars
        n_features_at_a_time: int = 1024,
        buffer_tokens_left: int = 8,
        buffer_tokens_right: int = 8,
    ):
        self.sae_path = sae_path
        if init_session:
            self.init_sae_session()

        self.feature_sparsity_path = feature_sparsity_path
        self.n_features_at_a_time = n_features_at_a_time
        self.buffer_tokens_left = buffer_tokens_left
        self.buffer_tokens_right = buffer_tokens_right
        self.n_batches_to_sample_from = n_batches_to_sample_from
        self.n_prompts_to_select = n_prompts_to_select

        # Deal with file structure
        if not os.path.exists(neuronpedia_parent_folder):
            os.makedirs(neuronpedia_parent_folder)
        self.neuronpedia_folder = (
            f"{neuronpedia_parent_folder}/{self.get_folder_name()}"
        )
        if not os.path.exists(self.neuronpedia_folder):
            os.makedirs(self.neuronpedia_folder)

    def get_folder_name(self):
        model = self.sparse_autoencoder.cfg.model_name
        hook_point = self.sparse_autoencoder.cfg.hook_point
        d_sae = self.sparse_autoencoder.cfg.d_sae
        dashboard_folder_name = f"{model}_{hook_point}_{d_sae}"

        return dashboard_folder_name

    def init_sae_session(self):
        (
            self.model,
            sae_group,
            self.activation_store,
        ) = LMSparseAutoencoderSessionloader.load_session_from_pretrained(self.sae_path)
        # TODO: handle multiple autoencoders
        self.sparse_autoencoder = sae_group.autoencoders[0]

    def get_tokens(
        self, n_batches_to_sample_from: int = 2**12, n_prompts_to_select: int = 4096 * 6
    ):
        all_tokens_list = []
        pbar = tqdm(range(n_batches_to_sample_from))
        for _ in pbar:
            batch_tokens = self.activation_store.get_batch_tokens()
            batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
                : batch_tokens.shape[0]
            ]
            all_tokens_list.append(batch_tokens)

        all_tokens = torch.cat(all_tokens_list, dim=0)
        all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
        return all_tokens[:n_prompts_to_select]

    def round_list(self, to_round: list[float]):
        return list(np.round(to_round, 3))

    def run(self):
        """
        Generate the Neuronpedia outputs.
        """

        if self.model is None:
            self.init_sae_session()

        self.n_features = self.sparse_autoencoder.cfg.d_sae
        assert self.n_features is not None

        # if we have feature sparsity, then use it to only generate outputs for non-dead features
        self.target_feature_indexes: list[int] = []
        if self.feature_sparsity_path:
            loaded = torch.load(
                self.feature_sparsity_path, map_location=self.sparse_autoencoder.device
            )
            self.target_feature_indexes = (
                (loaded > -5).nonzero(as_tuple=True)[0].tolist()
            )
        else:
            self.target_feature_indexes = list(range(self.n_features))
            print("No feat sparsity path specified - doing all indexes.")

        # divide into batches
        feature_idx = torch.tensor(self.target_feature_indexes)
        n_subarrays = np.ceil(len(feature_idx) / self.n_features_at_a_time).astype(int)
        feature_idx = np.array_split(feature_idx, n_subarrays)
        feature_idx = [x.tolist() for x in feature_idx]

        # write dead into file so we can create them as dead in Neuronpedia
        skipped_indexes = set(range(self.n_features)) - set(self.target_feature_indexes)
        skipped_indexes_json = json.dumps({"skipped_indexes": list(skipped_indexes)})
        with open(f"{self.neuronpedia_folder}/skipped_indexes.json", "w") as f:
            f.write(skipped_indexes_json)

        print(f"Total features to run: {len(self.target_feature_indexes)}")
        print(f"Total skipped: {len(skipped_indexes)}")
        print(f"Total batches: {len(feature_idx)}")

        print(f"Hook Point Layer: {self.sparse_autoencoder.cfg.hook_point_layer}")
        print(f"Hook Point: {self.sparse_autoencoder.cfg.hook_point}")
        print(f"Writing files to: {self.neuronpedia_folder}")

        # get tokens:
        start = time.time()
        tokens = self.get_tokens(
            self.n_batches_to_sample_from, self.n_prompts_to_select
        )
        end = time.time()
        print(f"Time to get tokens: {end - start}")

        vocab_dict = cast(Any, self.model.tokenizer).vocab
        vocab_dict = {
            v: k.replace("Ġ", " ").replace("\n", "\\n").replace("Ċ", "\n")
            for k, v in vocab_dict.items()
        }
        # pad with blank tokens to the actual vocab size
        for i in range(len(vocab_dict), self.model.cfg.d_vocab):
            vocab_dict[i] = " "

        with torch.no_grad():
            feature_batch_count = 0
            for features_to_process in tqdm(feature_idx):
                print(f"Doing batch: {feature_batch_count}")
                feature_batch_count = feature_batch_count + 1
                feature_vis_params = FeatureVisParams(
                    hook_point=self.sparse_autoencoder.cfg.hook_point,
                    n_groups=10,
                    minibatch_size_features=256,
                    minibatch_size_tokens=64,
                    first_group_size=20,
                    other_groups_size=5,
                    buffer=(self.buffer_tokens_left, self.buffer_tokens_right),
                    features=features_to_process,
                    verbose=False,
                    include_left_tables=True,
                )

                feature_data = get_feature_data(
                    encoder=self.sparse_autoencoder,  # type: ignore
                    model=self.model,
                    tokens=tokens,
                    fvp=feature_vis_params,
                )

                features_outputs = []
                for _, feat_index in enumerate(feature_data.keys()):
                    feature = feature_data[feat_index]

                    feature_output = {}
                    feature_output["featureIndex"] = feat_index

                    top10_logits = self.round_list(
                        feature.middle_plots_data.top10_logits
                    )
                    bottom10_logits = self.round_list(
                        feature.middle_plots_data.bottom10_logits
                    )

                    # TODO: don't precompute/store these. should do it on the frontend
                    max_value = max(
                        np.absolute(bottom10_logits).max(),
                        np.absolute(top10_logits).max(),
                    )
                    neg_bg_values = self.round_list(
                        np.absolute(bottom10_logits) / max_value
                    )
                    pos_bg_values = self.round_list(
                        np.absolute(top10_logits) / max_value
                    )
                    feature_output["neg_bg_values"] = neg_bg_values
                    feature_output["pos_bg_values"] = pos_bg_values

                    if feature.left_tables_data:
                        feature_output["neuron_alignment_indices"] = (
                            feature.left_tables_data.neuron_alignment_indices
                        )
                        feature_output["neuron_alignment_values"] = self.round_list(
                            feature.left_tables_data.neuron_alignment_values
                        )
                        feature_output["neuron_alignment_l1"] = self.round_list(
                            feature.left_tables_data.neuron_alignment_l1
                        )
                        feature_output["correlated_neurons_indices"] = (
                            feature.left_tables_data.correlated_neurons_indices
                        )
                        feature_output["correlated_neurons_l1"] = self.round_list(
                            feature.left_tables_data.correlated_neurons_l1
                        )
                        feature_output["correlated_neurons_pearson"] = self.round_list(
                            feature.left_tables_data.correlated_neurons_pearson
                        )
                        # feature_output["correlated_features_indices"] = (
                        #     feature.left_tables_data.correlated_features_indices
                        # )
                        # feature_output["correlated_features_l1"] = self.round_list(
                        #     feature.left_tables_data.correlated_features_l1
                        # )
                        # feature_output["correlated_features_pearson"] = self.round_list(
                        #     feature.left_tables_data.correlated_features_pearson
                        # )

                    feature_output["neg_str"] = to_str_tokens(
                        vocab_dict, feature.middle_plots_data.bottom10_token_ids
                    )
                    feature_output["neg_values"] = bottom10_logits
                    feature_output["pos_str"] = to_str_tokens(
                        vocab_dict, feature.middle_plots_data.top10_token_ids
                    )
                    feature_output["pos_values"] = top10_logits

                    feature_output["frac_nonzero"] = (
                        feature.middle_plots_data.frac_nonzero
                    )

                    freq_hist_data = feature.middle_plots_data.freq_histogram_data
                    freq_bar_values = self.round_list(freq_hist_data.bar_values)
                    feature_output["freq_hist_data_bar_values"] = freq_bar_values
                    feature_output["freq_hist_data_tick_vals"] = self.round_list(
                        freq_hist_data.tick_vals
                    )

                    # TODO: don't precompute/store these. should do it on the frontend
                    freq_bar_values_clipped = [
                        (0.4 * max(freq_bar_values) + 0.6 * v) / max(freq_bar_values)
                        for v in freq_bar_values
                    ]
                    freq_bar_colors = [
                        colors.rgb2hex(BG_COLOR_MAP(v)) for v in freq_bar_values_clipped
                    ]
                    feature_output["freq_hist_data_bar_heights"] = self.round_list(
                        freq_hist_data.bar_heights
                    )
                    feature_output["freq_bar_colors"] = freq_bar_colors

                    logits_hist_data = feature.middle_plots_data.logits_histogram_data
                    feature_output["logits_hist_data_bar_heights"] = self.round_list(
                        logits_hist_data.bar_heights
                    )
                    feature_output["logits_hist_data_bar_values"] = self.round_list(
                        logits_hist_data.bar_values
                    )
                    feature_output["logits_hist_data_tick_vals"] = self.round_list(
                        logits_hist_data.tick_vals
                    )

                    # TODO: check this
                    feature_output["num_tokens_for_dashboard"] = (
                        self.n_prompts_to_select
                    )

                    activations = []
                    sdbs = feature.sequence_data
                    for sgd in sdbs.seq_group_data:
                        for sd in sgd.seq_data:
                            if (
                                sd.top5_token_ids is not None
                                and sd.bottom5_token_ids is not None
                                and sd.top5_logits is not None
                                and sd.bottom5_logits is not None
                            ):
                                activation = {}
                                strs = []
                                posContribs = []
                                negContribs = []
                                for i in range(len(sd.token_ids)):
                                    strs.append(
                                        to_str_tokens(vocab_dict, sd.token_ids[i])
                                    )
                                    posContrib = {}
                                    posTokens = [
                                        to_str_tokens(vocab_dict, j)
                                        for j in sd.top5_token_ids[i]
                                    ]
                                    if len(posTokens) > 0:
                                        posContrib["t"] = posTokens
                                        posContrib["v"] = self.round_list(
                                            sd.top5_logits[i]
                                        )
                                    posContribs.append(posContrib)
                                    negContrib = {}
                                    negTokens = [
                                        to_str_tokens(vocab_dict, j)
                                        for j in sd.bottom5_token_ids[i]
                                    ]
                                    if len(negTokens) > 0:
                                        negContrib["t"] = negTokens
                                        negContrib["v"] = self.round_list(
                                            sd.bottom5_logits[i]
                                        )
                                    negContribs.append(negContrib)

                                activation["logitContributions"] = json.dumps(
                                    {"pos": posContribs, "neg": negContribs}
                                )
                                activation["tokens"] = strs
                                activation["values"] = self.round_list(sd.feat_acts)
                                activation["maxValue"] = max(activation["values"])
                                activation["lossValues"] = self.round_list(
                                    sd.contribution_to_loss
                                )

                                activations.append(activation)
                    feature_output["activations"] = activations

                    features_outputs.append(feature_output)

                json_object = json.dumps(features_outputs, cls=NpEncoder)

                with open(
                    f"{self.neuronpedia_folder}/batch-{feature_batch_count}.json", "w"
                ) as f:
                    f.write(json_object)

        return
