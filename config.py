from enum import Enum
VICUNA_PATH = "/home/pchao/vicuna-13b-v1.5"
LLAMA_PATH = "/home/pchao/Llama-2-7b-chat-hf"

ATTACK_TEMP = 1
TARGET_TEMP = 0
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1


## MODEL PARAMETERS ##
class Model(Enum):
    vicuna = "vicuna-13b-v1.5"
    llama_2 = "llama-2-7b-chat-hf"
    gpt_3_5 = "gpt-3.5-turbo-1106"
    gpt_4 = "gpt-4-0125-preview"
    claude_1 = "claude-instant-1.2"
    claude_2 = "claude-2.1"
    gemini = "gemini-pro"
    mixtral = "mixtral"
    llama_3_3_70b_turbo = "llama-3.3-70b-instruct-turbo"
    gemma_4_31b_it = "gemma-4-31b-it"
    gemma_3n_e4b_it = "gemma-3n-e4b-it"
    llama_3_8b_instruct_lite = "llama-3-8b-instruct-lite"
    qwen_2_5_7b_instruct_turbo = "qwen-2.5-7b-instruct-turbo"
    llama_guard_4_12b = "llama-guard-4-12b"
    ucsd_mistral_small = "api-mistral-small-3.2-2506"
    ucsd_mistral_large = "mistral.mistral-large-3-675b-instruct"
    ucsd_deepseek = "api-deepseek-v4-flash"
    ucsd_llama_4_scout = "api-llama-4-scout"
    ucsd_claude_sonnet = "claude-sonnet-4-6"

MODEL_NAMES = [model.value for model in Model]


HF_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "meta-llama/Llama-2-7b-chat-hf",
    Model.vicuna: "lmsys/vicuna-13b-v1.5",
    Model.mixtral: "mistralai/Mixtral-8x7B-Instruct-v0.1",
    Model.llama_3_3_70b_turbo: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    Model.gemma_4_31b_it: "google/gemma-4-31B-it",
    Model.gemma_3n_e4b_it: "google/gemma-3n-E4B-it",
    Model.llama_3_8b_instruct_lite: "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    Model.qwen_2_5_7b_instruct_turbo: "Qwen/Qwen2.5-7B-Instruct-Turbo",
    Model.llama_guard_4_12b: "meta-llama/Llama-Guard-4-12B",
}

TOGETHER_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "together_ai/meta-llama/Llama-2-7b-chat-hf",
    Model.vicuna: "together_ai/lmsys/vicuna-13b-v1.5",
    Model.mixtral: "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
    Model.llama_3_3_70b_turbo: "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    Model.gemma_4_31b_it: "together_ai/google/gemma-4-31B-it",
    Model.gemma_3n_e4b_it: "together_ai/google/gemma-3n-E4B-it",
    Model.llama_3_8b_instruct_lite: "together_ai/meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    Model.qwen_2_5_7b_instruct_turbo: "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
    Model.llama_guard_4_12b: "together_ai/meta-llama/Llama-Guard-4-12B",
}

OPENAI_COMPATIBLE_MODEL_NAMES: dict[Model, str] = {
    Model.ucsd_mistral_small: "openai/api-mistral-small-3.2-2506",
    Model.ucsd_mistral_large: "openai/mistral.mistral-large-3-675b-instruct",
    Model.ucsd_deepseek: "openai/api-deepseek-v4-flash",
    Model.ucsd_llama_4_scout: "openai/api-llama-4-scout",
    Model.ucsd_claude_sonnet: "openai/claude-sonnet-4-6",
}

API_BASE_ENV_NAMES: dict[Model, str] = {
    Model.ucsd_mistral_small: "OPENAI_BASE_URL",
    Model.ucsd_mistral_large: "OPENAI_BASE_URL",
    Model.ucsd_deepseek: "OPENAI_BASE_URL",
    Model.ucsd_llama_4_scout: "OPENAI_BASE_URL",
    Model.ucsd_claude_sonnet: "OPENAI_BASE_URL",
}

FASTCHAT_TEMPLATE_NAMES: dict[Model, str] = {
    Model.gpt_3_5: "gpt-3.5-turbo",
    Model.gpt_4: "gpt-4",
    Model.claude_1: "claude-instant-1.2",
    Model.claude_2: "claude-2.1",
    Model.gemini: "gemini-pro",
    Model.vicuna: "vicuna_v1.1",
    Model.llama_2: "llama-2-7b-chat-hf",
    Model.mixtral: "mixtral",
    Model.llama_3_3_70b_turbo: "llama-3",
    Model.gemma_4_31b_it: "gemma",
    Model.gemma_3n_e4b_it: "gemma",
    Model.llama_3_8b_instruct_lite: "llama-3",
    Model.qwen_2_5_7b_instruct_turbo: "qwen-7b-chat",
    Model.llama_guard_4_12b: "llama-3",
    Model.ucsd_mistral_small: "gpt-3.5-turbo",
    Model.ucsd_mistral_large: "gpt-3.5-turbo",
    Model.ucsd_deepseek: "gpt-3.5-turbo",
    Model.ucsd_llama_4_scout: "gpt-3.5-turbo",
    Model.ucsd_claude_sonnet: "gpt-3.5-turbo",
}

API_KEY_NAMES: dict[Model, str] = {
    Model.gpt_3_5:  "OPENAI_API_KEY",
    Model.gpt_4:    "OPENAI_API_KEY",
    Model.claude_1: "ANTHROPIC_API_KEY",
    Model.claude_2: "ANTHROPIC_API_KEY",
    Model.gemini:   "GEMINI_API_KEY",
    Model.vicuna:   "TOGETHER_API_KEY",
    Model.llama_2:  "TOGETHER_API_KEY",
    Model.mixtral:  "TOGETHER_API_KEY",
    Model.llama_3_3_70b_turbo: "TOGETHER_API_KEY",
    Model.gemma_4_31b_it: "TOGETHER_API_KEY",
    Model.gemma_3n_e4b_it: "TOGETHER_API_KEY",
    Model.llama_3_8b_instruct_lite: "TOGETHER_API_KEY",
    Model.qwen_2_5_7b_instruct_turbo: "TOGETHER_API_KEY",
    Model.llama_guard_4_12b: "TOGETHER_API_KEY",
    Model.ucsd_mistral_small: "OPENAI_API_KEY",
    Model.ucsd_mistral_large: "OPENAI_API_KEY",
    Model.ucsd_deepseek: "OPENAI_API_KEY",
    Model.ucsd_llama_4_scout: "OPENAI_API_KEY",
    Model.ucsd_claude_sonnet: "OPENAI_API_KEY",
}

LITELLM_TEMPLATES: dict[Model, dict] = {
    Model.vicuna: {"roles":{
                    "system": {"pre_message": "", "post_message": " "},
                    "user": {"pre_message": "USER: ", "post_message": " ASSISTANT:"},
                    "assistant": {
                        "pre_message": "",
                        "post_message": "",
                    },
                },
                "post_message":"</s>",
                "initial_prompt_value" : "",
                "eos_tokens": ["</s>"]         
                },
    Model.llama_2: {"roles":{
                    "system": {"pre_message": "[INST] <<SYS>>\n", "post_message": "\n<</SYS>>\n\n"},
                    "user": {"pre_message": "", "post_message": " [/INST]"},
                    "assistant": {"pre_message": "", "post_message": ""},
                },
                "post_message" : " </s><s>",
                "initial_prompt_value" : "",
                "eos_tokens" :  ["</s>", "[/INST]"]  
            },
    Model.mixtral: {"roles":{
                    "system": {
                        "pre_message": "[INST] ",
                        "post_message": " [/INST]"
                    },
                    "user": { 
                        "pre_message": "[INST] ",
                        "post_message": " [/INST]"
                    }, 
                    "assistant": {
                        "pre_message": " ",
                        "post_message": "",
                    }
                },
                "post_message": "</s>",
                "initial_prompt_value" : "<s>",
                "eos_tokens": ["</s>", "[/INST]"]
    },
    Model.llama_3_3_70b_turbo: {"roles":{
                    "system": {
                        "pre_message": "<|start_header_id|>system<|end_header_id|>\n\n",
                        "post_message": "<|eot_id|>",
                    },
                    "user": {
                        "pre_message": "<|start_header_id|>user<|end_header_id|>\n\n",
                        "post_message": "<|eot_id|>",
                    },
                    "assistant": {
                        "pre_message": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                        "post_message": "<|eot_id|>",
                    },
                },
                "post_message": "<|eot_id|>",
                "initial_prompt_value": "<|begin_of_text|>",
                "eos_tokens": ["<|eot_id|>"],
    }
}
