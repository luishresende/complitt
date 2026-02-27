import os
import json


def extract_response_metadata(response):
    """
    Extrai informações principais de um GenerateContentResponse e retorna um dicionário.
    """
    # Extrai tokens de uso
    usage = {
        "prompt_token_count": response.usage_metadata.prompt_token_count,
        "candidates_token_count": response.usage_metadata.candidates_token_count,
        "thoughts_token_count": response.usage_metadata.thoughts_token_count,
        "total_token_count": response.usage_metadata.total_token_count,
        "prompt_tokens_details": [
            {
                "modality": detail.modality.value,  # transforma Enum em string
                "token_count": detail.token_count
            } for detail in response.usage_metadata.prompt_tokens_details
        ]
    }

    # Extrai informações básicas de candidatos
    candidates_info = [
        {
            "index": c.index,
            "finish_reason": c.finish_reason.value
        } for c in response.candidates
    ]

    # Monta o JSON simplificado
    simplified_response = {
        "response_id": response.response_id,
        "model_version": response.model_version,
        "usage": usage,
        "candidates": candidates_info
    }

    return simplified_response


def save_results(output_results_dir, uuid, prompt, response):
    model_output = response.text
    response_metadata = extract_response_metadata(response)
    response_metadata["prompt"] = prompt


    output_metadata_path = os.path.join(output_results_dir, "metadata")
    output_model_path = os.path.join(output_results_dir, "contents")
    os.makedirs(output_metadata_path, exist_ok=True)
    os.makedirs(output_model_path, exist_ok=True)

    metadata_path = os.path.join(output_metadata_path, f'{uuid}.json')
    model_output_path = os.path.join(output_model_path, f'{uuid}.txt')

    with open(metadata_path, 'w') as metadata_file:
        json.dump(response_metadata, metadata_file)

    with open(model_output_path, 'w') as model_output_file:
        model_output_file.write(model_output)

