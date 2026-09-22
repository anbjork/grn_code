

def inference_simulated(output_path, inference_functions):

    (output_path / 'inferences.pkl').unlink(missing_ok=True)

    from grn_code.pipeline_code import inference
    for inference_function in inference_functions:
        inference(
            data_path = output_path / 'preprocessed_data.pkl',
            output_path = output_path / 'inferences.pkl',
            method_function = inference_function,
            )

