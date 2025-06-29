import torch
from datetime import datetime, timedelta

def split_text(text, max_length):
    lines = []
    while len(text) > max_length:
        split_index = text[:max_length].rfind(' ')
        if split_index == -1:
            split_index = max_length
        lines.append(text[:split_index])
        text = text[split_index:].lstrip()
    lines.append(text)
    return lines

def generate_prompt(preds, metadata, region_ids, history_data, num_nodes, output_len, debug_log):
    input_prompts = []
    personalized_prompts = []
    output_values_list = []

    # Filter invalid region IDs
    valid_region_ids = [rid for rid in region_ids if rid < num_nodes]
    if len(valid_region_ids) < len(region_ids):
        invalid_ids = set(region_ids) - set(valid_region_ids)
        print(f"Warning: Region IDs {invalid_ids} are out of the node number range and have been filtered.")

    def format_time_window(base_time, steps):
        time_strs = []
        for step in steps:
            try:
                current_time = base_time + timedelta(minutes=30 * step)
                # Force the correctness of the weekday information
                true_weekday = (current_time.weekday() + 1) % 7
                time_strs.append(
                    f"{current_time.year}-{current_time.month:02d}-{current_time.day:02d} "
                    f"{['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][true_weekday]} "
                    f"{current_time.hour:02d}:{current_time.minute:02d}"
                )
            except Exception as e:
                debug_log.write(f"Time format error: {str(e)}\n")
                time_strs.append("Time parsing error")
        return f"{time_strs[0]} to {time_strs[-1]}"

    # Ensure the metadata dimensions are correct
    if metadata.dim() != 4 or metadata.size(-1) < 6:
        debug_log.write(f"Invalid metadata dimensions: {metadata.shape}\n")
        metadata = torch.zeros_like(metadata)  # Prevent index out of bounds

    # Process prediction values (add inverse normalization and truncation)
    preds = torch.clamp(preds, min=0.0)

    future_output_len = 2

    for b in range(preds.size(0)):
        try:
            # Use the metadata of the first time step as the benchmark
            first_meta = metadata[b, 0, 0, :6].numpy().astype(int)
            base_time = datetime(*first_meta[:5])
            true_weekday = (base_time.weekday() + 1) % 7
        except Exception as e:
            # Fault-tolerant processing: Reconstruct the time from historical data
            try:
                history_time = history_data[b, 0, 0, :5].numpy().astype(int)
                base_time = datetime(*history_time)
                true_weekday = (base_time.weekday() + 1) % 7
            except:
                base_time = datetime(2016, 4, 1, 0, 0)
                true_weekday = 5

        # Dynamic time window
        history_steps = list(range(0, output_len))
        predict_steps = list(range(output_len, output_len + future_output_len))

        history_time_window = format_time_window(base_time, history_steps)
        predict_time_window = format_time_window(base_time, predict_steps)

        region_info_str = "---- Region Information: " + ", ".join([f"Region {rid}" for rid in valid_region_ids])

        personalized_prompt = [
            "You are a expert that can predict traffic demand.",
            "This is in the simulation fine-tuning stage, that is, to get familiar with the process of capturing spatio-temporal dependencies in advance on other cities similar to the target city, in order to prepare for the fine-tuning stage in the target city."
            ]

        max_length = 150
        personalized_prompt_split = []
        for prompt in personalized_prompt:
            personalized_prompt_split.extend(split_text(prompt, max_length))

        input_prompt = [
            "=" * 150,
            "*** Input Prompt ***",
            "Some important information is as follows:",
            region_info_str,
            f"---- Time Information: {history_time_window}",
        ]

        for rid in valid_region_ids:
            # Historical data
            region_history = history_data[b, rid, -output_len:].tolist()
            # Process negative values
            region_history = [max(0, x) for x in region_history]
            history_str = " ".join(f"{x:.1f}" for x in region_history)
            input_prompt.append(
                f"---- Region {rid} values in the past {output_len} historical periods ({output_len // 2} hours): {history_str}")
        input_prompt.append("=" * 150)

        output_values = [
            "=" * 150,
            "*** Output Values ***",
            region_info_str,
            f"---- Time Information: {predict_time_window}"
        ]

        for rid in valid_region_ids:
            # Prediction data
            region_pred = preds[b, :future_output_len, rid].tolist()
            predict_str = " ".join(f"{x:.1f}" for x in region_pred)
            output_values.append(
                f"---- Region {rid} values in the next {future_output_len} future periods (1 hour): {predict_str}")
        output_values.append("=" * 150)

        input_prompts.append("\n".join(input_prompt))
        personalized_prompts.append("\n".join([
            "=" * 150,
            "*** Personalized Prompts ***",
            "\n".join(personalized_prompt_split),
            "=" * 150
        ]))
        output_values_list.append("\n".join(output_values))

    return input_prompts, personalized_prompts, output_values_list