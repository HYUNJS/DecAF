import torch, math
from qwen_vl_utils import process_vision_info

from evaluation.inf_utils import (
    PROMPT_TMPL_OBJ_CLS, PROMPT_TMPL_BACKGROUND, PROMPT_TMPL_OBJ_CHOICE, 
    get_sharegpt_format_messages, get_sharegpt_format_text_messages, get_internvl_sharegpt_format_messages,
    SYS_QWEN2, SYS_INTERNVL,
    PROMPT_TMPL_OBJ_CLS_abl_v1, PROMPT_TMPL_OBJ_CLS_abl_v2, PROMPT_TMPL_OBJ_CLS_abl_v3,
    PROMPT_TMPL_BACKGROUND_abl_v1, PROMPT_TMPL_BACKGROUND_abl_v2, PROMPT_TMPL_BACKGROUND_abl_v3,
    PROMPT_TMPL_OBJ_CHOICE_abl_v1, PROMPT_TMPL_OBJ_CHOICE_abl_v2, PROMPT_TMPL_OBJ_CHOICE_abl_v3,
)

MAX_BATCH = 8

## Qwen LLM prompting functions
def obj_cls_predict_prompting(model, processor, ref_query, frame_file_path, video_flag, tokens_per_frame=768, resized_height=-1, resized_width=-1, prompt_ver=0):
    gen_kwargs = {
        "max_new_tokens": 128,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": True,
        "temperature": None,
        "return_dict_in_generate": True,
        "output_attentions": True,
        "output_hidden_states": False,
    }
    if prompt_ver == 0:
        obj_cls_prompt = PROMPT_TMPL_OBJ_CLS.format(sent=ref_query.lower())
    elif prompt_ver == 1:
        obj_cls_prompt = PROMPT_TMPL_OBJ_CLS_abl_v1.format(sent=ref_query.lower())
    elif prompt_ver == 2:
        obj_cls_prompt = PROMPT_TMPL_OBJ_CLS_abl_v2.format(sent=ref_query.lower())
    elif prompt_ver == 3:
        obj_cls_prompt = PROMPT_TMPL_OBJ_CLS_abl_v3.format(sent=ref_query.lower())
    else:
        raise NotImplementedError(f"[Obj-focused] prompt_ver {prompt_ver} not implemented")

    msg_format_kwargs = {
        "is_video": video_flag,
        "tokens_per_frame": tokens_per_frame,
        "resized_height": resized_height,
        "resized_width": resized_width,
    }
    
    if video_flag:
        messages = [get_sharegpt_format_messages(obj_cls_prompt, frame_file_path, **msg_format_kwargs)]
        visual_pad_token_idx = processor.video_token_id
    else:
        messages = [get_sharegpt_format_messages(obj_cls_prompt, [frame_file_path[i]], **msg_format_kwargs) for i in range(len(frame_file_path))]
        visual_pad_token_idx = processor.image_token_id
    text = [processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in messages]

    batch_size = len(text)
    num_iter = math.ceil(batch_size / MAX_BATCH)
    obj_cls_batch_list, attn_weights_batch_list, visual_token_mask_batch_list, visual_grid_thw_batch_list = [], [], [], []
    for iter_i in range(num_iter):
        start_i, end_i = MAX_BATCH * iter_i, MAX_BATCH * (iter_i+1)
        messages_batch = messages[start_i:end_i]
        text_batch = text[start_i:end_i]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = processor(text=text_batch, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        visual_grid_thw = inputs['video_grid_thw'] if video_flag else inputs['image_grid_thw']
        visual_token_mask = inputs["input_ids"] == visual_pad_token_idx
        
        model.model.language_model.visual_token_mask = visual_token_mask
        
        with torch.no_grad():
            generated_outputs = model.generate(**inputs, **gen_kwargs)
        model.model.language_model.visual_token_mask = None

        generated_ids = generated_outputs.sequences
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        obj_cls_preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for obj_cls_pred in obj_cls_preds:
            obj_cls_batch_list.append(obj_cls_pred.strip())

        attn_weights = generated_outputs.attentions[0][0]
        attn_weights_batch_list.append(attn_weights.cpu())
        visual_token_mask_batch_list.append(visual_token_mask.cpu())
        visual_grid_thw_batch_list.append(visual_grid_thw.cpu())
        torch.cuda.empty_cache()
    
    visual_token_mask_batch = torch.cat(visual_token_mask_batch_list, dim=0)
    visual_grid_thw_batch = torch.cat(visual_grid_thw_batch_list, dim=0)
    attn_weights_batch = torch.cat(attn_weights_batch_list, dim=0)
    attn_weights_batch = tuple([attn_weights_batch])

    obj_cls_results = {
        "obj_cls_list": obj_cls_batch_list,
        "attn_weights_list": attn_weights_batch,
        "visual_token_mask": visual_token_mask_batch,
        "visual_grid_thw": visual_grid_thw_batch,
    }
    return obj_cls_results

def background_caption_prompting(model, processor, obj_name, frame_file_path, video_flag, tokens_per_frame=768, resized_height=-1, resized_width=-1, prompt_ver=0, ref_query=None):
    gen_kwargs = {
        "max_new_tokens": 128,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": True,
        "temperature": None,
        "return_dict_in_generate": True,
        "output_attentions": True,
        "output_hidden_states": False,
    }
    if prompt_ver == 0:
        background_prompt = PROMPT_TMPL_BACKGROUND.format(obj_name=obj_name)
    elif prompt_ver == 1:
        background_prompt = PROMPT_TMPL_BACKGROUND_abl_v1.format(obj_name=obj_name)
    elif prompt_ver == 2:
        background_prompt = PROMPT_TMPL_BACKGROUND_abl_v2.format(obj_name=obj_name)
    elif prompt_ver == 3:
        assert ref_query is not None, "ref_query must be provided for Background prompt_ver 2"
        background_prompt = PROMPT_TMPL_BACKGROUND_abl_v3.format(sent=ref_query.lower())
    else:
        raise NotImplementedError(f"[Bg-focused] prompt_ver {prompt_ver} not implemented")

    msg_format_kwargs = {
        "is_video": video_flag,
        "tokens_per_frame": tokens_per_frame,
        "resized_height": resized_height,
        "resized_width": resized_width,
    }

    if video_flag:
        messages = [get_sharegpt_format_messages(background_prompt, frame_file_path, **msg_format_kwargs)]
        visual_pad_token_idx = processor.video_token_id
    else:
        messages = [get_sharegpt_format_messages(background_prompt, [frame_file_path[i]], **msg_format_kwargs) for i in range(len(frame_file_path))]
        visual_pad_token_idx = processor.image_token_id
    text = [processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in messages]
    
    batch_size = len(text)
    num_iter = math.ceil(batch_size / MAX_BATCH)
    bg_caption_batch_list, attn_weights_batch_list, visual_token_mask_batch_list, visual_grid_thw_batch_list = [], [], [], []
    for iter_i in range(num_iter):
        start_i, end_i = MAX_BATCH * iter_i, MAX_BATCH * (iter_i+1)
        messages_batch = messages[start_i:end_i]
        text_batch = text[start_i:end_i]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = processor(text=text_batch, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        visual_grid_thw = inputs['video_grid_thw'] if video_flag else inputs['image_grid_thw']
        visual_token_mask = inputs["input_ids"] == visual_pad_token_idx

        model.model.language_model.visual_token_mask = visual_token_mask

        with torch.no_grad():
            generated_outputs = model.generate(**inputs, **gen_kwargs)
        model.model.language_model.visual_token_mask = None

        generated_ids = generated_outputs.sequences
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        background_captions = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for background_caption in background_captions:
            bg_caption_batch_list.append(background_caption.strip())

        attn_weights = generated_outputs.attentions[0][0]
        attn_weights_batch_list.append(attn_weights.cpu())
        visual_token_mask_batch_list.append(visual_token_mask.cpu())
        visual_grid_thw_batch_list.append(visual_grid_thw.cpu())
        torch.cuda.empty_cache()

    visual_token_mask_batch = torch.cat(visual_token_mask_batch_list, dim=0)
    visual_grid_thw_batch = torch.cat(visual_grid_thw_batch_list, dim=0)
    attn_weights_batch = torch.cat(attn_weights_batch_list, dim=0)
    attn_weights_batch = tuple([attn_weights_batch])

    bg_caption_results = {
        "bg_caption_list": bg_caption_batch_list,
        "attn_weights_list": attn_weights_batch,
        "visual_token_mask": visual_token_mask_batch,
        "visual_grid_thw": visual_grid_thw_batch,
    }
    return bg_caption_results


def obj_choice_prompting(model, processor, ref_query, obj_cls_list, prompt_ver=0):
    ## get class name from the candidate classes
    gen_kwargs = {
        "max_new_tokens": 128,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": True,
        "temperature": None,
    }
    ref_query = ref_query.lower()
    obj_cls_list = ",".join([obj.lower() for obj in obj_cls_list])
    if prompt_ver == 0:
        obj_choice_prompt = PROMPT_TMPL_OBJ_CHOICE.format(ref_query=ref_query, obj_cls_list=obj_cls_list)
    elif prompt_ver == 1:
        obj_choice_prompt = PROMPT_TMPL_OBJ_CHOICE_abl_v1.format(ref_query=ref_query, obj_cls_list=obj_cls_list)
    elif prompt_ver == 2:
        obj_choice_prompt = PROMPT_TMPL_OBJ_CHOICE_abl_v2.format(ref_query=ref_query, obj_cls_list=obj_cls_list)
    elif prompt_ver == 3:
        obj_choice_prompt = PROMPT_TMPL_OBJ_CHOICE_abl_v3.format(ref_query=ref_query)
    else:    
        raise NotImplementedError(f"[Obj Choice] prompt_ver {prompt_ver} not implemented")
    
    messages = [get_sharegpt_format_text_messages(obj_choice_prompt)]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=None, videos=None, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    obj_name_preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    obj_name = obj_name_preds[0].strip().lower()
    
    torch.cuda.empty_cache()

    return obj_name


## LLaVA-OV prompting functions

def obj_cls_predict_prompting_llavaov(model, processor, ref_query, frame_file_path, video_flag):
        gen_kwargs = {
            "max_new_tokens": 128,
            "num_beams": 1,
            "do_sample": False,
            "use_cache": True,
            "temperature": None,
            "return_dict_in_generate": True,
            "output_attentions": True,
            "output_hidden_states": False,
        }

        obj_cls_prompt = PROMPT_TMPL_OBJ_CLS.format(sent=ref_query.lower())

        if video_flag:
            messages = [SYS_QWEN2 + get_sharegpt_format_messages(obj_cls_prompt, frame_file_path, is_video=video_flag)]
            visual_pad_token_idx = processor.video_token_id
        else:
            messages = [SYS_QWEN2 + get_sharegpt_format_messages(obj_cls_prompt, [frame_file_path[i]], is_video=video_flag) for i in range(len(frame_file_path))]
            visual_pad_token_idx = processor.image_token_id

        batch_size = len(messages)
        num_iter = math.ceil(batch_size / MAX_BATCH)
        obj_cls_batch_list, attn_weights_batch_list, visual_token_mask_batch_list, visual_grid_thw_batch_list = [], [], [], []
        for iter_i in range(num_iter):
            start_i, end_i = MAX_BATCH * iter_i, MAX_BATCH * (iter_i+1)
            messages_batch = messages[start_i:end_i]
            inputs = processor.apply_chat_template(messages_batch, add_generation_prompt=True,
                        tokenize=True, return_dict=True, return_tensors="pt").to(model.device)

            visual_token_mask = inputs["input_ids"] == visual_pad_token_idx
            visual_grid_thw = inputs.pop('video_grid_thw') if video_flag else inputs.pop('image_grid_thw')
            
            model.model.language_model.visual_token_mask = visual_token_mask
            with torch.no_grad():
                generated_outputs = model.generate(**inputs, **gen_kwargs)
            model.model.language_model.visual_token_mask = None

            generated_ids = generated_outputs.sequences
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            obj_cls_preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for obj_cls_pred in obj_cls_preds:
                obj_cls_batch_list.append(obj_cls_pred.strip())

            attn_weights = generated_outputs.attentions[0][0]
            attn_weights_batch_list.append(attn_weights.cpu())
            visual_token_mask_batch_list.append(visual_token_mask.cpu())
            visual_grid_thw_batch_list.append(visual_grid_thw.cpu())
            
            torch.cuda.empty_cache()

        visual_token_mask_batch = torch.cat(visual_token_mask_batch_list, dim=0)
        visual_grid_thw_batch = torch.cat(visual_grid_thw_batch_list, dim=0)
        attn_weights_batch = torch.cat(attn_weights_batch_list, dim=0)
        attn_weights_batch = tuple([attn_weights_batch])

        obj_cls_results = {
            "obj_cls_list": obj_cls_batch_list,
            "attn_weights_list": attn_weights_batch,
            "visual_token_mask": visual_token_mask_batch,
            "visual_grid_thw": visual_grid_thw_batch,
        }
        return obj_cls_results

def background_caption_prompting_llavaov(model, processor, obj_name, frame_file_path, video_flag):
    gen_kwargs = {
        "max_new_tokens": 128,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": True,
        "temperature": None,
        "return_dict_in_generate": True,
        "output_attentions": True,
        "output_hidden_states": False,
    }
    background_prompt = PROMPT_TMPL_BACKGROUND.format(obj_name=obj_name)

    if video_flag:
        messages = [SYS_QWEN2 + get_sharegpt_format_messages(background_prompt, frame_file_path, is_video=video_flag)]
        visual_pad_token_idx = processor.video_token_id
    else:
        messages = [SYS_QWEN2 + get_sharegpt_format_messages(background_prompt, [frame_file_path[i]], is_video=video_flag) for i in range(len(frame_file_path))]
        visual_pad_token_idx = processor.image_token_id
    
    batch_size = len(messages)
    num_iter = math.ceil(batch_size / MAX_BATCH)
    bg_caption_batch_list, attn_weights_batch_list, visual_token_mask_batch_list, visual_grid_thw_batch_list = [], [], [], []
    for iter_i in range(num_iter):
        start_i, end_i = MAX_BATCH * iter_i, MAX_BATCH * (iter_i+1)
        messages_batch = messages[start_i:end_i]
        inputs = processor.apply_chat_template(messages_batch, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt").to(model.device)

        visual_token_mask = inputs["input_ids"] == visual_pad_token_idx
        visual_grid_thw = inputs.pop('video_grid_thw') if video_flag else inputs.pop('image_grid_thw')

        model.model.language_model.visual_token_mask = visual_token_mask
        with torch.no_grad():
            generated_outputs = model.generate(**inputs, **gen_kwargs)
        model.model.language_model.visual_token_mask = None

        generated_ids = generated_outputs.sequences
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        background_captions = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for background_caption in background_captions:
            bg_caption_batch_list.append(background_caption.strip())

        attn_weights = generated_outputs.attentions[0][0]
        attn_weights_batch_list.append(attn_weights.cpu())
        visual_token_mask_batch_list.append(visual_token_mask.cpu())
        visual_grid_thw_batch_list.append(visual_grid_thw.cpu())
            
        torch.cuda.empty_cache()

    visual_token_mask_batch = torch.cat(visual_token_mask_batch_list, dim=0)
    visual_grid_thw_batch = torch.cat(visual_grid_thw_batch_list, dim=0)
    attn_weights_batch = torch.cat(attn_weights_batch_list, dim=0)
    attn_weights_batch = tuple([attn_weights_batch])


    bg_caption_results = {
        "bg_caption_list": bg_caption_batch_list,
        "attn_weights_list": attn_weights_batch,
        "visual_token_mask": visual_token_mask_batch,
        "visual_grid_thw": visual_grid_thw_batch,
    }
    return bg_caption_results

def obj_choice_prompting_llavaov(model, processor, ref_query, obj_cls_list):
    ## get class name from the candidate classes
    gen_kwargs = {
        "max_new_tokens": 128,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": True,
        "temperature": None,
    }
    ref_query = ref_query.lower()
    obj_cls_list = ",".join([obj.lower() for obj in obj_cls_list])
    obj_choice_prompt = PROMPT_TMPL_OBJ_CHOICE.format(ref_query=ref_query, obj_cls_list=obj_cls_list)

    messages = [SYS_QWEN2 + get_sharegpt_format_text_messages(obj_choice_prompt)]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=None, videos=None, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    obj_name_preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    obj_name = obj_name_preds[0].strip().lower()
    
    torch.cuda.empty_cache()

    return obj_name


## InternVL prompting functions

def obj_cls_predict_prompting_internvl(model, processor, ref_query, frame_file_path, video_flag, tiling_hw=None):
    gen_kwargs = {
        "max_new_tokens": 128,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": True,
        "temperature": None,
        "return_dict_in_generate": True,
        "output_attentions": True,
        "output_hidden_states": False,
    }

    obj_cls_prompt = PROMPT_TMPL_OBJ_CLS.format(sent=ref_query.lower())

    msg_format_kwargs = {
        "is_video": video_flag,
    }
    
    if video_flag:
        messages = [SYS_INTERNVL + get_internvl_sharegpt_format_messages(obj_cls_prompt, frame_file_path, **msg_format_kwargs)]
    else:
        messages = [SYS_INTERNVL + get_internvl_sharegpt_format_messages(obj_cls_prompt, [frame_file_path[i]], **msg_format_kwargs) for i in range(len(frame_file_path))]
    visual_pad_token_idx = processor.image_token_id
    
    batch_size = len(messages)
    num_iter = math.ceil(batch_size / MAX_BATCH)
    obj_cls_batch_list, attn_weights_batch_list, visual_token_mask_batch_list = [], [], []
    for iter_i in range(num_iter):
        start_i, end_i = MAX_BATCH * iter_i, MAX_BATCH * (iter_i+1)
        messages_batch = messages[start_i:end_i]
        inputs = processor.apply_chat_template(
            messages_batch,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        ).to(model.device, dtype=model.dtype)
        visual_token_mask = inputs["input_ids"] == visual_pad_token_idx
        model.model.language_model.visual_token_mask = visual_token_mask
        with torch.no_grad():
            generated_outputs = model.generate(**inputs, **gen_kwargs)
        model.model.language_model.visual_token_mask = None

        generated_ids = generated_outputs.sequences
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        obj_cls_preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for obj_cls_pred in obj_cls_preds:
            obj_cls_batch_list.append(obj_cls_pred.strip())

        attn_weights = generated_outputs.attentions[0][0].cpu()
        attn_weights_batch_list.append(attn_weights.cpu())
        visual_token_mask_batch_list.append(visual_token_mask.cpu())
        torch.cuda.empty_cache()

    visual_token_mask_batch = torch.cat(visual_token_mask_batch_list, dim=0)
    attn_weights_batch = torch.cat(attn_weights_batch_list, dim=0)
    if video_flag:
        # visual_grid_thw = torch.tensor([len(frame_file_path), 32, 32])
        visual_grid_thw = torch.tensor([len(frame_file_path), 16, 16])
        tiling_hw = None
    else:
        # visual_grid_thw = torch.tensor([1, 32, 32])
        visual_grid_thw = torch.tensor([1, 16, 16])
        tiling_hw = torch.tensor(tiling_hw) if tiling_hw is not None else torch.tensor([1, 1])
    
    obj_cls_results = {
        "obj_cls_list": obj_cls_batch_list,
        "attn_weights_list": tuple([attn_weights_batch]),
        "visual_token_mask": visual_token_mask_batch,
        "visual_grid_thw": tuple([visual_grid_thw]),
        "tiling_hw": tuple([tiling_hw]) if tiling_hw is not None else None,
    }
    return obj_cls_results


def obj_choice_prompting_internvl(model, processor, ref_query, obj_cls_list):
    ## get class name from the candidate classes
    gen_kwargs = {
        "max_new_tokens": 128,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": True,
        "temperature": None,
    }
    ref_query = ref_query.lower()
    obj_cls_list = ",".join([obj.lower() for obj in obj_cls_list])
    obj_choice_prompt = PROMPT_TMPL_OBJ_CHOICE.format(ref_query=ref_query, obj_cls_list=obj_cls_list)

    messages = [SYS_INTERNVL + get_sharegpt_format_text_messages(obj_choice_prompt)]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=None, videos=None, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    obj_name_preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    obj_name = obj_name_preds[0].strip().lower()
    
    torch.cuda.empty_cache()

    return obj_name


def background_caption_prompting_internvl(model, processor, obj_name, frame_file_path, video_flag, tiling_hw=None):
    gen_kwargs = {
        "max_new_tokens": 128,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": True,
        "temperature": None,
        "return_dict_in_generate": True,
        "output_attentions": True,
        "output_hidden_states": False,
    }
    background_prompt = PROMPT_TMPL_BACKGROUND.format(obj_name=obj_name)

    msg_format_kwargs = {
        "is_video": video_flag,
    }

    if video_flag:
        messages = [SYS_INTERNVL + get_internvl_sharegpt_format_messages(background_prompt, frame_file_path, **msg_format_kwargs)]
    else:
        messages = [SYS_INTERNVL + get_internvl_sharegpt_format_messages(background_prompt, [frame_file_path[i]], **msg_format_kwargs) for i in range(len(frame_file_path))]
    visual_pad_token_idx = processor.image_token_id

    batch_size = len(messages)
    num_iter = math.ceil(batch_size / MAX_BATCH)
    
    bg_caption_batch_list, attn_weights_batch_list, visual_token_mask_batch_list = [], [], []
    for iter_i in range(num_iter):
        start_i, end_i = MAX_BATCH * iter_i, MAX_BATCH * (iter_i+1)
        messages_batch = messages[start_i:end_i]
        inputs = processor.apply_chat_template(
                messages_batch,
                return_tensors="pt",
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            ).to(model.device, dtype=model.dtype)
        visual_token_mask = inputs["input_ids"] == visual_pad_token_idx
        model.model.language_model.visual_token_mask = visual_token_mask

        with torch.no_grad():
            generated_outputs = model.generate(**inputs, **gen_kwargs)
        model.model.language_model.visual_token_mask = None

        generated_ids = generated_outputs.sequences
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        background_captions = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for background_caption in background_captions:
            bg_caption_batch_list.append(background_caption.strip())

        attn_weights = generated_outputs.attentions[0][0].cpu()
        attn_weights_batch_list.append(attn_weights.cpu())
        visual_token_mask_batch_list.append(visual_token_mask.cpu())
        torch.cuda.empty_cache()
    
    visual_token_mask_batch = torch.cat(visual_token_mask_batch_list, dim=0)
    attn_weights_batch = torch.cat(attn_weights_batch_list, dim=0)
    if video_flag:
        # visual_grid_thw = torch.tensor([len(frame_file_path), 32, 32])
        visual_grid_thw = torch.tensor([len(frame_file_path), 16, 16])
        tiling_hw = None
    else:
        # visual_grid_thw = torch.tensor([1, 32, 32])
        visual_grid_thw = torch.tensor([1, 16, 16])
        tiling_hw = torch.tensor(tiling_hw) if tiling_hw is not None else torch.tensor([1, 1])

    bg_caption_results = {
        "bg_caption_list": bg_caption_batch_list,
        "attn_weights_list": tuple([attn_weights_batch]),
        "visual_token_mask": visual_token_mask_batch,
        "visual_grid_thw": tuple([visual_grid_thw]),
        "tiling_hw": tuple([tiling_hw]) if tiling_hw is not None else None,
    }
    return bg_caption_results