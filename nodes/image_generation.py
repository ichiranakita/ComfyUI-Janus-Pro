import torch
import numpy as np
from PIL import Image

class JanusImageGeneration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("JANUS_MODEL",),
                "processor": ("JANUS_PROCESSOR",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful photo of"
                }),
                "seed": ("INT", {
                    "default": 666666666666666,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_images"
    CATEGORY = "Janus-Pro"

    def generate_images(self, model, processor, prompt, seed, batch_size=1, temperature=1.0, cfg_weight=5.0, top_p=0.95):
        try:
            from janus.models import MultiModalityCausalLM
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        # Set up device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            # Ensure model is on the correct device
            model = model.to(device)
            # MPS optimization settings
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
        else:
            device = torch.device("cpu")
            model = model.to(device)
            
        # Set random seed
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        # Image generation parameters
        image_token_num = 576  # 24x24 patches
        img_size = 384  # Output image size
        patch_size = 16  # Size of each patch
        parallel_size = batch_size

        # Prepare conversation format
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        try:
            # Prepare input
            sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + processor.image_start_tag

            # Encode input text
            input_ids = processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids).to(device)

            # Prepare conditional and unconditional inputs
            tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int32, device=device)
            for i in range(parallel_size*2):
                tokens[i, :] = input_ids
                if i % 2 != 0:  # Unconditional input
                    tokens[i, 1:-1] = processor.pad_id

            # Get text embeddings
            with torch.device(device):
                inputs_embeds = model.language_model.get_input_embeddings()(tokens)

            # Generate image tokens
            generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int32, device=device)
            outputs = None

            # Autoregressive generation
            for i in range(image_token_num):
                with torch.device(device):
                    outputs = model.language_model.model(
                        inputs_embeds=inputs_embeds, 
                        use_cache=True, 
                        past_key_values=outputs.past_key_values if i != 0 else None
                    )
                    hidden_states = outputs.last_hidden_state
                    
                    # Get logits and apply CFG
                    logits = model.gen_head(hidden_states[:, -1, :])
                    logit_cond = logits[0::2, :]
                    logit_uncond = logits[1::2, :]
                    
                    logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
                    probs = torch.softmax(logits / temperature, dim=-1)

                    # Sample next token
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens[:, i] = next_token.squeeze(dim=-1)

                    # Prepare input for next step
                    next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                    img_embeds = model.prepare_gen_img_embeds(next_token)
                    inputs_embeds = img_embeds.unsqueeze(dim=1)

                # MPS synchronization
                if device.type == "mps":
                    torch.mps.synchronize()

            # Decode generated tokens to image
            with torch.device(device):
                dec = model.gen_vision_model.decode_code(
                    generated_tokens.to(dtype=torch.int32), 
                    shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
                )
            
            # MPS synchronization
            if device.type == "mps":
                torch.mps.synchronize()
            
            # Convert to numpy for processing
            dec = dec.to(torch.float32).cpu().numpy()
            
            # Ensure BCHW format
            if dec.shape[1] != 3:
                dec = np.repeat(dec, 3, axis=1)
            
            # Convert from [-1,1] to [0,1] range
            dec = (dec + 1) / 2
            
            # Ensure values are in [0,1] range
            dec = np.clip(dec, 0, 1)
            
            # Convert to ComfyUI format [B,C,H,W] -> [B,H,W,C]
            dec = np.transpose(dec, (0, 2, 3, 1))
            
            # Convert to tensor
            images = torch.from_numpy(dec).float()
            
            # Ensure correct format
            assert images.ndim == 4 and images.shape[-1] == 3, f"Unexpected shape: {images.shape}"
            
            return (images,)
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            raise e
        finally:
            # Clean up MPS cache
            if device.type == "mps":
                torch.mps.empty_cache()

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed