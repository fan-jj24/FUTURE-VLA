from swift.llm import PtEngine, InferRequest, RequestConfig, get_model_tokenizer, get_template, safe_snapshot_download
from swift.tuners import Swift

import logging
import re
import torch
from typing import List, Tuple, Dict, Any
from pathlib import Path
import sys

root_dir = Path(__file__).parent
sys.path.append(str(root_dir / '1d-tokenizer'))
from titok_codec import TiTokCodec
from transformers import AutoProcessor
import cv2
import numpy as np

Libero_MEAN = np.array(
    [
        0.06278137117624283,
        0.0868409126996994,
        -0.09037282317876816,
        0.0005407406715676188,
        0.005643361248075962,
        -0.005229088477790356,
        -0.04964079707860947*0,
    ],
    dtype=np.float32,
)
Libero_STD = np.array(
    [
        0.3355240225791931,
        0.3784470558166504,
        0.44472837448120117,
        0.03924351558089256*5,
        0.06339313089847565*5,
        0.07797032594680786*5,
        0.9987710118293762*0+1,
    ],
    dtype=np.float32,
)


class Policy:

    ActionREQUEST_CONFIG = RequestConfig(
        max_tokens=128,
        top_k=1,
        return_details=True,
    )
    HILREQUEST_CONFIG = RequestConfig(
        max_tokens=2048,
        top_k=1,
        return_details=True,
    )


    def __init__(self, model_id_or_path: str, checkpoint_path: str = None, request_type: str = "action", horizon: int = 16, view: int = 2):
        
        model_path = checkpoint_path if checkpoint_path else model_id_or_path
        
        if checkpoint_path:
            model_path = safe_snapshot_download(checkpoint_path)
            logging.info(f"Loading fine-tuned model from checkpoint: {model_path}")
        
        model, tokenizer = get_model_tokenizer(model_path)
        template_type = model.model_meta.template
        template = get_template(template_type, tokenizer)

        self.A_START_ID = template.A_START_ID
        self.A_END_ID = template.A_END_ID
        self.B_START_ID = template.B_START_ID
        self.B_END_ID = template.B_END_ID
        self.ACT_BASE_ID = template.ACT_BASE_ID
        self.REC_BASE_ID = template.REC_BASE_ID
        
        self.engine = PtEngine.from_model_template(model, template, max_batch_size=2)

        self.request_type = request_type
        self.request_config = self.ActionREQUEST_CONFIG if request_type == "action" else self.HILREQUEST_CONFIG
        self.horizon = horizon
        self.view = view
        self.obs_window = [None for _ in range(view * horizon)]
        self.text = "" 

        self._init_tok()

    def _init_tok(self, titok_ckpt_dir: str = 'ckpt/TiTok'):
        self.titok_codec = TiTokCodec(titok_ckpt_dir)
        self.FAST = AutoProcessor.from_pretrained('physical-intelligence/fast',trust_remote_code=True)
        self.FAST.decode([[300, 496, 1179, 273, 469, 265, 398, 289, 482, 278, 972, 953, 360, 416, 413, 262, 258]], time_horizon = self.horizon, action_dim = 7)
    
    def reset(self):
        self.obs_window = [None for _ in range(self.view * self.horizon)]
        self.text = ""
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info("Policy reset: observation window and text cleared.")
        
    def update_obs_window(self, new_obs):
        assert len(new_obs) == self.view, "Number of new observations must match the view size."
    
        if self.obs_window[0] is None:
            logging.info("Filling observation window for the first time.")
            for i in range(self.horizon):
                for j in range(self.view):
                    self.obs_window[i * self.view + j] = new_obs[j]
        else:
            self.obs_window = self.obs_window[self.view:] + new_obs

    def extract_request_data(self, request_data: dict):
        dict_obs = request_data.get("observations")
        new_obs = [dict_obs['agentview_image'], dict_obs['robot0_eye_in_hand_image']]
        self.update_obs_window(new_obs)

        text = request_data.get("text", "")
        self.text = text

    def decode_action_tokens(self, token_ids: List[int]) -> Dict[str, Any]:

        result = {
            'act_codes': [],
            'rec_codes': []
        }
        
        i = 0
        while i < len(token_ids):
            token_id = token_ids[i]
            
            if token_id == self.A_START_ID:
                i += 1
                act_codes = []
                while i < len(token_ids) and token_ids[i] != self.A_END_ID:
                    code = token_ids[i] - self.ACT_BASE_ID
                    if code < 0 or code >= 2048:
                        return None
                    act_codes.append(code)
                    i += 1
                result['act_codes'].extend(act_codes)
                if i < len(token_ids) and token_ids[i] == self.A_END_ID:
                    i += 1
                continue
            
            if token_id == self.B_START_ID:
                i += 1
                rec_codes = []
                while i < len(token_ids) and token_ids[i] != self.B_END_ID:
                    code = token_ids[i] - self.REC_BASE_ID
                    if code < 0 or code >= 4096:
                        return None
                    rec_codes.append(code)
                    i += 1
                result['rec_codes'].extend(rec_codes)
                if i < len(token_ids) and token_ids[i] == self.B_END_ID:
                    i += 1
                continue
            
            i += 1

        
        raw_act = self.FAST.decode([result['act_codes']])
        if raw_act is None:
            return None
        
        normed_act = (raw_act * Libero_STD) + Libero_MEAN

        video1_images, video2_images = [], []

        if self.request_type == "hil":
            if len(result['rec_codes']) % 32 != 0:
                return None

            N = len(result['rec_codes']) // 32
            result['rec_codes'] = [result['rec_codes'][i*32:(i+1)*32] for i in range(N)]

            rec_tokens = torch.tensor(result['rec_codes'], dtype=torch.long)
            images = self.titok_codec.decode_tokens(rec_tokens)
            
            video1_images, video2_images = self.images_to_videos(images)
        
        return {"chunk actions" : normed_act, "video1_images": video1_images, "video2_images": video2_images}
    
    def images_to_videos(
        self, 
        images: List, 
        output_dir: str = "./output_videos",
        fps: int = 5,
        video_name_prefix: str = "action"
    ) -> Tuple[str, str]:
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video1_images = [images[i] for i in range(0, 32, 2)]  
        video2_images = [images[i] for i in range(1, 32, 2)]  
        
        video1_path = output_dir / f"{video_name_prefix}_view1.mp4"
        video2_path = output_dir / f"{video_name_prefix}_view2.mp4"
        
        self._write_video(video1_images, str(video1_path), fps)
        self._write_video(video2_images, str(video2_path), fps)
        
        return video1_images, video2_images
    
    def _write_video(self, images: List, output_path: str, fps: int):
        if len(images) == 0:
            raise ValueError("Image list is empty")
        
        first_img = images[0]
        width, height = first_img.size
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for img in images:
            img_array = np.array(img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)
        
        writer.release()

    def infer(self, request_config: RequestConfig = None):
        if request_config is None:
            request_config = self.request_config

        infer_request = InferRequest(
            messages = [
            {'role': 'user', 'content': f"<image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image>Task: {self.text}. Based on the images, what action should be executed?"}],
            images = self.obs_window,
        )
        

        resp_list = self.engine.infer([infer_request], request_config)
        
        response_choice = resp_list[0].choices[0]
        
        token_ids = response_choice.token_ids
        return token_ids
    

    def get_action(self, resample: int = 0):
        """
        Get action from policy with automatic retry on decode failure.
        
        Args:
            resample: Temperature multiplier for initial inference.
                     0 = use greedy decoding initially (top_k=1)
                     >0 = start with temperature = 1.0 * resample
        """
        max_retries = 5
        
        # Initial inference attempt
        if resample == 0:
            # Use default greedy config
            token_ids = self.infer()
            result = self.decode_action_tokens(token_ids)
            initial_temperature = 0.7
        else:
            # Use sampling from the start
            initial_temperature = 1.0 * resample
            token_ids = self.infer(request_config=RequestConfig(
                max_tokens=self.request_config.max_tokens,
                temperature=initial_temperature,
                top_k=50,  # Enable sampling
                top_p=0.9,
                return_details=True,
            ))
            result = self.decode_action_tokens(token_ids)
        
        # Retry with increasing temperature if decode failed
        if result is None:
            temperature = initial_temperature
            retry_count = 0
            
            while result is None and retry_count < max_retries:
                retry_count += 1
                temperature *= 1.5
                
                token_ids = self.infer(request_config=RequestConfig(
                    max_tokens=self.request_config.max_tokens,
                    temperature=temperature,
                    top_k=50,  # Remove greedy constraint for diverse sampling
                    top_p=0.9,
                    return_details=True,
                ))
                result = self.decode_action_tokens(token_ids)
            
            if result is None:
                error_msg = (
                    f"Failed to decode valid actions after {max_retries} retries "
                    f"(resample={resample}, final_temp={temperature:.2f})"
                )
                raise RuntimeError(error_msg)
        
        return result