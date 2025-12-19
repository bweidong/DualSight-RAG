from vllm import LLM, SamplingParams
import os

class VLLMEngine:
    def __init__(self, model_path=""):
        print(f"🚀 Initializing vLLM Engine on GPU 1 & 2...")
        
        #  强制指定卡号
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=2,   # 使用 2 张卡
            trust_remote_code=True,
            gpu_memory_utilization=0.9, # 吃满显存
            dtype="float16",
            enforce_eager=False
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=1024,
            stop=["<|im_end|>"]
        )

    def generate(self, prompt, system_prompt="You are a helpful assistant."):
        """
        Text-Only Inference based on Retrieved Context
      
        """
        # Qwen ChatML 格式
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n"
        full_prompt += f"<|im_start|>assistant\n"

        outputs = self.llm.generate(
            prompts=[full_prompt],
            sampling_params=self.sampling_params
        )
        return outputs[0].outputs[0].text