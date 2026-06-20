import json
import re
from typing import List, Dict, Tuple, Optional, AsyncIterator
import numpy as np
import httpx
from fastapi import HTTPException

from app.models.db import mem

from app.config import (
    ENABLED_EMOTION_MODELS,
    DEFAULT_LLM_MODEL,
    LLM_BACKEND,
    LLAMA_SERVER_BASE_URL,
)

class LLMService:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or LLAMA_SERVER_BASE_URL).rstrip("/")
        self.backend = (LLM_BACKEND or "llama_cpp").strip().lower()
        self.default_model = DEFAULT_LLM_MODEL
        self.chat_completions_url = f"{self.base_url}/v1/chat/completions"
        self.models_url = f"{self.base_url}/v1/models"

    async def get_available_models(self) -> List[Dict]:
        """Lấy danh sách model từ llama-server HTTP API."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.models_url)
                response.raise_for_status()
                payload = response.json()
                return payload.get("data", payload.get("models", []))
        except Exception as e:
            print(f"llama-server connection error: {e}")
            return []

    async def get_available_emotion_models(self) -> List[str]:
        return ENABLED_EMOTION_MODELS

    def _strip_markdown_fence(self, text: str) -> str:
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return raw

    def _extract_json_payload(self, text: str) -> Optional[Dict]:
        candidate = self._strip_markdown_fence(text)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        start = candidate.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(candidate)):
            ch = candidate[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = candidate[start:i + 1]
                    try:
                        parsed = json.loads(snippet)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        return None
        return None

    def _parse_ai_response(self, text: str) -> Tuple[str, str]:
        raw = text.strip()
        payload = self._extract_json_payload(raw)
        if payload is not None:
            emotion = str(payload.get("Emotion") or payload.get("emotion") or "Bình thường").strip()
            advice = str(payload.get("Response") or payload.get("response") or "").strip()
            if advice:
                return emotion or "Bình thường", advice
            return emotion or "Bình thường", raw

        emotion_match = re.search(r"\"?Emotion\"?\s*:\s*\"?([^\"\n]+)\"?", raw, re.IGNORECASE)
        response_match = re.search(r"\"?Response\"?\s*:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)

        emotion = emotion_match.group(1).strip() if emotion_match else "Bình thường"
        advice = response_match.group(1).strip().strip('"') if response_match else raw

        return emotion, advice

    def _build_messages(self, message: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": """Bạn là một chuyên gia tâm lý học AI chuyên sâu về hỗ trợ tinh thần. 
Nhiệm vụ của bạn là nhận diện trạng thái cảm xúc từ lời tâm sự và phản hồi thấu cảm.

YÊU CẦU BẮT BUỘC:
1. Luôn luôn trả về kết quả dưới định dạng JSON duy nhất. KHÔNG kèm theo lời dẫn giải ngoài JSON.
2. Cấu trúc JSON phải chính xác như sau:
{
  "Emotion": "Tên cảm xúc",
  "Response": "Nội dung phản hồi"
}

QUY TẮC NỘI DUNG:
- Trường 'Emotion': Chỉ được chọn từ danh sách sau: [Buồn bã, Lo âu, Lạc quan, Cô đơn, Other, Vui vẻ, Chán ghét, Ngạc nhiên, Sợ hãi, Tức giận, Highly Negative, Trung lập, Hối tiếc].
- Trường 'Response': 
  + Sử dụng đại từ 'mình' - 'bạn'.
  + Phải bắt đầu bằng việc gọi tên và xác nhận cảm xúc của người dùng (Validation).
  + Không đưa ra lời khuyên sáo rỗng hoặc phán xét.

VÍ DỤ ĐẦU RA:
{
  "Emotion": "Lo âu",
  "Response": "Mình hiểu là bạn đang cảm thấy rất lo âu về những dự định sắp tới. Việc đối mặt với những điều chưa rõ ràng quả thực không hề dễ dàng chút nào..."
}
""",
            },
            {"role": "user", "content": message},
        ]

    async def _post_chat_completion(self, message: str, model_name: Optional[str] = None, stream: bool = False):
        payload = {
            "model": model_name or self.default_model,
            "messages": self._build_messages(message),
            "temperature": 0.2,
            "max_tokens": 512,
            "stream": stream,
        }
        async with httpx.AsyncClient(timeout=None if stream else 60.0) as client:
            if stream:
                async with client.stream("POST", self.chat_completions_url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break
                            yield data
            else:
                response = await client.post(self.chat_completions_url, json=payload)
                response.raise_for_status()
                yield response.json()

    async def generate_response(self, message: str, user_id: Optional[str] = None, conversation_id: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> Dict:
        """Tạo phản hồi có tích hợp Memori (Non-stream)

        Hỗ trợ hai cách gọi:
        - Legacy: generate_response(message, model_name)
        - New: generate_response(message, user_id, conversation_id, model_name)
        """
        # Backward compatibility: if caller passed second positional arg (legacy),
        # it will be mapped to `user_id` here — treat that as `model_name` when
        # `conversation_id` and `model_name` are not provided.
        if conversation_id is None and model_name is None and user_id is not None:
            model_name = user_id
            user_id = "guest_user"
        user_id = user_id or "guest_user"

        selected_model = model_name or self.default_model

        try:
            raw_text = ""
            async for result in self._post_chat_completion(message, selected_model, stream=False):
                raw_text = result["choices"][0]["message"]["content"]

            emotion, advice = self._parse_ai_response(raw_text)

            return {
                "status": "success",
                "emotion": emotion,
                "advice": advice,
                "model_used": selected_model,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"LLM/Memori Error: {str(e)}"
            }

    async def generate_response_stream(self, message: str, user_id: Optional[str] = None, conversation_id: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> AsyncIterator[Dict]:
        """Tạo phản hồi stream có tích hợp Memori

        Hỗ trợ legacy call pattern: generate_response_stream(message, model_name)
        and new pattern: generate_response_stream(message, user_id, conversation_id, model_name)
        """
        # Backward compatibility: if caller passed model as second positional arg,
        # it will be mapped to `user_id` here — treat that as `model_name`.
        if conversation_id is None and model_name is None and user_id is not None:
            model_name = user_id
            user_id = "guest_user"
        user_id = user_id or "guest_user"

        selected_model = model_name or self.default_model
        aggregated_text = ""

        try:
            async for data in self._post_chat_completion(message, selected_model, stream=True):
                try:
                    payload = json.loads(data)
                    delta = payload.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        aggregated_text += content
                        yield {"type": "chunk", "content": content}
                except Exception:
                    continue

            emotion, advice = self._parse_ai_response(aggregated_text)
            yield {
                "type": "final",
                "emotion": emotion,
                "advice": advice,
                "raw_text": aggregated_text,
                "show_emotion": True,
                "reliability_score": 1.0,
                "model_used": selected_model,
            }
        except Exception as e:
            yield {
                "type": "error",
                "message": f"LLM/Memori Error: {str(e)}",
            }

    async def calculate_cosine_similarity_between_two_labels(self, label_a: str, label_b: str, embedder) -> float:
        vec_a = embedder.encode([label_a])[0]
        vec_b = embedder.encode([label_b])[0]
        
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

# Khởi tạo instance
llm_service = LLMService()