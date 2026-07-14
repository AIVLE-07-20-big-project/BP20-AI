import os
import google.generativeai as genai
from openai import OpenAI
from app.schemas.review import ReviewAnalyzeResponse

class ABSAService:
    def __init__(self):
        self.provider = os.environ.get("AI_PROVIDER", "GEMINI").upper()

        if self.provider == "OPENAI":
            self.openai_client = OpenAI()
            self.model_name = "gpt-4o-mini"
        else:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not exist")
            genai.configure(api_key=api_key)

    def _call_llm(self, text:str, rag_context: str = "") -> ReviewAnalyzeResponse:
        """다음 역할을 토대로 리뷰를 분석하는 요청"""

        system_prompt = (
            "너는 쇼핑몰 리뷰를 분석하는 전문 AI 에이전트야.\n"
            "리뷰에서 언급된 속성(aspect)과 감정(sentiment)을 추출해줘.\n"
            "단, 문맥에 모르는 신조어, 밈, 혹은 중의적 표현이 있어서 확실하게 분석하기 어렵다면 "
            "고민하지 말고 `needs_rag` 필드를 True로 설정해서 대답해줘."
        )

        if rag_context:
            system_prompt += f"\n\n[참고 지식(RAG)]:\n{rag_context}\n위 지식을 바탕으로 신조어나 난해한 문맥을 파악해."
        
        if self.provider == "OPENAI":
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format=ReviewAnalyzeResponse
            )
            return completion.choices[0].message.parsed
        else:
            gemini_model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=system_prompt
            )

            response = gemini_model.generate_content(
                contents=text,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ReviewAnalyzeResponse
                )
            )
            return ReviewAnalyzeResponse.model_validate_json(response.text)

    def analyze_review(self, text: str, review_id: int) -> ReviewAnalyzeResponse:
        result = self._call_llm(text)
        result.review_id = review_id

        if result.needs_rag:
            print("모르는 단어가 있습니다. rag를 참고합니다")
            
            # Vector DB 구축 후 코드 추가
        
        return result