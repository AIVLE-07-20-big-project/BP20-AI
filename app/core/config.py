import os

class Settings:
    PROJECT_NAME: str = "RoBERTa ABSA API Service"
    VERSION: str = "1.0.0"
    
    # 모델 폴더 경로 (학습 완료된 roberta_absa_best_4class 경로)
    MODEL_PATH: str = "./roberta_absa_best_4class"
    
    # ABSA 분석 속성 및 감성 매핑
    ASPECTS: list = ["food", "service", "convenience", "price", "atmosphere"]
    LABEL_MAP: dict = {0: "부정", 1: "중립", 2: "긍정", 3: "none"}
    MAX_LENGTH: int = 128

settings = Settings()