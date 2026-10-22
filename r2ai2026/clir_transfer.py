"""Cross-Lingual Information Retrieval (CLIR) Knowledge Transfer Engine.

Transfers Vietnamese telehealth queries into simplified Chinese medical nomenclature
(for querying 240,000 Chinese clinical Q&A articles: 120ask, 39.net, familydoctor)
and English MeSH clinical terminology (for PubMed).
"""

from __future__ import annotations

import re
from typing import Sequence

# High-frequency Vietnamese-to-Chinese medical terminology mapping
VI_TO_ZH_MEDICAL_MAP: dict[str, str] = {
    # Gastrointestinal / Tiêu hóa
    "dạ dày": "胃",
    "đau dạ dày": "胃痛 胃溃疡",
    "viêm loét dạ dày": "胃溃疡",
    "loét hành tá tràng": "十二指肠溃疡",
    "tá tràng": "十二指肠",
    "trào ngược dạ dày": "胃食管反流",
    "trào ngược thực quản": "胃食管反流病",
    "vi khuẩn hp": "幽门螺杆菌",
    "ợ chua": "反酸",
    "ợ hơi": "嗳气",
    "đau thượng vị": "上腹痛 胃脘痛",
    "xuất huyết tiêu hóa": "消化道出血",
    "đi ngoài ra máu": "便血",
    "tiêu chảy": "腹泻",
    "táo bón": "便秘",
    "đầy hơi": "腹胀",
    
    # Liver / Gan mật
    "men gan": "转氨酶 肝功能",
    "men gan cao": "转氨酶升高 肝功能异常",
    "viêm gan b": "乙型肝炎 乙肝",
    "viêm gan c": "丙型肝炎 丙肝",
    "xơ gan": "肝硬化",
    "gan nhiễm mỡ": "脂肪肝",
    "vàng da": "黄疸",
    "túi mật": "胆囊",
    "sỏi mật": "胆结石",
    
    # Respiratory / Hô hấp
    "ho": "咳嗽",
    "ho có đờm": "咳嗽咳痰",
    "ho ra máu": "咯血",
    "viêm họng": "咽喉炎",
    "viêm amidan": "扁桃体炎",
    "viêm xoang": "鼻窦炎",
    "viêm phế quản": "支气管炎",
    "viêm phổi": "肺炎",
    "khó thở": "呼吸困难 气短",
    "sốt": "发热 发烧",
    "sốt cao": "高热",
    
    # Dental / Răng miệng
    "chảy máu chân răng": "牙龈出血",
    "chảy máu răng": "牙龈出血",
    "răng": "牙齿 牙龈",
    "viêm lợi": "牙龈炎",
    "sâu răng": "龋齿 蛀牙",
    "nhiệt miệng": "口腔溃疡",
    
    # Oncology & Biomarkers / Ung bướu & Xét nghiệm
    "ung thư": "癌症 恶性肿瘤",
    "khối u": "肿瘤 肿块",
    "scc": "鳞状细胞癌抗原 SCC",
    "beta-hcg": "人绒毛膜促性腺激素 HCG",
    "beta hcg": "HCG 早期妊娠",
    "amh": "抗缪勒氏管激素 AMH 卵巢储备",
    "afp": "甲胎蛋白 AFP 原发性肝癌",
    "cea": "癌胚抗原 CEA",
    "ca 125": "糖类抗原 CA125 卵巢癌",
    "tpha": "梅毒螺旋体特异抗体 TPHA",
    "sinh thiết": "病理活检",
    "fna": "细针穿刺细胞学检查",
    
    # Cardiology / Tim mạch
    "huyết áp": "血压",
    "tăng huyết áp": "高血压",
    "huyết áp cao": "高血压",
    "đau ngực": "胸痛",
    "nhồi máu cơ tim": "心肌梗死",
    "suy tim": "心力衰竭",
    "rối loạn nhịp tim": "心律失常",
    
    # Nephrology / Thận tiết niệu
    "suy thận": "肾功能不全 肾衰竭",
    "sỏi thận": "肾结石",
    "tiểu đêm": "夜尿增多",
    "tiểu buốt": "尿痛 尿急",
    "tiểu rắt": "尿频 尿急",
    
    # Endocrinology / Nội tiết
    "tiểu đường": "糖尿病",
    "đường huyết": "血糖",
    "tuyến giáp": "甲状腺",
    "bướu cổ": "甲状腺肿",
    "cường giáp": "甲状腺功能亢进 甲亢",
    "suy giáp": "甲状腺功能减退 甲减",
    
    # Dermatology / Da liễu
    "mụn": "痤疮 青春痘",
    "ngứa": "皮肤瘙痒",
    "mề đay": "荨麻疹",
    "dị ứng": "过敏",
    "vảy nến": "银屑病",
    "zona": "带状疱疹",
    
    # General queries / Ý định chung
    "uống thuốc gì": "用药指南 药物治疗",
    "điều trị": "治疗方案",
    "nguyên nhân": "病因",
    "triệu chứng": "临床表现 症状",
    "có nguy hiểm không": "危害 预后",
    "xét nghiệm gì": "检查项目 诊断",
    "ăn gì": "饮食禁忌 食疗",
    "kiêng gì": "注意事项 禁忌",
}


def transfer_vi_to_zh(query_vi: str) -> str:
    """Transfer a Vietnamese clinical query to Chinese medical search terms."""
    q_lower = query_vi.lower()
    zh_terms: list[str] = []
    
    for vi_kw, zh_equiv in VI_TO_ZH_MEDICAL_MAP.items():
        if vi_kw in q_lower:
            zh_terms.append(zh_equiv)
            
    if not zh_terms:
        # Fallback to general inquiry keyword
        zh_terms.append("疾病 诊断 治疗")
        
    return " ".join(dict.fromkeys(zh_terms))


def enrich_query_multilingual(query_vi: str, query_en: str = "") -> dict[str, str]:
    """Return enriched representations for all 3 languages."""
    return {
        "query_vi": query_vi,
        "query_zh": transfer_vi_to_zh(query_vi),
        "query_en": query_en or "",
    }
