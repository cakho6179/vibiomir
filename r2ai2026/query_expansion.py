"""Clinical Query Expansion (CQE) and Medical Acronym Disambiguation.

Based on BioASQ and TREC Precision Medicine SOTA practices:
Translates and expands conversational patient queries by extracting clinical
abbreviations, lab test markers, disease acronyms, and symptoms into standardized
medical terms in both Vietnamese and English.

This dramatically improves BM25 keyword matching and dense cross-encoder alignment.
"""

from __future__ import annotations

import re
from typing import Sequence

# Clinical Knowledge Base of Medical Abbreviations & Biomarkers found in ViBioMIR
CLINICAL_ACRONYMS: dict[str, dict[str, str]] = {
    "AFP": {
        "vi": "Alpha-fetoprotein xét nghiệm tầm soát ung thư tế bào gan",
        "en": "Alpha-fetoprotein liver cancer biomarker",
    },
    "ALT": {
        "vi": "Alanine aminotransferase men gan SGPT chức năng gan",
        "en": "Alanine aminotransferase liver function enzyme",
    },
    "AST": {
        "vi": "Aspartate aminotransferase men gan SGOT tổn thương gan tim",
        "en": "Aspartate aminotransferase liver heart enzyme",
    },
    "SGOT": {
        "vi": "Men gan AST tổn thương tế bào gan",
        "en": "Serum glutamic oxaloacetic transaminase AST liver enzyme",
    },
    "SGPT": {
        "vi": "Men gan ALT chức năng gan mật",
        "en": "Serum glutamic pyruvic transaminase ALT liver enzyme",
    },
    "GGT": {
        "vi": "Gamma-glutamyl transferase men gan ứ mật do rượu",
        "en": "Gamma-glutamyl transferase cholestasis liver enzyme",
    },
    "AMH": {
        "vi": "Anti-Mullerian Hormone xét nghiệm đánh giá dự trữ buồng trứng sinh sản",
        "en": "Anti-Mullerian hormone ovarian reserve fertility test",
    },
    "ANA": {
        "vi": "Antinuclear antibody kháng thể kháng nhân bệnh tự miễn Lupus ban đỏ",
        "en": "Antinuclear antibody autoimmune disease systemic lupus",
    },
    "AZF": {
        "vi": "Mất đoạn nhiễm sắc thể Y vô sinh nam không có tinh trùng",
        "en": "Azoospermia factor Y chromosome microdeletion male infertility",
    },
    "BETA-HCG": {
        "vi": "Định lượng Beta HCG nồng độ hormone thai nghén thử thai",
        "en": "Beta human chorionic gonadotropin pregnancy test",
    },
    "HCG": {
        "vi": "Hormone thai nghén Beta HCG xét nghiệm có thai",
        "en": "Human chorionic gonadotropin pregnancy hormone",
    },
    "CA": {
        "vi": "Cancer Antigen dấu ấn ung thư tầm soát khối u",
        "en": "Cancer antigen tumor marker",
    },
    "CMV": {
        "vi": "Cytomegalovirus nhiễm virus cự bào",
        "en": "Cytomegalovirus infection",
    },
    "COVID": {
        "vi": "Nhiễm virus SARS-CoV-2 đường hô hấp cấp",
        "en": "Coronavirus SARS-CoV-2 respiratory infection",
    },
    "CRP": {
        "vi": "C-reactive protein protein phản ứng C chỉ số phản ứng viêm nhiễm khuẩn",
        "en": "C-reactive protein systemic inflammation infection marker",
    },
    "CT": {
        "vi": "Chụp cắt lớp vi tính hình ảnh học chẩn đoán",
        "en": "Computed tomography CT scan diagnostic radiology",
    },
    "MRI": {
        "vi": "Chụp cộng hưởng từ chẩn đoán hình ảnh sọ não cơ xương khớp",
        "en": "Magnetic resonance imaging MRI diagnostic radiology",
    },
    "XQ": {
        "vi": "Chụp X-quang chẩn đoán hình ảnh tim phổi xương",
        "en": "X-ray radiography chest bone imaging",
    },
    "FNA": {
        "vi": "Chọc hút tế bào bằng kim nhỏ sinh thiết khối u nhân tuyến giáp",
        "en": "Fine-needle aspiration cytology biopsy thyroid nodule",
    },
    "HBV": {
        "vi": "Virus viêm gan B HBsAg viêm gan mạn tính",
        "en": "Hepatitis B virus chronic hepatitis HBsAg",
    },
    "HCV": {
        "vi": "Virus viêm gan C Anti-HCV xơ gan",
        "en": "Hepatitis C virus chronic infection",
    },
    "HGB": {
        "vi": "Hemoglobin huyết sắc tố định lượng hồng cầu thiếu máu",
        "en": "Hemoglobin red blood cell anemia level",
    },
    "HIV": {
        "vi": "Virus suy giảm miễn dịch người bệnh AIDS",
        "en": "Human immunodeficiency virus HIV AIDS infection",
    },
    "HP": {
        "vi": "Vi khuẩn Helicobacter pylori viêm loét dạ dày tá tràng",
        "en": "Helicobacter pylori gastric ulcer stomach infection",
    },
    "HPV": {
        "vi": "Human papillomavirus u nhú người sùi mào gà ung thư cổ tử cung",
        "en": "Human papillomavirus genital warts cervical cancer",
    },
    "HSV": {
        "vi": "Herpes simplex virus mụn rộp sinh dục môi miệng",
        "en": "Herpes simplex virus oral genital herpes infection",
    },
    "IUI": {
        "vi": "Bơm tinh trùng vào buồng tử cung hỗ trợ sinh sản thụ tinh nhân tạo",
        "en": "Intrauterine insemination assisted reproduction fertility",
    },
    "IVF": {
        "vi": "Thụ tinh trong ống nghiệm hỗ trợ sinh sản cấy phôi",
        "en": "In vitro fertilization embryo transfer fertility",
    },
    "LDL": {
        "vi": "Low-density lipoprotein cholesterol xấu mỡ máu xơ vữa động mạch",
        "en": "Low-density lipoprotein bad cholesterol atherosclerosis lipid panel",
    },
    "PCR": {
        "vi": "Xét nghiệm sinh học phân tử khuếch đại gen realtime PCR",
        "en": "Polymerase chain reaction molecular diagnostic nucleic acid",
    },
    "PSA": {
        "vi": "Prostate-specific antigen kháng nguyên đặc hiệu tuyến tiền liệt ung thư tiền liệt tuyến",
        "en": "Prostate-specific antigen prostate cancer tumor marker",
    },
    "RPR": {
        "vi": "Xét nghiệm huyết thanh sàng lọc bệnh giang mai",
        "en": "Rapid plasma reagin screening test syphilis infection",
    },
    "TPHA": {
        "vi": "Xét nghiệm khẳng định bệnh giang mai Treponema pallidum",
        "en": "Treponema pallidum hemagglutination confirmation syphilis test",
    },
    "TPPA": {
        "vi": "Xét nghiệm ngưng kết hạt thụ động giang mai Treponema pallidum",
        "en": "Treponema pallidum particle agglutination syphilis test",
    },
    "SCC": {
        "vi": "Squamous Cell Carcinoma Antigen kháng nguyên ung thư tế bào vảy",
        "en": "Squamous cell carcinoma antigen tumor marker",
    },
    "THA": {
        "vi": "Tăng huyết áp bệnh lý cao huyết áp tim mạch",
        "en": "Hypertension high blood pressure cardiovascular disease",
    },
    "TSH": {
        "vi": "Thyroid-stimulating hormone hormone kích thích tuyến giáp suy giáp cường giáp",
        "en": "Thyroid-stimulating hormone hypothyroidism hyperthyroidism test",
    },
    "WBC": {
        "vi": "White blood cell số lượng bạch cầu tổng phân tích tế bào máu sốt nhiễm trùng",
        "en": "White blood cell count leukocyte infection blood test",
    },
}

# Regex for detecting word boundaries of acronyms (case-insensitive for key terms)
_ACRONYM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CLINICAL_ACRONYMS.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def expand_query(
    query: str,
    *,
    lang: str = "vi",
    max_expansions: int = 3,
) -> str:
    """Expand clinical query with disambiguated medical terms and synonyms.

    Args:
        query: Original user query string.
        lang: Target expansion language ('vi' or 'en').
        max_expansions: Maximum number of acronyms to expand in a single query.
    """
    query = (query or "").strip()
    if not query:
        return ""

    matches = _ACRONYM_PATTERN.findall(query)
    if not matches:
        return query

    # Collect unique expansions
    added_terms: list[str] = []
    seen = set()
    for m in matches:
        key = m.upper()
        if key in CLINICAL_ACRONYMS and key not in seen:
            seen.add(key)
            expansion = CLINICAL_ACRONYMS[key].get(lang) or CLINICAL_ACRONYMS[key].get("vi")
            if expansion:
                added_terms.append(expansion)
            if len(added_terms) >= max_expansions:
                break

    if not added_terms:
        return query

    # Append expanded clinical context
    expanded = f"{query} {' '.join(added_terms)}"
    return expanded


def batch_expand_queries(
    queries: Sequence[str],
    *,
    lang: str = "vi",
) -> list[str]:
    """Batch expand a sequence of queries."""
    return [expand_query(q, lang=lang) for q in queries]


# -------------------------------------------------------------------------
# Sub-15B LLM Prompt Templates for Query Rewriting & Expansion (Qwen2.5 / Gemma-2)
# -------------------------------------------------------------------------

def build_hyde_prompt(query_vi: str, target_lang: str = "vi") -> str:
    """Build a prompt for Hypothetical Document Embeddings (HyDE).

    Instructs a sub-15B LLM (e.g. Qwen2.5-7B/14B) to write a detailed, authoritative
    medical clinical answer snippet (120-150 words) that bridges the semantic gap
    between colloquial patient questions and formal clinical literature.
    """
    if target_lang == "zh":
        return (
            "你是一位资深专科临床医生。请根据以下患者的就医咨询，撰写一段约150字的标准医学科普与临床诊断答复。"
            "答复应包括：可能的疾病诊断、病因机制、典型伴随症状、推荐的临床检查项目（如生化、影像）及常用药物治疗方案（含药品通用名）。"
            f"\n\n患者咨询：{query_vi}\n\n医生专业解答（简体中文）："
        )
    return (
        "Bạn là một bác sĩ chuyên khoa giàu kinh nghiệm lâm sàng. Dựa vào câu hỏi tư vấn sức khỏe của người bệnh dưới đây, "
        "hãy viết một đoạn cẩm nang y khoa chuẩn mực (khoảng 120-150 từ) giải đáp chi tiết thắc mắc này. "
        "Đoạn văn cần nêu rõ: chẩn đoán bệnh lý nghi ngờ cao nhất, nguyên nhân bệnh sinh, các xét nghiệm cận lâm sàng cần làm "
        "và phác đồ điều trị, tên hoạt chất thuốc đặc trị chuẩn của Bộ Y tế.\n\n"
        f"Câu hỏi bệnh nhân: {query_vi}\n\n"
        "Đoạn giải đáp y khoa chuyên môn:"
    )


def build_decomposition_prompt(query_vi: str) -> str:
    """Build a prompt to decompose a clinical query into 4 orthogonal search facets.

    Outputs structured JSON with clinical_symptoms, pathology_etiology, pharmacology_drugs, diagnostic_tests.
    """
    return (
        "Bạn là chuyên gia tin học y tế. Hãy phân tích câu hỏi lâm sàng dưới đây và tách thành 4 trường tìm kiếm độc lập "
        "dưới định dạng JSON chính xác:\n"
        "{\n"
        '  "clinical_symptoms": "<danh sách triệu chứng lâm sàng và cảm giác cơ năng>",\n'
        '  "pathology_etiology": "<cơ chế bệnh sinh, nguyên nhân tiềm ẩn và chẩn đoán phân biệt>",\n'
        '  "pharmacology_drugs": "<tên nhóm thuốc, hoạt chất điều trị đặc hiệu>",\n'
        '  "diagnostic_tests": "<các xét nghiệm máu, chẩn đoán hình ảnh hoặc thủ thuật thăm dò>"\n'
        "}\n\n"
        f"Câu hỏi bệnh nhân: {query_vi}\n\n"
        "Chỉ xuất JSON, không giải thích thêm:"
    )


def build_stepback_prompt(query_vi: str) -> str:
    """Build a step-back prompting prompt to abstract specific symptoms into high-level medical pathology."""
    return (
        "Dựa vào câu hỏi cụ thể của người bệnh dưới đây, hãy trừu tượng hóa (Step-Back) thành nguyên lý bệnh học y khoa tổng quát, "
        "cơ chế sinh lý bệnh nền tảng và phân loại hội chứng lâm sàng liên quan trong 2 câu ngắn gọn.\n\n"
        f"Câu hỏi: {query_vi}\n\n"
        "Nguyên lý bệnh học trừu tượng:"
    )

