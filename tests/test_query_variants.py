import json

import pandas as pd

from r2ai2026.stages import query_variants as qv

TABLE = {"acronyms": {"ALT": {"vi": "men gan SGPT", "en": "alanine aminotransferase"}},
         "vi_zh": {"dạ dày": "胃", "đau dạ dày": "胃痛 胃溃疡"}}


def test_table_only_variants():
    rows = qv.variants(7, "Chỉ số ALT cao và đau dạ dày", None, TABLE)
    got = {name: (lang, text) for _, name, lang, text in rows}
    assert got["orig"] == ("vi", "Chỉ số ALT cao và đau dạ dày")
    assert got["acro"][1].endswith("men gan SGPT")
    assert got["en_kw"] == ("en", "alanine aminotransferase")
    assert got["zh_kw"] == ("zh", "胃痛 胃溃疡 胃")  # longest VI phrase first, terms deduplicated
    assert "kw" not in got and "zh_q" not in got  # empty frame fields are dropped


def test_acronym_needs_word_boundary():
    assert qv.acronyms_in("SALT and ALTA", TABLE["acronyms"]) == []
    assert qv.acronyms_in("ALT/AST", TABLE["acronyms"]) == ["ALT"]


def test_frame_fields_and_dedup():
    frame = {"entities": json.dumps([{"text": "viêm gan", "aliases": ["hepatitis", "viêm gan siêu vi"]}]),
             "subqueries": json.dumps(["viêm gan là gì?", "viêm gan là gì?"]),
             "keywords": "viêm gan", "keywords_en": "hepatitis", "query_zh": "肝炎是什么", "keywords_zh": "肝炎",
             "hyde": "Viêm gan là tình trạng ..."}
    rows = qv.variants(1, "viêm gan", frame, TABLE)
    names = [r[1] for r in rows]
    assert names == ["orig", "sub0", "hyde", "en_kw", "zh_q", "zh_kw"]  # kw == orig and sub1 == sub0 dropped
    assert dict((r[1], r[3]) for r in rows)["en_kw"] == "hepatitis"  # keyword and alias merged once
    assert qv.unique_terms("ALT liver", "alt", "Liver enzyme") == "ALT liver enzyme"


def test_build_joins_frame_by_qid():
    queries = pd.DataFrame({"id": [1, 2], "query": ["đau dạ dày", "ho"]})
    frames = pd.DataFrame({"qid": [2], "keywords": ["ho khan"]})
    df = qv.build(queries, frames, TABLE)
    assert list(df.columns) == ["qid", "variant", "lang", "text"]
    assert set(df[df.qid == 2]["variant"]) == {"orig", "kw"}
    assert set(df[df.qid == 1]["variant"]) == {"orig", "zh_kw"}


def test_query_frame_normalise_new_fields():
    from r2ai2026.stages import query_frame as qf
    obj = {"entities": [{"text": "ALT", "aliases": "alanine aminotransferase, SGPT"}],
           "keywords_en": ["ALT", "liver enzyme"], "query_zh": " ALT高是什么意思 "}
    f = qf.normalise(obj, "ALT cao")
    assert f["entities"][0]["aliases"] == ["alanine aminotransferase", "SGPT"]
    assert f["keywords_en"] == "ALT liver enzyme" and f["query_zh"] == "ALT高是什么意思"
    assert set(qf.empty_frame()) == set(qf.FIELDS)


def test_query_frame_parse_skips_reasoning():
    from r2ai2026.stages import query_frame as qf
    raw = 'Thinking Process: use {braces} here.\n{"entities": [], "keywords": "x", "hyde": "y"}'
    assert qf.parse(raw)["keywords"] == "x"
