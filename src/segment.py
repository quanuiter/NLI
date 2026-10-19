"""Tách từ tiếng Việt cho PhoBERT.

PhoBERT được huấn luyện trên văn bản đã tách từ bằng RDRSegmenter của VnCoreNLP,
nên backend mặc định là "vncorenlp" (cần Java). "underthesea" là phương án dự phòng.

Ví dụ: "Tôi là sinh viên trường Đại học Công nghệ Thông tin."
    -> "Tôi là sinh_viên trường Đại_học Công_nghệ Thông_tin ."
"""

from __future__ import annotations

import os
from pathlib import Path

from .data import ROOT

VNCORENLP_DIR = ROOT / "tools" / "vncorenlp"


class Segmenter:
    def __init__(self, backend: str = "vncorenlp", vncorenlp_dir=VNCORENLP_DIR):
        self.backend = backend
        if backend == "vncorenlp":
            import py_vncorenlp

            vncorenlp_dir = Path(vncorenlp_dir).resolve()
            vncorenlp_dir.mkdir(parents=True, exist_ok=True)
            if not (vncorenlp_dir / "VnCoreNLP-1.2.jar").exists():
                py_vncorenlp.download_model(save_dir=str(vncorenlp_dir))
            cwd = os.getcwd()  # py_vncorenlp tự đổi thư mục làm việc, cần đổi lại
            try:
                self._model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(vncorenlp_dir))
            finally:
                os.chdir(cwd)
        elif backend == "underthesea":
            from underthesea import word_tokenize

            self._word_tokenize = word_tokenize
        else:
            raise ValueError('backend phải là "vncorenlp" hoặc "underthesea"')

    def __call__(self, text: str) -> str:
        if not text:
            return ""
        if self.backend == "vncorenlp":
            # word_segment trả về list câu; nối lại thành một chuỗi
            return " ".join(self._model.word_segment(text))
        return self._word_tokenize(text, format="text")
