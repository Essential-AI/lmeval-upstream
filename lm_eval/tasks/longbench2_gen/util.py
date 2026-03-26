import re
from typing import List

def extract_answer(text: str) -> str:
    """
    Extract the answer from the text.
    """
    match = re.search(r"The correct answer is ([A-D]).*", text)
    if match:
        return match.group(1)
    else:
        match = re.search(r"The correct answer is \(([A-D])\).*", text)
        if match:
            return match.group(1)
        else:
            match = re.search(r".*\(([A-D])\).*", text)
            if match:
                return match.group(1)
            else:
                match = re.search(r".*\b([A-D])\b.*", text)
                if match:
                    return match.group(1)
                else:
                    return "[invalid]"

def get_response(results: List[List[str]], docs: List[dict]) -> List[List[str]]:
    return [[extract_answer(res) for res in result] for result in results]
