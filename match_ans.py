import json
import difflib
import re

def match_to_choices(prediction, choices: list[str]) -> str:
    # Handle case where prediction is a list
    if isinstance(prediction, list):
        if prediction:
            pred = str(prediction[0]).lower().strip()
        else:
            pred = ""
    else:
        pred = str(prediction).lower().strip()
    
    lower_choices = [c.lower() for c in choices]
    
    # First, try to extract specific answer patterns
    # Look for "the correct answer is: X" or "Therefore, the correct answer is: X"
    answer_patterns = [
        r'(?:the\s+)?correct\s+answer\s+is:?\s*([^.\n]+)',
        r'therefore,?\s+(?:the\s+)?(?:correct\s+)?answer\s+is:?\s*([^.\n]+)',
        r'answer:?\s*([^.\n]+)',
        r'the\s+answer\s+is:?\s*([^.\n]+)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, pred, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).strip().rstrip('.')
            # Try to match this extracted answer to choices
            for i, choice in enumerate(lower_choices):
                if choice in extracted_answer.lower() or extracted_answer.lower() in choice:
                    return choices[i]
    
    # Special handling for numbers/quantities
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    
    # Extract numbers from prediction
    pred_numbers = set()
    for word, digit in number_words.items():
        if word in pred:
            pred_numbers.add(digit)
    
    # Also look for actual digits
    digit_matches = re.findall(r'\b\d+\b', pred)
    pred_numbers.update(digit_matches)
    
    # Check if any choice matches the extracted numbers
    for choice in choices:
        choice_lower = choice.lower()
        # Check if choice contains any of the predicted numbers
        for num in pred_numbers:
            if num in choice_lower or number_words.get(choice_lower) == num:
                return choice
    
    # Use fuzzy matching with higher threshold
    matches = difflib.get_close_matches(pred, lower_choices, n=1, cutoff=0.6)
    if matches:
        idx = lower_choices.index(matches[0])
        return choices[idx]
    
    # Check each choice individually for substring matches
    # Prioritize longer matches
    best_match = None
    best_score = 0
    
    for i, choice in enumerate(lower_choices):
        # Check if choice appears in prediction
        if choice in pred:
            score = len(choice)  # Longer matches get higher scores
            if score > best_score:
                best_score = score
                best_match = choices[i]
        
        # Also check if prediction contains the choice
        elif any(word in choice for word in pred.split() if len(word) > 2):
            score = sum(len(word) for word in pred.split() if word in choice)
            if score > best_score:
                best_score = score
                best_match = choices[i]
    
    if best_match:
        return best_match
    
    # Fallback: return first choice
    return choices[0] if choices else ""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Match predictions to choices in a JSON file.")
    parser.add_argument("--input", type=str, help="Path to the input JSON file")
    parser.add_argument("--output", type=str, help="Path to save the output JSON file")
    args = parser.parse_args()
    ans_file = "mmau-test-mini.json"
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(ans_file, "r", encoding="utf-8") as f:
        ans_data = json.load(f)
    ans_data = {str(item["id"]): item["answer"] for item in ans_data}
    for item in data:
        prediction = item.get("desta_response", "")
        choices = item.get("choices", [])
        if prediction and choices:
            matched = match_to_choices(prediction, choices)
            print(f"Matching prediction '{prediction[:100]}...' to choices: {choices} -> Matched: {matched}")
            item["prediction"] = matched  
        item["answer"] = ans_data.get(str(item["id"]), "")

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ finished matching predictions to choices and saved to {args.output}")   

    # calculate accuracy
    correct = sum(1 for item in data if item.get("prediction") == item.get("answer"))
    total = len(data)
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")