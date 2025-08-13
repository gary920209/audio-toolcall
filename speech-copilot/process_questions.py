import json
import os

def load_json_file(filepath):
    """Load JSON data from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_answer_from_desta(audio_id, question, choices):
    """
    This function would interface with DESTA to get answers.
    For now, it's a placeholder that returns the first choice.
    You'll need to replace this with actual DESTA API calls.
    """
    # TODO: Implement actual DESTA API call here
    # This is a placeholder implementation
    print(f"Processing: {audio_id}")
    print(f"Question: {question}")
    print(f"Choices: {choices}")
    
    # Placeholder: return first choice as answer
    if choices and len(choices) > 0:
        return choices[0]
    return "No answer available"

def process_all_questions():
    """Process all questions in the JSON file"""
    # Load the original JSON file
    input_file = "/Users/garylee/Desktop/project/speech-copilot/mmau-test-mini.json"
    output_file = "/Users/garylee/Desktop/project/speech-copilot/mmau-test-mini-answers.json"
    
    try:
        data = load_json_file(input_file)
        print(f"Loaded {len(data)} questions from {input_file}")
        
        # Process each question
        results = []
        for i, item in enumerate(data):
            print(f"\nProcessing question {i+1}/{len(data)}")
            
            # Extract required fields
            audio_id = item.get("audio_id", "")
            question = item.get("question", "")
            choices = item.get("choices", [])
            
            # Get answer from DESTA
            predicted_answer = get_answer_from_desta(audio_id, question, choices)
            
            # Create result entry
            result = {
                "id": item.get("id", f"question_{i+1}"),
                "audio_id": audio_id,
                "question": question,
                "choices": choices,
                "original_answer": item.get("answer", ""),
                "predicted_answer": predicted_answer,
                "dataset": item.get("dataset", ""),
                "task": item.get("task", ""),
                "category": item.get("category", ""),
                "sub-category": item.get("sub-category", ""),
                "difficulty": item.get("difficulty", "")
            }
            
            results.append(result)
        
        # Save results to new JSON file
        save_json_file(results, output_file)
        print(f"\nProcessing complete! Results saved to {output_file}")
        print(f"Total questions processed: {len(results)}")
        
        return results
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_file}")
        return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def main():
    """Main function to run the processing"""
    print("Starting DESTA question processing...")
    results = process_all_questions()
    
    if results:
        print(f"\nSample result:")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
